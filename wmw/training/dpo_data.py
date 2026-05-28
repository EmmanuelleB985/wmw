from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from wmw.datasets.common import EvalExample


def trace_to_prompt(trace: dict, image_path: str | None = None) -> list[dict]:
    question = trace.get("question", "")
    msgs = [
        {"role": "system", "content":
         "You are a physics expert. Analyze the physical scene step by step: "
         "initial state, transition (physical law + effect), resulting state, "
         "a short derivation, then final answer. Respond only with valid JSON."},
    ]
    user_content: list[dict] = []
    if image_path and Path(image_path).exists():
        user_content.append({"type": "image", "image": image_path})
    schema_snippet = (
        "Respond ONLY with a JSON object with fields: state_0, transition, "
        "state_1, derivation, answer. Do not include facts not visible unless "
        "listed as assumptions."
    )
    user_content.append({
        "type": "text",
        "text": f"Question: {question}\n\nReason step by step through the physics.\n\n{schema_snippet}",
    })
    msgs.append({"role": "user", "content": user_content})
    return msgs


def trace_to_completion(trace: dict) -> str:
    keep = {
        "state_0": trace.get("state_0", {}),
        "transition": trace.get("transition", {}),
        "state_1": trace.get("state_1", {}),
        "derivation": trace.get("derivation", ""),
        "answer": trace.get("answer", {}),
    }
    return json.dumps(keep, separators=(",", ": "))


def build_dpo_record(
    pair: dict,
    image_root: Path | None = None,
) -> dict | None:
    chosen = pair.get("chosen") or {}
    rejected = pair.get("rejected") or {}
    if not chosen or not rejected:
        return None


    img_path = None
    if image_root:
        cand = image_root / f"{pair.get('source_trace_id')}.png"
        if cand.exists():
            img_path = str(cand)

    prompt_msgs = trace_to_prompt(chosen, image_path=img_path)
    return {
        "id": pair["id"],
        "source_trace_id": pair["source_trace_id"],
        "perturbation_type": pair["perturbation_type"],
        "perturbation_family": pair["perturbation_family"],
        "image": img_path,


        "prompt": json.dumps(prompt_msgs),
        "messages": prompt_msgs,
        "chosen": trace_to_completion(chosen),
        "rejected": trace_to_completion(rejected),
    }


def build_dpo_split(
    pairs_path: Path,
    splits_path: Path,
    output_dir: Path,
    image_root: Path | None = None,
) -> dict[str, int]:
    splits_raw = json.load(open(splits_path))


    if "pair_splits" in splits_raw:
        pair_splits = {"train": set(), "val": set(), "test": set()}
        for pid, split in splits_raw["pair_splits"].items():
            if split in pair_splits:
                pair_splits[split].add(pid)
    elif "splits" in splits_raw:
        pair_splits = {
            "train": set(splits_raw["splits"]["train"]["pair_ids"]),
            "val":   set(splits_raw["splits"]["val"]["pair_ids"]),
            "test":  set(splits_raw["splits"]["test"]["pair_ids"]),
        }
    else:
        raise KeyError(
            f"Could not locate pair splits in {splits_path}; "
            f"expected 'pair_splits' or 'splits' key, got {list(splits_raw)}"
        )


    counts = {"train": 0, "val": 0, "test": 0}
    output_dir.mkdir(parents=True, exist_ok=True)
    out_files = {k: open(output_dir / f"{k}.jsonl", "w") for k in counts}

    with open(pairs_path) as f:
        for line in f:
            pair = json.loads(line)
            rec = build_dpo_record(pair, image_root=image_root)
            if rec is None:
                continue
            for split_name, ids in pair_splits.items():
                if pair["id"] in ids:
                    if split_name == "train" and pair.get("perturbation_family") == "held_out":

                        continue
                    out_files[split_name].write(json.dumps(rec) + "\n")
                    counts[split_name] += 1
                    break

    for f in out_files.values():
        f.close()
    return counts
