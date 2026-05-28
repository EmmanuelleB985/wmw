#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from wmw.datasets.common import load_examples
from wmw.evaluation.open_vlm import (
    OPEN_VLM_REGISTRY, open_vlm_config,
    wait_for_server, check_server_model, ensure_local_key,
)
from wmw.evaluation.prompts import build_prompt, SYSTEM_PROMPTS
from wmw.evaluation.vlm_caller import call_vlm
from wmw.evaluation.trace_parser import parse_trace, extract_answer, answers_match
from wmw.verifiers.pipeline import verify_trace


def stage_answer_validity(config, examples) -> dict:
    n, n_correct, n_valid, n_parsed = 0, 0, 0, 0
    per_example = []
    for ex in examples:
        prompt = build_prompt(ex, condition="full_trace")
        resp = call_vlm(config, SYSTEM_PROMPTS["full_trace"], prompt, ex.image_path)
        td, status = parse_trace(resp.raw_text)
        pred = extract_answer(td)
        correct = answers_match(
            pred, ex.gold_answer,
            options=ex.options if hasattr(ex, "options") else None,
        )

        if td is not None:
            td.setdefault("id", ex.id)
            td.setdefault("scenario_family", ex.topic or "unknown")
            td.setdefault("question", ex.question)
            td.setdefault("metadata", {"source": "model_generated"})
            vr = verify_trace(td)
            valid = bool(vr.all_ok)
        else:
            valid = False
        n += 1
        n_correct += int(correct)
        n_valid += int(valid)
        n_parsed += int(status != "failed")
        per_example.append({
            "id": ex.id, "correct": correct, "valid": valid,
            "predicted": str(pred), "gold": str(ex.gold_answer),
        })
    return {
        "n": n,
        "answer_accuracy": n_correct / max(n, 1),
        "trace_validity": n_valid / max(n, 1),
        "parse_rate": n_parsed / max(n, 1),
        "per_example": per_example,
    }


def stage_preference_accuracy(
    config, pairs_path: Path, only_family: str | None = None,
) -> dict:
    import urllib.request, base64
    n, correct = 0, 0
    per_pair = []
    with open(pairs_path) as f:
        for line in f:
            pair = json.loads(line)
            if only_family and pair.get("perturbation_family") != only_family:
                continue
            chosen = pair["chosen"]; rejected = pair["rejected"]

            def short(t):
                return json.dumps({
                    "state_0": t.get("state_0"),
                    "transition": t.get("transition"),
                    "state_1": t.get("state_1"),
                    "answer": t.get("answer"),
                }, separators=(",", ": "))

            import random
            order = random.random() < 0.5
            a, b = (chosen, rejected) if order else (rejected, chosen)
            q = chosen.get("question", "")
            user_text = (
                f"Question: {q}\n\n"
                f"Two candidate physical traces are given.\n"
                f"Trace A:\n{short(a)}\n\n"
                f"Trace B:\n{short(b)}\n\n"
                f"Which trace is physically consistent and answer-faithful? "
                f"Respond with only the JSON {{\"choice\":\"A\"}} or {{\"choice\":\"B\"}}."
            )
            resp = call_vlm(config, "You are a careful physics auditor. Answer with JSON only.", user_text)
            try:
                choice = json.loads(resp.raw_text)["choice"].strip().upper()
            except Exception:

                txt = (resp.raw_text or "").strip().upper()
                choice = "A" if "A" in txt and "B" not in txt else ("B" if "B" in txt else "?")
            correct_choice = "A" if order else "B"
            n += 1
            if choice == correct_choice:
                correct += 1
            per_pair.append({
                "id": pair["id"],
                "perturbation_type": pair["perturbation_type"],
                "perturbation_family": pair["perturbation_family"],
                "model_choice": choice,
                "correct": choice == correct_choice,
            })
    return {
        "n_pairs": n,
        "preference_accuracy": correct / max(n, 1),
        "per_pair": per_pair,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True, help="Open VLM key")
    p.add_argument("--adapter-path", default=None,
                   help="LoRA adapter path; if None, evaluates base model")
    p.add_argument("--server-url", default=None)
    p.add_argument("--data", default="data/eval_data/merged_eval.jsonl")
    p.add_argument("--pairs", default="data/preference_pairs_seed.jsonl")
    p.add_argument("--splits", default="data/splits.json")
    p.add_argument("--max-examples", type=int, default=200)
    p.add_argument("--output", default=None,
                   help="Output JSON; default = data/results/{name}_dpo_eval.json")
    args = p.parse_args()

    ensure_local_key()
    config = open_vlm_config(args.base_model, base_url=args.server_url)
    name = (Path(args.adapter_path).name if args.adapter_path else args.base_model) or args.base_model
    print(f"\n══ DPO model eval: {config.name} (adapter: {name}) ══")

    if not wait_for_server(config.base_url):
        print(f"ERROR: vLLM server unreachable at {config.base_url}")
        sys.exit(2)


    if args.adapter_path:

        config.model_id = Path(args.adapter_path).name


    examples = load_examples(Path(args.data))
    if args.max_examples > 0:
        examples = examples[:args.max_examples]
    print(f"\n  ── Answer + validity on {len(examples)} examples ──")
    av = stage_answer_validity(config, examples)
    print(f"    answer_acc={av['answer_accuracy']:.1%}  "
          f"trace_valid={av['trace_validity']:.1%}  "
          f"parse={av['parse_rate']:.1%}")


    print(f"\n  ── Preference acc on HELD-OUT pertubations ──")
    held = stage_preference_accuracy(config, Path(args.pairs), only_family="held_out")
    print(f"    held_out_pref_acc={held['preference_accuracy']:.1%}  "
          f"(n={held['n_pairs']})")


    print(f"\n  ── Preference acc on SEEN perturbations ──")
    seen = stage_preference_accuracy(config, Path(args.pairs), only_family="seen")
    print(f"    seen_pref_acc={seen['preference_accuracy']:.1%}  "
          f"(n={seen['n_pairs']})")

    out = {
        "model": config.name,
        "adapter": args.adapter_path,
        "answer_validity": av,
        "held_out_preference": held,
        "seen_preference": seen,
    }
    output = args.output or f"data/results/{name}_dpo_eval.json"
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n══ Saved → {output} ══")


if __name__ == "__main__":
    main()
