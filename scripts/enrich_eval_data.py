#!/usr/bin/env python3

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wmw.datasets.common import EvalExample, save_examples, load_examples


def link_synthetic_diagrams(
    eval_path: Path,
    diagram_dir: Path,
) -> int:
    exs = load_examples(eval_path)
    linked = 0

    for ex in exs:
        if ex.source != "synthetic":
            continue
        if ex.image_path and Path(ex.image_path).exists():
            continue


        trace_id = ex.id
        candidates = list(diagram_dir.glob(f"{trace_id}.png"))
        if not candidates:

            family = ex.topic or ex.extra.get("trace", {}).get("scenario_family", "")
            if family:
                candidates = list(diagram_dir.glob(f"{family}_*.png"))

        if candidates:
            ex.image_path = str(candidates[0])
            linked += 1

    if linked > 0:
        save_examples(exs, eval_path)
    return linked


def enrich_clevrer_with_scenes(eval_path: Path) -> int:
    exs = load_examples(eval_path)
    enriched = 0

    for ex in exs:
        if ex.source != "clevrer":
            continue
        if ex.image_path and Path(ex.image_path).exists():
            continue


        scene_desc = ex.extra.get("scene_description", "")
        if scene_desc and scene_desc not in ex.question:
            ex.question = f"[Scene: {scene_desc.strip()}] {ex.question}"
            enriched += 1

    if enriched > 0:
        save_examples(exs, eval_path)
    return enriched


def main():
    eval_path = Path("data/eval_data/merged_eval.jsonl")
    if not eval_path.exists():
        print(f"ERROR: {eval_path} not found. Run download_datasets.py first.")
        sys.exit(1)


    hf_diagrams = Path("data/hf_release/images")
    local_diagrams = Path("data/diagrams")

    diagram_dir = None
    if hf_diagrams.exists() and any(hf_diagrams.glob("*.png")):
        diagram_dir = hf_diagrams
    elif local_diagrams.exists() and any(local_diagrams.glob("*.png")):
        diagram_dir = local_diagrams

    if diagram_dir:
        n = link_synthetic_diagrams(eval_path, diagram_dir)
        print(f"Linked {n} synthetic diagrams from {diagram_dir}")
    else:
        print("No diagram directory found. Run build_hf_dataset.py first to generate diagrams.")
        print("  Or generate diagrams only:")
        print("  python -c \"from wmw.diagrams.renderer import render_trace_diagram; ...\"")


    n = enrich_clevrer_with_scenes(eval_path)
    print(f"Enriched {n} CLEVRER examples with scene descriptions")


    exs = load_examples(eval_path)
    from collections import Counter
    status = Counter()
    for ex in exs:
        if ex.image_path and Path(ex.image_path).exists():
            status[f"{ex.source}_has_image"] += 1
        else:
            status[f"{ex.source}_no_image"] += 1

    print(f"\nImage availability after enrichment:")
    for k, v in sorted(status.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
