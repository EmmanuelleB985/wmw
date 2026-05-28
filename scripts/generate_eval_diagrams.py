#!/usr/bin/env python3

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wmw.datasets.common import load_examples, save_examples
from wmw.diagrams.renderer import render_trace_diagram


def main():
    eval_path = Path("data/eval_data/merged_eval.jsonl")
    if not eval_path.exists():
        print(f"ERROR: {eval_path} not found")
        sys.exit(1)

    exs = load_examples(eval_path)
    diagram_dir = Path("data/eval_data/diagrams")
    diagram_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    for ex in exs:
        if ex.source != "synthetic":
            continue
        if ex.image_path and Path(ex.image_path).exists():
            continue


        trace = ex.extra.get("trace")
        if not trace:
            continue

        try:
            img_path = render_trace_diagram(trace, output_dir=str(diagram_dir))
            if img_path and img_path.exists():
                ex.image_path = str(img_path)
                generated += 1
        except Exception as e:
            pass

        if generated % 50 == 0 and generated > 0:
            print(f"  Generated {generated} diagrams...")

    if generated > 0:
        save_examples(exs, eval_path)

    print(f"Generated {generated} synthetic diagrams → {diagram_dir}")


    from collections import Counter
    status = Counter()
    for ex in exs:
        has = bool(ex.image_path and Path(ex.image_path).exists())
        status[f"{ex.source}_{'has' if has else 'no'}_image"] += 1
    print(f"\nImage availability:")
    for k, v in sorted(status.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
