#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from wmw.training.dpo_data import build_dpo_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", default="data/preference_pairs_seed.jsonl")
    p.add_argument("--splits", default="data/splits.json")
    p.add_argument("--output-dir", default="data/dpo")
    p.add_argument("--image-root", default="data/eval_data/diagrams",
                   help="Where rendered diagrams live (matched by source_trace_id)")
    args = p.parse_args()

    img_root = Path(args.image_root) if Path(args.image_root).exists() else None
    counts = build_dpo_split(
        Path(args.pairs), Path(args.splits),
        Path(args.output_dir), image_root=img_root,
    )
    print("DPO splits written:")
    for k, v in counts.items():
        print(f"  {k}: {v} pairs")


if __name__ == "__main__":
    main()
