#!/usr/bin/env python3

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def bundle(model_name: str, results_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        f"{model_name}_full_trace.json",
        f"{model_name}_answer_only.json",
        f"{model_name}_state_to_answer.json",
        f"{model_name}_verification.json",
        f"{model_name}_stress.json",
    ]

    copied = 0
    for fname in files_to_copy:
        src = results_dir / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)
            size_kb = src.stat().st_size / 1024
            print(f"  Copied {fname} ({size_kb:.1f} KB)")
            copied += 1
        else:
            print(f"  SKIP (not found): {fname}")


    ft_path = results_dir / f"{model_name}_full_trace.json"
    ver_path = results_dir / f"{model_name}_verification.json"

    if ft_path.exists() and ver_path.exists():
        with open(ft_path) as f:
            ft_data = json.load(f)
        with open(ver_path) as f:
            ver_data = json.load(f)

        summary_path = output_dir / f"{model_name}_traces.jsonl"
        with open(summary_path, "w") as f:
            per_example = ft_data.get("per_example", [])
            for i, ex in enumerate(per_example):
                record = {
                    "index": i,
                    "answer": ex.get("answer"),
                    "correct": ex.get("correct"),
                    "parse_status": ex.get("parse"),
                }
                if i < len(ver_data):
                    record["verifier"] = ver_data[i]
                json.dump(record, f)
                f.write("\n")
        print(f"  Created {model_name}_traces.jsonl ({len(per_example)} examples)")

    return copied


def main():
    parser = argparse.ArgumentParser(description="Bundle model outputs for HF release")
    parser.add_argument("--model", action="append", required=True,
                        help="Model name(s) to bundle (repeat for multiple)")
    parser.add_argument("--results-dir", type=str, default="data/results",
                        help="Where run_evaluation.py wrote results")
    parser.add_argument("--hf-dir", type=str, default="data/hf_release/model_outputs",
                        help="Output directory in HF release")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.hf_dir)

    if not results_dir.exists():
        print(f"ERROR: {results_dir} not found. Run run_evaluation.py first.")
        sys.exit(1)

    total = 0
    for model in args.model:
        print(f"\n  Bundling {model}:")
        total += bundle(model, results_dir, output_dir)

    print(f"\n  Total: {total} files bundled → {output_dir}")


if __name__ == "__main__":
    main()
