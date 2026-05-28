#!/usr/bin/env python3

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wmw.verifiers.pipeline import verify_trace
from wmw.metrics import compute_diagnostics, preference_pair_stats
from wmw.schemas.models import FAILURE_LABELS


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main():
    parser = argparse.ArgumentParser(description="WMW Verifier Audit Protocol")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--sample-size", type=int, default=50, help="Stratified sample size for human audit")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    random.seed(args.seed)
    data_dir = Path(args.data_dir)


    traces_path = data_dir / "trace_examples_seed.jsonl"
    pairs_path = data_dir / "preference_pairs_seed.jsonl"

    if not traces_path.exists():
        print(f"ERROR: {traces_path} not found. Run generate_tracebank.py first.")
        sys.exit(1)

    traces = load_jsonl(traces_path)
    pairs = load_jsonl(pairs_path) if pairs_path.exists() else []

    print(f"═══ WMW Verifier Audit ═══")
    print(f"  Traces loaded:  {len(traces)}")
    print(f"  Pairs loaded:   {len(pairs)}")
    print()


    print("Verifying positive traces (expect ~0 errors)...")
    pos_results = []
    false_positives = {lbl: 0 for lbl in FAILURE_LABELS}
    for t in traces:
        r = verify_trace(t)
        pos_results.append(r)
        for lbl in r.labels:
            if lbl in false_positives:
                false_positives[lbl] += 1

    n = len(traces)
    print(f"\n  False positive rates on gold traces (N={n}):")
    for lbl in FAILURE_LABELS:
        rate = false_positives[lbl] / n if n > 0 else 0
        flag = " ⚠" if rate > 0.05 else ""
        print(f"    {lbl:<16s}  {false_positives[lbl]:>4d}/{n}  ({rate:.1%}){flag}")


    if pairs:
        print(f"\nVerifying rejected traces (expect high detection)...")
        detected = {lbl: 0 for lbl in FAILURE_LABELS}
        total_by_label = {lbl: 0 for lbl in FAILURE_LABELS}

        for pair in pairs:
            injected_label = pair.get("perturbation_type", "")
            rejected = pair.get("rejected", {})
            r = verify_trace(rejected)

            if injected_label in total_by_label:
                total_by_label[injected_label] += 1
                if injected_label in r.labels:
                    detected[injected_label] += 1

        print(f"\n  Detection rates on rejected traces:")
        for lbl in FAILURE_LABELS:
            tot = total_by_label[lbl]
            det = detected[lbl]
            rate = det / tot if tot > 0 else 0
            flag = " ⚠" if rate < 0.5 and tot > 0 else ""
            print(f"    {lbl:<16s}  {det:>4d}/{tot:<4d}  ({rate:.1%}){flag}")


    print(f"\n  Stratified sample for human audit (target: {args.sample_size}):")


    sample = []
    if pairs:
        by_label = {}
        for p in pairs:
            lbl = p.get("perturbation_type", "other")
            by_label.setdefault(lbl, []).append(p)

        per_label = max(1, args.sample_size // len(by_label))
        for lbl, pool in sorted(by_label.items()):
            k = min(per_label, len(pool))
            sample.extend(random.sample(pool, k))

        random.shuffle(sample)
        sample = sample[:args.sample_size]


    pos_sample_size = min(args.sample_size // 4, len(traces))
    pos_sample = random.sample(traces, pos_sample_size)


    audit_path = data_dir / "audit_sample.jsonl"
    with open(audit_path, "w") as f:
        for item in pos_sample:
            json.dump({"type": "positive", "trace": item}, f)
            f.write("\n")
        for item in sample:
            json.dump({"type": "rejected_pair", "pair": item}, f)
            f.write("\n")

    print(f"    Positive traces sampled:  {pos_sample_size}")
    print(f"    Rejected pairs sampled:   {len(sample)}")
    print(f"    Written to:               {audit_path}")


    diag = compute_diagnostics(pos_results, answer_correct=[True] * n)
    print(f"\n{diag.summary()}")
    print(f"\n═══ Audit Complete ═══")


if __name__ == "__main__":
    main()
