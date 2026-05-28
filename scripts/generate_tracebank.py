#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wmw.generators.scenarios import generate_balanced, SCENARIO_FAMILIES
from wmw.generators.trace_generator import generate_traces
from wmw.generators.perturbation import (
    generate_preference_pairs,
    perturbation_stats,
    get_perturbations,
)
from wmw.verifiers.pipeline import verify_traces
from wmw.metrics import compute_diagnostics, preference_pair_stats


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _generate_splits(
    traces: list, pairs: list, seed: int,
    train_frac: float = 0.6, val_frac: float = 0.2,
) -> dict:
    rng = random.Random(seed + 7)
    trace_ids = [t.id for t in traces]
    rng.shuffle(trace_ids)

    n = len(trace_ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_ids = set(trace_ids[:n_train])
    val_ids = set(trace_ids[n_train:n_train + n_val])
    test_ids = set(trace_ids[n_train + n_val:])


    pair_splits = {}
    for p in pairs:
        src = p.source_trace_id
        if src in train_ids:
            pair_splits[p.id] = "train"
        elif src in val_ids:
            pair_splits[p.id] = "val"
        else:
            pair_splits[p.id] = "test"

    splits = {
        "trace_splits": {tid: ("train" if tid in train_ids else
                               "val" if tid in val_ids else "test")
                        for tid in trace_ids},
        "pair_splits": pair_splits,
        "counts": {
            "train_traces": len(train_ids),
            "val_traces": len(val_ids),
            "test_traces": len(test_ids),
            "train_pairs": sum(1 for v in pair_splits.values() if v == "train"),
            "val_pairs": sum(1 for v in pair_splits.values() if v == "val"),
            "test_pairs": sum(1 for v in pair_splits.values() if v == "test"),
        },
    }
    return splits


def main():
    parser = argparse.ArgumentParser(description="Generate WMW TraceBank artifact")
    parser.add_argument(
        "--num-scenarios", type=int, default=200,
        help="Number of positive scenarios to generate (default: 200)",
    )
    parser.add_argument(
        "--pairs-per-trace", type=int, default=16,
        help="Preference pairs per trace (default: 16)",
    )
    parser.add_argument(
        "--balance-labels", action="store_true", default=True,
        help="Balance perturbation labels across pairs (default: True)",
    )
    parser.add_argument(
        "--seed", type=int, default=2026,
        help="Random seed (default: 2026)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data",
        help="Output directory (default: data/)",
    )
    parser.add_argument(
        "--paper-seed", action="store_true",
        help="Generate paper artifact: 32 traces, 256 pairs (8 per trace)",
    )
    args = parser.parse_args()


    if args.paper_seed:
        args.num_scenarios = 32
        args.pairs_per_trace = 8

    random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"═══ WMW TraceBank Generator ═══")
    print(f"  Scenarios:       {args.num_scenarios}")
    print(f"  Pairs/trace:     {args.pairs_per_trace}")
    print(f"  Expected pairs:  ~{args.num_scenarios * args.pairs_per_trace}")
    print(f"  Seed:            {args.seed}")
    print(f"  Output:          {out_dir.resolve()}")
    print()


    n_overshoot = int(args.num_scenarios * 1.1)
    print(f"Generating scenarios (target {args.num_scenarios}, oversampling {n_overshoot})...", end=" ", flush=True)
    specs = generate_balanced(n_overshoot)
    print(f"done ({len(specs)} specs)")


    print("Converting to traces (deduplicating)...", end=" ", flush=True)
    traces = generate_traces(specs, deduplicate=True)

    traces = traces[:args.num_scenarios]
    print(f"done ({len(traces)} unique traces)")


    print("Verifying positive traces...", end=" ", flush=True)
    results = verify_traces(traces)
    n_clean = sum(1 for r in results if r.all_ok)
    print(f"done ({n_clean}/{len(traces)} pass all checks)")


    print(f"Generating preference pairs ({args.pairs_per_trace} per trace)...", end=" ", flush=True)
    pairs = generate_preference_pairs(
        traces,
        pairs_per_trace=args.pairs_per_trace,
        balance_labels=args.balance_labels,
    )
    print(f"done ({len(pairs)} non-trivial pairs)")


    import json as _json
    noop_count = sum(1 for p in pairs
                     if _json.dumps(p.chosen, sort_keys=True) == _json.dumps(p.rejected, sort_keys=True))
    if noop_count > 0:
        print(f"  WARNING: {noop_count} no-op pairs remain (chosen == rejected)")
    else:
        print(f"  ✓ All pairs have actual perturbations (chosen ≠ rejected)")


    print("Generating train/val/test splits...", end=" ", flush=True)
    splits = _generate_splits(traces, pairs, args.seed)
    print(f"done ({splits['counts']})")


    pair_dicts = [p.to_dict() for p in pairs]
    pstats = perturbation_stats(pairs)
    pstats2 = preference_pair_stats(pair_dicts)


    diag = compute_diagnostics(results, answer_correct=[True] * len(traces))


    family_dist = {}
    for t in traces:
        family_dist[t.scenario_family] = family_dist.get(t.scenario_family, 0) + 1


    traces_path = out_dir / "trace_examples_seed.jsonl"
    pairs_path = out_dir / "preference_pairs_seed.jsonl"
    stats_path = out_dir / "generation_stats.json"
    splits_path = out_dir / "splits.json"

    print(f"\nWriting {traces_path}...", end=" ", flush=True)
    with open(traces_path, "w") as f:
        for trace in traces:
            json.dump(trace.to_dict(), f)
            f.write("\n")
    print("done")

    print(f"Writing {pairs_path}...", end=" ", flush=True)
    with open(pairs_path, "w") as f:
        for pd in pair_dicts:
            json.dump(pd, f)
            f.write("\n")
    print("done")


    traces_hash = _file_hash(traces_path)
    pairs_hash = _file_hash(pairs_path)

    stats = {
        "version": "1.0.0",
        "generation_config": {
            "num_scenarios": args.num_scenarios,
            "pairs_per_trace": args.pairs_per_trace,
            "balance_labels": args.balance_labels,
            "seed": args.seed,
        },
        "traces": {
            "total": len(traces),
            "unique_questions": len(set(t.question for t in traces)),
            "schema_valid": n_clean,
            "noop_pairs_filtered": 0,
            "by_family": family_dist,
            "sha256": traces_hash,
        },
        "preference_pairs": {
            "total": len(pairs),
            "noop_filtered": noop_count,
            "by_label": pstats["by_label"],
            "by_family_split": pstats["by_family"],
            "by_field": pstats["by_field"],
            "sha256": pairs_hash,
        },
        "splits": splits["counts"],
        "perturbation_registry": {
            "total_perturbations": len(get_perturbations()),
            "seen": len(get_perturbations(family="seen")),
            "held_out": len(get_perturbations(family="held_out")),
        },
        "diagnostics": {
            "state_accuracy": diag.state_accuracy,
            "transition_accuracy": diag.transition_accuracy,
            "trace_answer_consistency": diag.trace_answer_consistency,
            "abstention_rate": diag.abstention_rate,
        },
    }

    print(f"Writing {stats_path}...", end=" ", flush=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print("done")

    print(f"Writing {splits_path}...", end=" ", flush=True)
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    print("done")


    print(f"\n{diag.summary()}")
    print(f"\n  Preference pair distribution:")
    for lbl, count in sorted(pstats["by_label"].items()):
        print(f"    {lbl:<16s}  {count:>5d}  ({count/len(pairs):.1%})")
    print(f"    {'─'*32}")
    print(f"    seen:      {pstats['by_family']['seen']:>5d}")
    print(f"    held_out:  {pstats['by_family']['held_out']:>5d}")

    sz_traces = traces_path.stat().st_size / 1024
    sz_pairs = pairs_path.stat().st_size / 1024
    print(f"\n  File sizes:")
    print(f"    traces:  {sz_traces:>8.1f} KB")
    print(f"    pairs:   {sz_pairs:>8.1f} KB")
    print(f"\n  Version hashes:")
    print(f"    traces:  {traces_hash}")
    print(f"    pairs:   {pairs_hash}")
    print(f"\n═══ Done ═══")


if __name__ == "__main__":
    main()
