#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wmw.verifiers.pipeline import verify_trace
from wmw.verifiers.llm_judge import call_llm_judge
from wmw.verifiers.ensemble import (
    merge_verifier_results, compute_ensemble_stats, EnsembleResult,
)
from wmw.evaluation.vlm_caller import MODELS
from wmw.schemas.models import FAILURE_LABELS


def load_traces(path: Path) -> list[dict]:
    traces = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    return traces


def load_rejected_traces(path: Path) -> list[dict]:
    rejected = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                pair = json.loads(line)
                rejected.append(pair["rejected"])
    return rejected


def run_ensemble_comparison(
    traces: list[dict],
    judge_model: str = "mock",
    include_gold: bool = False,
    max_traces: int | None = None,
) -> list[EnsembleResult]:
    config = MODELS.get(judge_model, MODELS["mock"])
    results = []

    if max_traces:
        traces = traces[:max_traces]

    for i, td in enumerate(traces):

        rule_result = verify_trace(td)


        question = td.get("question", "")
        gold_answer = td.get("metadata", {}).get("gold_answer")
        judge_result = call_llm_judge(
            td, question=question, gold_answer=gold_answer,
            include_gold=include_gold, model_config=config,
        )


        ensemble = merge_verifier_results(rule_result, judge_result)
        results.append(ensemble)

        if (i + 1) % 25 == 0 or i == len(traces) - 1:
            print(f"  [{i+1}/{len(traces)}] rules={len(rule_result.labels)} "
                  f"judge={len(judge_result.labels)} "
                  f"ensemble={len(ensemble.merged.labels)}")

    return results


def print_disagreement_matrix(stats):
    print(f"\n{'Label':<18s} {'Rules':>6s} {'Judge':>6s} {'Both':>6s} {'Neither':>7s}")
    print("─" * 45)
    for lbl in FAILURE_LABELS:
        d = stats.by_label.get(lbl, {})
        print(f"  {lbl:<16s} {d.get('rules_only',0):>6d} "
              f"{d.get('judge_only',0):>6d} "
              f"{d.get('both',0):>6d} "
              f"{d.get('neither',0):>7d}")
    print("─" * 45)
    print(f"  {'TOTAL':<16s} {stats.rules_only_count:>6d} "
          f"{stats.judge_only_count:>6d} "
          f"{stats.both_count:>6d} "
          f"{stats.neither_count:>7d}")
    print(f"\n  Ensemble detection gain over rules-only: "
          f"{stats.ensemble_detection_gain:.1%}")


def main():
    parser = argparse.ArgumentParser(
        description="WMW Verifier Ensemble Comparison")
    parser.add_argument("--traces", type=str, default=None,
                        help="Path to traces JSONL (positive traces)")
    parser.add_argument("--pairs", type=str, default=None,
                        help="Path to preference pairs JSONL (use rejected traces)")
    parser.add_argument("--judge-model", type=str, default="mock",
                        help="Model for LLM judge (default: mock)")
    parser.add_argument("--include-gold", action="store_true",
                        help="Show gold answer to the LLM judge")
    parser.add_argument("--max-traces", type=int, default=None,
                        help="Cap number of traces to verify")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    args = parser.parse_args()

    if not args.traces and not args.pairs:
        print("ERROR: provide --traces or --pairs")
        sys.exit(1)


    if args.traces:
        print(f"Loading traces from {args.traces}")
        traces = load_traces(Path(args.traces))
        label = "positive"
    else:
        print(f"Loading rejected traces from {args.pairs}")
        traces = load_rejected_traces(Path(args.pairs))
        label = "rejected"

    print(f"  Loaded {len(traces)} {label} traces")
    print(f"  Judge model: {args.judge_model}")


    ensemble_results = run_ensemble_comparison(
        traces, args.judge_model, args.include_gold, args.max_traces,
    )


    stats = compute_ensemble_stats(ensemble_results)


    print(f"\n═══ Verifier Ensemble Results ({stats.n_traces} traces) ═══")
    print_disagreement_matrix(stats)


    n_rules_flagged = sum(1 for er in ensemble_results if er.rules.labels)
    n_judge_flagged = sum(1 for er in ensemble_results if er.llm_judge.labels)
    n_ensemble_flagged = sum(1 for er in ensemble_results if er.merged.labels)
    print(f"\n  Traces flagged:")
    print(f"    Rules-only:  {n_rules_flagged}/{stats.n_traces} ({n_rules_flagged/stats.n_traces:.1%})")
    print(f"    Judge-only:  {n_judge_flagged}/{stats.n_traces} ({n_judge_flagged/stats.n_traces:.1%})")
    print(f"    Ensemble:    {n_ensemble_flagged}/{stats.n_traces} ({n_ensemble_flagged/stats.n_traces:.1%})")


    output_path = args.output or f"data/ensemble_{label}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output = {
        "n_traces": stats.n_traces,
        "label": label,
        "judge_model": args.judge_model,
        "rules_only_count": stats.rules_only_count,
        "judge_only_count": stats.judge_only_count,
        "both_count": stats.both_count,
        "neither_count": stats.neither_count,
        "ensemble_detection_gain": stats.ensemble_detection_gain,
        "by_label": stats.by_label,
        "traces_flagged": {
            "rules": n_rules_flagged,
            "judge": n_judge_flagged,
            "ensemble": n_ensemble_flagged,
        },
        "per_trace": [
            {
                "rules_labels": er.rules.labels,
                "judge_labels": er.llm_judge.labels,
                "ensemble_labels": er.merged.labels,
                "rules_only": er.rules_only_labels,
                "judge_only": er.judge_only_labels,
            }
            for er in ensemble_results
        ],
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved → {output_path}")


if __name__ == "__main__":
    main()
