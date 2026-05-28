#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from wmw.datasets.common import load_examples
from wmw.evaluation.vlm_caller import MODELS, ModelConfig
from wmw.evaluation.reranker import rerank_traces
from wmw.evaluation.trace_parser import extract_answer, answers_match
from wmw.evaluation.open_vlm import open_vlm_config, OPEN_VLM_REGISTRY, ensure_local_key
from wmw.verifiers.pipeline import verify_trace


def resolve_model(model_key: str) -> ModelConfig:
    if model_key in MODELS:
        return MODELS[model_key]
    if model_key in OPEN_VLM_REGISTRY:
        ensure_local_key()
        return open_vlm_config(model_key)
    raise KeyError(f"Unknown model '{model_key}'. "
                   f"Closed: {list(MODELS)}. Open: {list(OPEN_VLM_REGISTRY)}")


def run_sweep(model_key: str, examples, ks, scoring, judge_config=None,
              temperature: float = 0.7, output: Path | None = None) -> dict:
    config = resolve_model(model_key)
    print(f"\n══ Rerank sweep: {config.name}  (k ∈ {ks}) ══")
    print(f"  n_examples: {len(examples)}, scoring={scoring}, temperature={temperature}")


    k_max = max(ks)
    per_example = []
    for i, ex in enumerate(examples):
        t0 = time.time()
        best, all_traces, all_scores = rerank_traces(
            ex, config, k=k_max, temperature=temperature, scoring=scoring,
            judge_config=judge_config,
        )
        per_example.append({
            "id": ex.id,
            "gold_answer": str(ex.gold_answer),
            "all_scores": all_scores,
            "all_answers": [str(extract_answer(t)) for t in all_traces],
            "all_correct": [
                answers_match(extract_answer(t), ex.gold_answer,
                              options=ex.options if hasattr(ex, "options") else None)
                for t in all_traces
            ],
            "all_traces_valid": [
                bool(t and verify_trace(t).all_ok) for t in all_traces
            ],
            "latency_ms": (time.time() - t0) * 1000,
        })
        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{len(examples)}]  last_lat={per_example[-1]['latency_ms']:.0f}ms")

        if output:
            tmp = output.with_suffix(".jsonl.tmp")
            with open(tmp, "a") as f:
                f.write(json.dumps(per_example[-1]) + "\n")


    curve = {}
    for k in ks:
        n = len(per_example)
        rerank_correct = 0
        rerank_valid = 0
        for rec in per_example:
            scores = rec["all_scores"][:k]
            correct = rec["all_correct"][:k]
            valid = rec["all_traces_valid"][:k]
            if not scores:
                continue
            best_i = max(range(len(scores)), key=lambda i: scores[i])
            if correct[best_i]:
                rerank_correct += 1
            if valid[best_i]:
                rerank_valid += 1
        curve[k] = {
            "k": k,
            "accuracy": rerank_correct / n if n else 0.0,
            "trace_validity": rerank_valid / n if n else 0.0,
            "n_examples": n,
        }
        print(f"  k={k:>2d}:  acc={curve[k]['accuracy']:.1%}  "
              f"valid={curve[k]['trace_validity']:.1%}")

    result = {
        "model": config.name,
        "model_key": model_key,
        "scoring": scoring,
        "temperature": temperature,
        "curve": curve,
        "n_examples": len(examples),
        "per_example": per_example,
    }
    if output:
        with open(output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved → {output}")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Model key (closed or open)")
    p.add_argument("--data", default="data/eval_data/merged_eval.jsonl")
    p.add_argument("--ks", default="1,4,8,16",
                   help="Comma-separated k values for the curve")
    p.add_argument("--scoring", default="rules",
                   choices=["rules", "llm_judge", "ensemble", "majority_vote"])
    p.add_argument("--judge-model", default=None,
                   help="Model key for LLM judge (only for scoring=llm_judge|ensemble)")
    p.add_argument("--max-examples", type=int, default=50,
                   help="Cap eval set (rerank is k× the cost of a single pass)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output-dir", default="data/results")
    args = p.parse_args()

    random.seed(args.seed)
    examples = load_examples(Path(args.data))
    if args.max_examples > 0:
        random.shuffle(examples)
        examples = examples[:args.max_examples]

    ks = [int(x) for x in args.ks.split(",")]
    judge_config = resolve_model(args.judge_model) if args.judge_model else None
    output = Path(args.output_dir) / f"{args.model}_rerank_sweep.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    run_sweep(args.model, examples, ks, args.scoring, judge_config,
              args.temperature, output)


if __name__ == "__main__":
    main()
