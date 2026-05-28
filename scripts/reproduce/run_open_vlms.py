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

from wmw.datasets.common import EvalExample, load_examples
from wmw.evaluation.prompts import build_prompt, SYSTEM_PROMPTS
from wmw.evaluation.vlm_caller import call_vlm
from wmw.evaluation.trace_parser import parse_trace, extract_answer, answers_match
from wmw.evaluation.reranker import rerank_traces, revise_with_feedback
from wmw.evaluation.open_vlm import (
    OPEN_VLM_REGISTRY, open_vlm_config,
    wait_for_server, check_server_model, ensure_local_key,
)
from wmw.verifiers.pipeline import verify_trace


def run_condition_streaming(
    examples: list[EvalExample],
    config,
    condition: str,
    output_jsonl: Path,
    resume: bool = True,
    log_every: int = 25,
) -> dict:
    done_ids: set[str] = set()
    if resume and output_jsonl.exists():
        with open(output_jsonl) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_ids.add(rec["id"])
                except Exception:
                    pass
        print(f"  Resuming from {len(done_ids)} completed examples")

    system = SYSTEM_PROMPTS.get(condition, SYSTEM_PROMPTS["full_trace"])
    n_correct = 0
    n_parsed = 0
    n_total = 0

    with open(output_jsonl, "a") as f:
        for i, ex in enumerate(examples):
            if ex.id in done_ids:
                continue
            kwargs = {}
            if condition == "gold_state_answer" and ex.extra.get("trace"):
                kwargs["gold_state"] = ex.extra["trace"].get("state_0")
            elif condition == "gold_trans_answer" and ex.extra.get("trace"):
                kwargs["gold_transition"] = ex.extra["trace"].get("transition")

            prompt = build_prompt(ex, condition=condition, **kwargs)
            t0 = time.time()
            resp = call_vlm(config, system, prompt, ex.image_path)
            latency_ms = (time.time() - t0) * 1000

            td, status = parse_trace(resp.raw_text)
            pred = extract_answer(td)
            correct = answers_match(
                pred, ex.gold_answer,
                options=ex.options if hasattr(ex, "options") else None,
            )

            rec = {
                "id": ex.id,
                "source": ex.source,
                "condition": condition,
                "parse_status": status,
                "predicted_answer": str(pred),
                "gold_answer": str(ex.gold_answer),
                "correct": correct,
                "latency_ms": latency_ms,
                "raw_text": resp.raw_text[:8000],
                "trace": td,
                "error": resp.error,
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()

            n_total += 1
            n_correct += int(correct)
            n_parsed += int(status != "failed")

            if (n_total) % log_every == 0:
                acc = n_correct / n_total
                par = n_parsed / n_total
                print(f"      [{condition}] {n_total}/{len(examples) - len(done_ids)} "
                      f"acc={acc:.1%} parse={par:.1%}")


    answers, correct, parsed, latency = [], [], [], []
    traces = []
    with open(output_jsonl) as f:
        for line in f:
            r = json.loads(line)
            answers.append(r["predicted_answer"])
            correct.append(r["correct"])
            parsed.append(r["parse_status"] != "failed")
            latency.append(r["latency_ms"])
            traces.append(r["trace"])

    n = len(correct)
    summary = {
        "model": config.name,
        "condition": condition,
        "n_examples": n,
        "accuracy": sum(correct) / max(n, 1),
        "parse_rate": sum(parsed) / max(n, 1),
        "mean_latency_ms": sum(latency) / max(n, 1),
    }
    return summary, traces


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(OPEN_VLM_REGISTRY),
                   help="Open VLM key to evaluate (one at a time)")
    p.add_argument("--conditions", default="answer_only,full_trace,state_to_answer,gold_state_answer,gold_trans_answer",
                   help="Comma-separated prompt conditions, or 'all'")
    p.add_argument("--data", default="data/eval_data/merged_eval.jsonl",
                   help="JSONL of EvalExample to evaluate on")
    p.add_argument("--output-dir", default="data/results", type=str)
    p.add_argument("--max-examples", type=int, default=200)
    p.add_argument("--max-synthetic", type=int, default=200,
                   help="Cap synthetic subset (gold conditions only use synthetic)")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--server-url", default=None,
                   help="vLLM server URL; defaults to $OPEN_VLM_URL")
    p.add_argument("--wait-server", type=int, default=600,
                   help="Seconds to wait for server before failing")
    p.add_argument("--no-resume", action="store_true",
                   help="Overwrite existing per-example JSONL")
    args = p.parse_args()

    random.seed(args.seed)
    ensure_local_key()

    config = open_vlm_config(args.model, base_url=args.server_url)
    print(f"\n══ Open VLM eval: {config.name} ══")
    print(f"  HF id:    {config.model_id}")
    print(f"  Server:   {config.base_url}")

    if not wait_for_server(config.base_url, timeout_s=args.wait_server):
        print(f"ERROR: vLLM server not reachable at {config.base_url}")
        sys.exit(2)
    served = check_server_model(config.base_url)
    if served and config.model_id not in served:
        print(f"WARNING: server serves {served}, not {config.model_id}. "
              "Will use the served model id.")
        if served:
            config.model_id = served[0]


    examples = load_examples(Path(args.data))
    by_source: dict[str, list] = {}
    for ex in examples:
        by_source.setdefault(ex.source, []).append(ex)
    all_examples = []
    for src, exs in by_source.items():
        cap = min(args.max_examples, len(exs))
        all_examples.extend(exs[:cap])
    synth_examples = [e for e in all_examples if e.source == "synthetic"][:args.max_synthetic]
    print(f"  Total examples: {len(all_examples)} ({len(synth_examples)} synthetic)")


    if args.conditions == "all":
        conditions = ["answer_only", "state_to_answer", "full_trace",
                      "gold_state_answer", "gold_trans_answer"]
    else:
        conditions = [c.strip() for c in args.conditions.split(",")]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    full_trace_traces = None
    full_trace_examples = None
    for cond in conditions:
        target = synth_examples if cond in ("gold_state_answer", "gold_trans_answer") else all_examples
        print(f"\n  ── Running {cond} ({len(target)} examples) ──")
        per_ex_path = output_dir / f"{args.model}_{cond}_per_example.jsonl"
        if args.no_resume and per_ex_path.exists():
            per_ex_path.unlink()
        summary, traces = run_condition_streaming(
            target, config, cond, per_ex_path, resume=not args.no_resume,
        )
        summaries[cond] = summary
        print(f"  → acc={summary['accuracy']:.1%}  "
              f"parse={summary['parse_rate']:.1%}  "
              f"mean_lat={summary['mean_latency_ms']:.0f}ms")


        with open(output_dir / f"{args.model}_{cond}.json", "w") as f:
            json.dump(summary, f, indent=2)

        if cond == "full_trace":
            full_trace_traces = traces
            full_trace_examples = target


    if full_trace_traces is not None:
        traces_path = output_dir / f"{args.model}_raw_traces.jsonl"
        with open(traces_path, "w") as f:
            for i, (td, ex) in enumerate(zip(full_trace_traces, full_trace_examples)):
                rec = {"index": i, "trace": td, "example": ex.to_dict()}
                f.write(json.dumps(rec) + "\n")
        print(f"\n  Saved {len(full_trace_traces)} raw traces → {traces_path.name}")

    print("\n══ Done. Summaries: ══")
    for c, s in summaries.items():
        print(f"  {c:<25s} acc={s['accuracy']:.1%} parse={s['parse_rate']:.1%}")


if __name__ == "__main__":
    main()
