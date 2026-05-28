#!/usr/bin/env python3

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wmw.datasets.common import EvalExample, save_examples, load_examples
from wmw.datasets.prepare import prepare_all_datasets, prepare_synthetic
from wmw.evaluation.prompts import build_prompt, SYSTEM_PROMPTS
from wmw.evaluation.vlm_caller import ModelConfig, MODELS, call_vlm, VLMResponse
from wmw.evaluation.trace_parser import parse_trace, extract_answer, answers_match
from wmw.evaluation.reranker import rerank_traces, revise_with_feedback
from wmw.evaluation.stress_tests import (
    StressTestResult, run_trace_ablation, run_counterfactual_edit,
    run_held_out_eval, run_natural_rejected_eval,
)
from wmw.evaluation.latex_tables import (
    table_main, table_failures, table_human_agreement,
    table_stress, table_cross_source, write_all_tables,
)
from wmw.verifiers.pipeline import verify_trace
from wmw.metrics import compute_diagnostics, visual_state_gap, transition_gap
from wmw.schemas.models import VerifierResult, FAILURE_LABELS
from wmw.generators.scenarios import generate_balanced
from wmw.generators.trace_generator import generate_traces


@dataclass
class ConditionResult:
    model: str
    condition: str
    examples: list[dict] = field(default_factory=list)
    traces: list[dict | None] = field(default_factory=list)
    parse_statuses: list[str] = field(default_factory=list)
    answers: list = field(default_factory=list)
    correct: list[bool] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    verifier_results: list[VerifierResult | None] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.examples)

    @property
    def accuracy(self) -> float:
        return sum(self.correct) / self.n if self.n else 0.0

    @property
    def parse_rate(self) -> float:
        ok = sum(1 for s in self.parse_statuses if s != "failed")
        return ok / self.n if self.n else 0.0


@dataclass
class ModelResults:
    model_name: str
    conditions: dict[str, ConditionResult] = field(default_factory=dict)
    diagnostic_report: dict | None = None
    ensemble_stats: dict | None = None
    stress_results: list[StressTestResult] = field(default_factory=list)


def stage_data(args) -> dict[str, list[EvalExample]]:
    print("\n" + "=" * 60)
    print("STAGE 1: DATA PREPARATION")
    print("=" * 60)

    data_dir = Path(args.output_dir) / "eval_data"
    merged_path = data_dir / "merged_eval.jsonl"

    if merged_path.exists() and not args.force_download:
        print(f"  Loading cached data from {data_dir}")
        all_examples = load_examples(merged_path)
        by_source: dict[str, list[EvalExample]] = {}
        for ex in all_examples:
            by_source.setdefault(ex.source, []).append(ex)
        for src, exs in by_source.items():
            print(f"    {src}: {len(exs)} examples")
        return by_source

    if args.synthetic_only:
        print("  Synthetic only mode")
        synth = prepare_synthetic(data_dir, args.n_synthetic, args.seed)
        return {"synthetic": synth}

    return prepare_all_datasets(
        output_dir=data_dir,
        n_synthetic=args.n_synthetic,
        max_scienceqa=args.max_examples,
        max_clevrer=args.max_examples,
        max_mathvista=args.max_examples,
        seed=args.seed,
        skip_download_errors=True,
    )


def _run_condition(
    examples: list[EvalExample],
    config: ModelConfig,
    condition: str,
    rate_limit_delay: float = 0.0,
) -> ConditionResult:
    cr = ConditionResult(model=config.name, condition=condition)
    system = SYSTEM_PROMPTS.get(condition, SYSTEM_PROMPTS["full_trace"])

    for i, ex in enumerate(examples):

        kwargs = {}
        if condition == "gold_state_answer" and ex.extra.get("trace"):
            kwargs["gold_state"] = ex.extra["trace"].get("state_0")
        elif condition == "gold_trans_answer" and ex.extra.get("trace"):
            kwargs["gold_transition"] = ex.extra["trace"].get("transition")

        prompt = build_prompt(ex, condition=condition, **kwargs)
        resp = call_vlm(config, system, prompt, ex.image_path)


        td, status = parse_trace(resp.raw_text)
        pred_answer = extract_answer(td)
        correct = answers_match(pred_answer, ex.gold_answer,
                               options=ex.options if hasattr(ex, 'options') else None)

        cr.examples.append(ex.to_dict())
        cr.traces.append(td)
        cr.parse_statuses.append(status)
        cr.answers.append(pred_answer)
        cr.correct.append(correct)
        cr.latencies_ms.append(resp.latency_ms)

        if rate_limit_delay > 0:
            time.sleep(rate_limit_delay)

        if (i + 1) % 25 == 0:
            print(f"      [{condition}] {i+1}/{len(examples)} "
                  f"acc={sum(cr.correct)/(i+1):.1%} parse={cr.parse_rate:.1%}")

    return cr


def stage_eval(
    datasets: dict[str, list[EvalExample]],
    args,
) -> dict[str, ModelResults]:
    print("\n" + "=" * 60)
    print("STAGE 2: VLM EVALUATION")
    print("=" * 60)


    all_examples = []
    for source, exs in datasets.items():
        cap = min(len(exs), args.max_examples or len(exs))
        all_examples.extend(exs[:cap])
    print(f"  Total examples: {len(all_examples)}")

    models = args.models.split(",")
    conditions = ["answer_only", "full_trace", "state_to_answer"]


    synth_examples = [ex for ex in all_examples if ex.source == "synthetic"]
    external_examples = [ex for ex in all_examples if ex.source != "synthetic"]

    all_model_results: dict[str, ModelResults] = {}

    for model_name in models:
        config = MODELS.get(model_name)
        if config is None:
            print(f"  WARNING: Unknown model '{model_name}', skipping")
            continue

        print(f"\n  ── Model: {config.name} ──")
        mr = ModelResults(model_name=config.name)

        rate_delay = 60.0 / config.rate_limit_rpm if config.provider != "mock" else 0.0


        for cond in conditions:
            print(f"    Running condition: {cond}")
            cr = _run_condition(all_examples, config, cond, rate_delay)
            mr.conditions[cond] = cr
            print(f"    → acc={cr.accuracy:.1%}, parse={cr.parse_rate:.1%}, "
                  f"mean_latency={sum(cr.latencies_ms)/max(len(cr.latencies_ms),1):.0f}ms")


        if synth_examples:
            for cond in ["gold_state_answer", "gold_trans_answer"]:
                print(f"    Running condition: {cond} (synthetic subset, n={len(synth_examples)})")
                cr = _run_condition(synth_examples, config, cond, rate_delay)
                mr.conditions[cond] = cr
                print(f"    → acc={cr.accuracy:.1%}")


        rerank_n = min(30, len(all_examples))
        rerank_examples = random.sample(all_examples, rerank_n)


        judge_model_name = args.judge_model if hasattr(args, 'judge_model') else "mock"
        judge_cfg = MODELS.get(judge_model_name, MODELS["mock"])

        for scoring_method in ["rules", "majority_vote", "llm_judge", "ensemble"]:
            method_label = f"rerank_{scoring_method}"
            print(f"    Running reranking: {scoring_method} (n={rerank_n}, k=5)")
            rerank_traces_list = []
            rerank_correct = 0
            for ex in rerank_examples:
                best, _, scores = rerank_traces(
                    ex, config, k=5, scoring=scoring_method,
                    judge_config=judge_cfg,
                )
                pred = extract_answer(best)
                if answers_match(pred, ex.gold_answer,
                               options=ex.options if hasattr(ex, 'options') else None):
                    rerank_correct += 1
                rerank_traces_list.append(best)
            mr.conditions[method_label] = ConditionResult(
                model=config.name, condition=method_label,
                traces=rerank_traces_list,
                correct=[answers_match(extract_answer(t), ex.gold_answer,
                                       options=ex.options if hasattr(ex, 'options') else None)
                         for t, ex in zip(rerank_traces_list, rerank_examples)],
            )
            print(f"    → {method_label} acc={rerank_correct/rerank_n:.1%}")


        ablation_n = min(30, len(all_examples))
        ablation_examples = random.sample(all_examples, ablation_n)
        print(f"    Running schema ablation (n={ablation_n})")

        cr_cot = _run_condition(ablation_examples, config, "ablation", rate_delay)
        mr.conditions["schema_ablation_cot"] = cr_cot
        print(f"    → CoT acc={cr_cot.accuracy:.1%}, parse={cr_cot.parse_rate:.1%}")


        ft = mr.conditions.get("full_trace")
        if ft:
            error_indices = [i for i, t in enumerate(ft.traces)
                           if t is not None and not verify_trace(t).all_ok][:15]
            revise_improved = 0
            for idx in error_indices:
                revised, feedback = revise_with_feedback(
                    EvalExample(**ft.examples[idx]), config, ft.traces[idx]
                )
                if revised:
                    new_vr = verify_trace(revised)
                    old_vr = verify_trace(ft.traces[idx])
                    if new_vr.violation_count < old_vr.violation_count:
                        revise_improved += 1
            revise_rate = revise_improved / max(len(error_indices), 1)
            print(f"    → revise improved {revise_improved}/{len(error_indices)} "
                  f"({revise_rate:.0%})")

        all_model_results[model_name] = mr


    results_dir = Path(args.output_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    for mname, mr in all_model_results.items():
        for cname, cr in mr.conditions.items():
            out = {
                "model": mr.model_name,
                "condition": cname,
                "n_examples": cr.n,
                "accuracy": cr.accuracy,
                "parse_rate": cr.parse_rate,
                "mean_latency_ms": sum(cr.latencies_ms) / max(len(cr.latencies_ms), 1),
                "per_example": [
                    {"answer": str(a), "correct": c, "parse": s}
                    for a, c, s in zip(cr.answers, cr.correct, cr.parse_statuses)
                ] if cr.answers else [],
            }
            with open(results_dir / f"{mname}_{cname}.json", "w") as f:
                json.dump(out, f, indent=2)


        ft = mr.conditions.get("full_trace")
        if ft and ft.traces:
            traces_path = results_dir / f"{mname}_raw_traces.jsonl"
            with open(traces_path, "w") as f:
                for i, (td, ex) in enumerate(zip(ft.traces, ft.examples)):
                    record = {"index": i, "trace": td, "example": ex}
                    json.dump(record, f)
                    f.write("\n")
            print(f"    Saved {len(ft.traces)} raw traces → {traces_path.name}")

    return all_model_results


def stage_verify(
    all_results: dict[str, ModelResults],
    args,
) -> dict[str, ModelResults]:
    print("\n" + "=" * 60)
    print("STAGE 3: VERIFICATION (rules + LLM judge + ensemble)")
    print("=" * 60)

    from wmw.verifiers.llm_judge import call_llm_judge
    from wmw.verifiers.ensemble import merge_verifier_results, compute_ensemble_stats


    judge_model_name = args.judge_model if hasattr(args, 'judge_model') else "mock"
    judge_config = MODELS.get(judge_model_name, MODELS["mock"])

    for mname, mr in all_results.items():
        ft = mr.conditions.get("full_trace")
        if not ft:
            continue

        print(f"  Verifying {mr.model_name} full_trace ({ft.n} traces)")
        print(f"  Judge model: {judge_config.name}")

        vr_list = []
        judge_list = []
        ensemble_list = []

        for i, td in enumerate(ft.traces):
            if td is None:
                vr_list.append(None)
                judge_list.append(None)
                ensemble_list.append(None)
                continue


            if "id" not in td:
                td["id"] = "unknown"
            if "scenario_family" not in td:
                td["scenario_family"] = "unknown"
            if "question" not in td:
                td["question"] = ""
            if "metadata" not in td:
                td["metadata"] = {"source": "model_generated"}


            vr = verify_trace(td)
            vr_list.append(vr)


            question = td.get("question", "")
            gold = td.get("metadata", {}).get("gold_answer")
            jr = call_llm_judge(td, question=question, gold_answer=gold,
                               model_config=judge_config)
            judge_list.append(jr)


            ens = merge_verifier_results(vr, jr)
            ensemble_list.append(ens)

            if (i + 1) % 25 == 0:
                print(f"    [{i+1}/{ft.n}] verified")

        ft.verifier_results = vr_list


        ens_valid = [e for e in ensemble_list if e is not None]
        if ens_valid:
            ens_stats = compute_ensemble_stats(ens_valid)
            mr.ensemble_stats = {
                "rules_only_count": ens_stats.rules_only_count,
                "judge_only_count": ens_stats.judge_only_count,
                "both_count": ens_stats.both_count,
                "neither_count": ens_stats.neither_count,
                "ensemble_detection_gain": ens_stats.ensemble_detection_gain,
                "by_label": ens_stats.by_label,
            }
            print(f"    Ensemble: rules_only={ens_stats.rules_only_count} "
                  f"judge_only={ens_stats.judge_only_count} "
                  f"both={ens_stats.both_count} "
                  f"gain={ens_stats.ensemble_detection_gain:.1%}")


        valid_vrs = [v for v in vr_list if v is not None]
        valid_correct = [c for c, v in zip(ft.correct, vr_list) if v is not None]
        if valid_vrs:
            report = compute_diagnostics(valid_vrs, valid_correct)
            mr.diagnostic_report = {
                "n_traces": report.n_traces,
                "answer_accuracy": report.answer_accuracy,
                "state_accuracy": report.state_accuracy,
                "transition_accuracy": report.transition_accuracy,
                "hidden_inconsistency_rate": report.hidden_inconsistency_rate,
                "trace_answer_consistency": report.trace_answer_consistency,
                "abstention_rate": report.abstention_rate,
                "failure_counts": report.failure_counts,
                "failure_rates": report.failure_rates,
            }
            print(report.summary())


        results_dir = Path(args.output_dir) / "results"
        vr_data = []
        for i, (vr, jr, ens) in enumerate(zip(vr_list, judge_list, ensemble_list)):
            if vr is not None:
                vr_data.append({
                    "index": i,
                    "rules": {
                        "all_ok": vr.all_ok, "labels": vr.labels,
                        "state_ok": vr.state_ok, "transition_ok": vr.transition_ok,
                    },
                    "judge": {
                        "labels": jr.labels if jr else [],
                        "abstained": jr.abstained if jr else True,
                    },
                    "ensemble": {
                        "labels": ens.merged.labels if ens else [],
                        "rules_only": ens.rules_only_labels if ens else [],
                        "judge_only": ens.judge_only_labels if ens else [],
                    },
                })
        with open(results_dir / f"{mname}_verification.json", "w") as f:
            json.dump(vr_data, f, indent=2)

    return all_results


def stage_stress(
    all_results: dict[str, ModelResults],
    datasets: dict[str, list[EvalExample]],
    args,
) -> dict[str, ModelResults]:
    print("\n" + "=" * 60)
    print("STAGE 4: STRESS TESTS")
    print("=" * 60)

    for mname, mr in all_results.items():
        config = MODELS.get(mname, MODELS["mock"])
        ft = mr.conditions.get("full_trace")
        stress_results = []


        all_examples = []
        for exs in datasets.values():
            all_examples.extend(exs)
        cap = min(len(all_examples), args.max_examples or 50)
        stress_examples = all_examples[:cap]


        print(f"\n  [{mr.model_name}] Stress test 1: Trace ablation")
        ablation_traces = ft.traces[:cap] if ft else None
        st1 = run_trace_ablation(stress_examples, config, ablation_traces)
        stress_results.append(st1)
        print(f"    direct_acc={st1.direct_accuracy:.1%} "
              f"trace_acc={st1.trace_accuracy:.1%} "
              f"change_rate={st1.answer_change_rate:.1%}")


        if ft and ft.traces:
            print(f"  [{mr.model_name}] Stress test 2: Counterfactual editing")
            st2 = run_counterfactual_edit(stress_examples, config, ft.traces[:cap])
            stress_results.append(st2)
            print(f"    cf_change_rate={st2.counterfactual_change_rate:.1%}")


        synth = datasets.get("synthetic", [])
        if synth:
            print(f"  [{mr.model_name}] Stress test 3: Held-out perturbation")

            random.seed(args.seed)
            specs = generate_balanced(min(50, args.n_synthetic))
            traces = generate_traces(specs)
            st3 = run_held_out_eval(traces, n_pairs=min(100, len(traces) * 4))
            stress_results.append(st3)
            print(f"    seen_det={st3.seen_detection_rate:.1%} "
                  f"held_det={st3.held_out_detection_rate:.1%}")


        if ft and ft.traces:
            print(f"  [{mr.model_name}] Stress test 4: Natural rejected")
            gold_answers = [ex.gold_answer for ex in stress_examples]
            st4 = run_natural_rejected_eval(
                stress_examples, ft.traces[:cap], gold_answers[:cap]
            )
            stress_results.append(st4)
            print(f"    valid_acc={st4.valid_trace_accuracy:.1%} "
                  f"invalid_acc={st4.invalid_trace_accuracy:.1%} "
                  f"consistency={st4.consistency_rate:.1%}")

        mr.stress_results = stress_results


        results_dir = Path(args.output_dir) / "results"
        with open(results_dir / f"{mname}_stress.json", "w") as f:
            json.dump([s.to_dict() for s in stress_results], f, indent=2)

    return all_results


def stage_tables(
    all_results: dict[str, ModelResults],
    datasets: dict[str, list[EvalExample]],
    args,
):
    print("\n" + "=" * 60)
    print("STAGE 5: LATEX TABLE GENERATION")
    print("=" * 60)

    tables = {}


    main_rows = []
    for mname, mr in all_results.items():
        diag = mr.diagnostic_report or {}
        ft = mr.conditions.get("full_trace")
        rerank = mr.conditions.get("rerank")

        main_rows.append({
            "model": mr.model_name,
            "answer_acc": (ft.accuracy * 100) if ft else None,
            "state_acc": diag.get("state_accuracy", 0) * 100 if diag else None,
            "transition_acc": diag.get("transition_accuracy", 0) * 100 if diag else None,
            "hidden_incons": diag.get("hidden_inconsistency_rate", 0) * 100 if diag else None,
            "revise_gain": None,
            "rerank_gain": ((rerank.accuracy - ft.accuracy) * 100
                           if rerank and ft else None),
        })
    if main_rows:
        tables["table_main"] = table_main(main_rows)


    failure_data = {}
    total_data = {}
    for mname, mr in all_results.items():
        diag = mr.diagnostic_report or {}
        if diag.get("failure_counts"):
            failure_data[mr.model_name] = diag["failure_counts"]
            total_data[mr.model_name] = diag.get("n_traces", 1)
    if failure_data:
        tables["table_failures"] = table_failures(failure_data, total_data)


    stress_rows = []
    for mname, mr in all_results.items():
        for st in mr.stress_results:
            stress_rows.append(st.to_dict())
    if stress_rows:
        tables["table_stress"] = table_stress(stress_rows)


    if all_results:
        first_mr = list(all_results.values())[0]
        ft = first_mr.conditions.get("full_trace")
        if ft and ft.examples:
            cross_data = {}
            for source in datasets:
                source_indices = [
                    i for i, ex in enumerate(ft.examples)
                    if ex.get("source") == source
                ]
                if source_indices:
                    source_correct = [ft.correct[i] for i in source_indices]
                    source_vrs = [ft.verifier_results[i] for i in source_indices
                                 if i < len(ft.verifier_results) and ft.verifier_results[i]]
                    n = len(source_indices)
                    cross_data[source] = {
                        "answer_acc": sum(source_correct) / n * 100 if n else 0,
                        "state_acc": (sum(1 for v in source_vrs if v.state_ok)
                                     / max(len(source_vrs), 1) * 100),
                        "transition_acc": (sum(1 for v in source_vrs if v.transition_ok)
                                          / max(len(source_vrs), 1) * 100),
                        "hidden_incons": (sum(1 for i_s, v in zip(
                            [ft.correct[i] for i in source_indices[:len(source_vrs)]],
                            source_vrs
                        ) if i_s and (v.state_ok is False or v.transition_ok is False))
                            / max(n, 1) * 100),
                    }
            if cross_data:
                tables["table_cross_source"] = table_cross_source(cross_data)


    output_path = Path(args.output_dir) / "results" / "paper_tables.tex"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_all_tables(tables, str(output_path))


    for name, tex in tables.items():
        with open(output_path.parent / f"{name}.tex", "w") as f:
            f.write(tex)

    print(f"  Generated {len(tables)} tables")


def main():
    parser = argparse.ArgumentParser(
        description="WMW End-to-End Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test run with mock model:
  python scripts/run_evaluation.py --model mock --max-examples 20

  # Full run with GPT-4o:
  python scripts/run_evaluation.py --model gpt4o --max-examples 100

  # Multi-model comparison:
  python scripts/run_evaluation.py --models gpt4o,claude_sonnet --max-examples 50

  # Synthetic only (no external dataset download):
  python scripts/run_evaluation.py --model mock --synthetic-only

  # Generate tables from cached results:
  python scripts/run_evaluation.py --model mock --stages data,tables
        """,
    )
    parser.add_argument("--model", type=str, default="mock",
                        help="Model name (see MODELS dict)")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names (overrides --model)")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Output directory")
    parser.add_argument("--max-examples", type=int, default=50,
                        help="Max examples per source")
    parser.add_argument("--n-synthetic", type=int, default=200,
                        help="Number of synthetic traces")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--synthetic-only", action="store_true",
                        help="Skip external dataset downloads")
    parser.add_argument("--force-download", action="store_true",
                        help="Re-download even if cached")
    parser.add_argument("--stages", type=str, default="data,eval,verify,stress,tables",
                        help="Comma-separated stages to run")
    parser.add_argument("--judge-model", type=str, default="mock",
                        help="Model for LLM judge verifier (default: mock)")
    args = parser.parse_args()

    if args.models is None:
        args.models = args.model

    random.seed(args.seed)
    stages = set(args.stages.split(","))

    print("╔══════════════════════════════════════════════════════════╗")
    print("║      WMW: World Models in Words — Evaluation Pipeline  ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Models:      {args.models:<42s} ║")
    print(f"║  Max/source:  {args.max_examples:<42d} ║")
    print(f"║  Stages:      {args.stages:<42s} ║")
    print(f"║  Output:      {args.output_dir:<42s} ║")
    print("╚══════════════════════════════════════════════════════════╝")


    datasets = {}
    if "data" in stages:
        datasets = stage_data(args)
    else:

        merged_path = Path(args.output_dir) / "eval_data" / "merged_eval.jsonl"
        if merged_path.exists():
            all_ex = load_examples(merged_path)
            for ex in all_ex:
                datasets.setdefault(ex.source, []).append(ex)
        else:
            print("  No cached data found. Run with --stages data first.")
            datasets = {"synthetic": prepare_synthetic(
                Path(args.output_dir) / "eval_data", args.n_synthetic, args.seed
            )}


    all_results = {}
    if "eval" in stages:
        all_results = stage_eval(datasets, args)
    else:

        results_dir = Path(args.output_dir) / "results"
        if results_dir.exists():
            models = args.models.split(",")
            for model_name in models:
                ft_path = results_dir / f"{model_name}_full_trace.json"
                if ft_path.exists():
                    print(f"  Loading cached eval results for {model_name}")
                    config = MODELS.get(model_name, MODELS["mock"])
                    mr = ModelResults(model_name=config.name)

                    with open(ft_path) as f:
                        ft_data = json.load(f)


                    cr = ConditionResult(model=config.name, condition="full_trace")
                    per_ex = ft_data.get("per_example", [])
                    for ex in per_ex:
                        cr.answers.append(ex.get("answer"))
                        cr.correct.append(ex.get("correct", False))
                        cr.parse_statuses.append(ex.get("parse", "ok"))


                    traces_path = results_dir / f"{model_name}_raw_traces.jsonl"
                    if traces_path.exists():
                        raw_traces = []
                        raw_examples = []
                        with open(traces_path) as f:
                            for line in f:
                                rec = json.loads(line.strip())
                                raw_traces.append(rec.get("trace"))
                                raw_examples.append(rec.get("example", {}))
                        cr.traces = raw_traces
                        cr.examples = raw_examples
                        print(f"    Loaded {len(raw_traces)} raw traces for re-verification")
                    else:

                        cr.traces = [None] * len(per_ex)
                        cr.examples = [{}] * len(per_ex)
                        print(f"    WARNING: No raw traces cached. Run --stages eval first to enable re-verification.")

                    mr.conditions["full_trace"] = cr


                    for cond in ["answer_only", "state_to_answer"]:
                        cond_path = results_dir / f"{model_name}_{cond}.json"
                        if cond_path.exists():
                            with open(cond_path) as f:
                                cond_data = json.load(f)
                            cc = ConditionResult(model=config.name, condition=cond)
                            for ex in cond_data.get("per_example", []):
                                cc.answers.append(ex.get("answer"))
                                cc.correct.append(ex.get("correct", False))
                                cc.parse_statuses.append(ex.get("parse", "ok"))
                            mr.conditions[cond] = cc

                    all_results[model_name] = mr
                    print(f"    Loaded {len(per_ex)} examples for {config.name}")

        if not all_results:
            print("  No cached eval results found. Run with --stages eval first.")


    if "verify" in stages and all_results:
        all_results = stage_verify(all_results, args)


    if "stress" in stages and all_results:
        all_results = stage_stress(all_results, datasets, args)


    if "tables" in stages and all_results:
        stage_tables(all_results, datasets, args)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    results_dir = Path(args.output_dir) / "results"
    if results_dir.exists():
        files = list(results_dir.glob("*"))
        print(f"  Output files in {results_dir}:")
        for f in sorted(files):
            size_kb = f.stat().st_size / 1024
            print(f"    {f.name:<40s} {size_kb:>8.1f} KB")


if __name__ == "__main__":
    main()
