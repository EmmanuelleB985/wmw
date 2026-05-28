from __future__ import annotations
import copy
import json
import random
from dataclasses import dataclass, field
from typing import Any

from wmw.datasets.common import EvalExample
from wmw.evaluation.vlm_caller import ModelConfig, call_vlm
from wmw.evaluation.prompts import build_prompt, SYSTEM_PROMPTS
from wmw.evaluation.trace_parser import parse_trace, extract_answer, answers_match
from wmw.verifiers.pipeline import verify_trace
from wmw.generators.perturbation import get_perturbations, perturb_trace
from wmw.generators.trace_generator import spec_to_trace
from wmw.schemas.models import Trace


@dataclass
class StressTestResult:
    test_name: str
    n_examples: int = 0


    direct_accuracy: float = 0.0
    trace_accuracy: float = 0.0
    answer_change_rate: float = 0.0


    counterfactual_change_rate: float = 0.0


    held_out_detection_rate: float = 0.0
    seen_detection_rate: float = 0.0


    consistency_rate: float = 0.0
    invalid_trace_accuracy: float = 0.0
    valid_trace_accuracy: float = 0.0

    details: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "test_name": self.test_name,
            "n_examples": self.n_examples,
            "direct_accuracy": round(self.direct_accuracy, 4),
            "trace_accuracy": round(self.trace_accuracy, 4),
            "answer_change_rate": round(self.answer_change_rate, 4),
            "counterfactual_change_rate": round(self.counterfactual_change_rate, 4),
            "held_out_detection_rate": round(self.held_out_detection_rate, 4),
            "seen_detection_rate": round(self.seen_detection_rate, 4),
            "consistency_rate": round(self.consistency_rate, 4),
            "invalid_trace_accuracy": round(self.invalid_trace_accuracy, 4),
            "valid_trace_accuracy": round(self.valid_trace_accuracy, 4),
        }


def run_trace_ablation(
    examples: list[EvalExample],
    config: ModelConfig,
    trace_results: list[dict] | None = None,
) -> StressTestResult:
    result = StressTestResult(test_name="trace_ablation", n_examples=len(examples))
    trace_correct = 0
    direct_correct = 0
    answer_changed = 0

    for i, ex in enumerate(examples):

        if trace_results and i < len(trace_results):
            trace_ans = extract_answer(trace_results[i])
        else:
            prompt = build_prompt(ex, condition="full_trace")
            resp = call_vlm(config, SYSTEM_PROMPTS["full_trace"], prompt, ex.image_path)
            td, _ = parse_trace(resp.raw_text)
            trace_ans = extract_answer(td)


        prompt_abl = build_prompt(ex, condition="ablation")
        resp_abl = call_vlm(config, SYSTEM_PROMPTS["ablation"], prompt_abl, ex.image_path)
        td_abl, _ = parse_trace(resp_abl.raw_text)
        direct_ans = extract_answer(td_abl)


        if answers_match(trace_ans, ex.gold_answer):
            trace_correct += 1
        if answers_match(direct_ans, ex.gold_answer):
            direct_correct += 1
        if not answers_match(trace_ans, direct_ans):
            answer_changed += 1

        result.details.append({
            "id": ex.id,
            "trace_answer": str(trace_ans),
            "direct_answer": str(direct_ans),
            "gold": str(ex.gold_answer),
            "changed": not answers_match(trace_ans, direct_ans),
        })

    n = len(examples)
    result.trace_accuracy = trace_correct / n if n else 0
    result.direct_accuracy = direct_correct / n if n else 0
    result.answer_change_rate = answer_changed / n if n else 0
    return result


def run_counterfactual_edit(
    examples: list[EvalExample],
    config: ModelConfig,
    trace_results: list[dict] | None = None,
) -> StressTestResult:
    result = StressTestResult(test_name="counterfactual_edit", n_examples=0)
    changed = 0
    total = 0

    for i, ex in enumerate(examples):
        if not trace_results or i >= len(trace_results):
            continue
        td = trace_results[i]
        if td is None:
            continue


        variables = td.get("state_0", {}).get("variables", {})
        numeric_vars = {k: v for k, v in variables.items()
                       if isinstance(v, (int, float)) and v != 0}
        if not numeric_vars:
            continue


        edit_key = random.choice(list(numeric_vars.keys()))
        edit_val = -numeric_vars[edit_key]
        edited_field = {
            "state_0": {"variables": {edit_key: edit_val}},
            "note": f"Changed {edit_key} from {numeric_vars[edit_key]} to {edit_val}",
        }


        orig_ans = extract_answer(td)


        prompt = build_prompt(ex, condition="counterfactual", edited_field=edited_field)
        resp = call_vlm(config, SYSTEM_PROMPTS["counterfactual"], prompt, ex.image_path)
        cf_td, _ = parse_trace(resp.raw_text)
        cf_ans = extract_answer(cf_td)

        total += 1
        if not answers_match(orig_ans, cf_ans):
            changed += 1

        result.details.append({
            "id": ex.id,
            "edit": f"{edit_key}: {numeric_vars[edit_key]} → {edit_val}",
            "original_answer": str(orig_ans),
            "counterfactual_answer": str(cf_ans),
            "changed": not answers_match(orig_ans, cf_ans),
        })

    result.n_examples = total
    result.counterfactual_change_rate = changed / total if total else 0
    return result


def run_held_out_eval(
    traces: list[Trace],
    n_pairs: int = 200,
) -> StressTestResult:
    result = StressTestResult(test_name="held_out_perturbations")

    seen_perts = get_perturbations(family="seen")
    held_perts = get_perturbations(family="held_out")


    seen_detected = 0
    seen_total = 0
    held_detected = 0
    held_total = 0

    for trace in traces:

        for _ in range(min(2, n_pairs // len(traces) + 1)):
            if seen_total >= n_pairs:
                break
            pert = random.choice(seen_perts)
            pair = perturb_trace(trace, pert)
            vr = verify_trace(pair.rejected)
            seen_total += 1
            if not vr.all_ok:
                seen_detected += 1


        for _ in range(min(2, n_pairs // len(traces) + 1)):
            if held_total >= n_pairs:
                break
            pert = random.choice(held_perts)
            pair = perturb_trace(trace, pert)
            vr = verify_trace(pair.rejected)
            held_total += 1
            if not vr.all_ok:
                held_detected += 1

    result.n_examples = seen_total + held_total
    result.seen_detection_rate = seen_detected / seen_total if seen_total else 0
    result.held_out_detection_rate = held_detected / held_total if held_total else 0

    result.details.append({
        "seen_total": seen_total,
        "seen_detected": seen_detected,
        "held_total": held_total,
        "held_detected": held_detected,
    })
    return result


def run_natural_rejected_eval(
    examples: list[EvalExample],
    trace_results: list[dict],
    gold_answers: list[Any],
) -> StressTestResult:
    result = StressTestResult(test_name="natural_rejected")

    valid_correct = 0
    valid_total = 0
    invalid_correct = 0
    invalid_total = 0

    for i, (td, gold) in enumerate(zip(trace_results, gold_answers)):
        if td is None:
            continue

        pred_ans = extract_answer(td)
        correct = answers_match(pred_ans, gold)


        if "id" not in td:
            td["id"] = examples[i].id if i < len(examples) else f"ex_{i}"
        if "scenario_family" not in td:
            td["scenario_family"] = "unknown"
        if "metadata" not in td:
            td["metadata"] = {"source": "model_generated"}

        vr = verify_trace(td)
        trace_valid = vr.all_ok

        if trace_valid:
            valid_total += 1
            if correct:
                valid_correct += 1
        else:
            invalid_total += 1
            if correct:
                invalid_correct += 1

        result.details.append({
            "id": examples[i].id if i < len(examples) else f"ex_{i}",
            "answer_correct": correct,
            "trace_valid": trace_valid,
            "labels": vr.labels,
        })

    result.n_examples = valid_total + invalid_total
    result.valid_trace_accuracy = valid_correct / valid_total if valid_total else 0
    result.invalid_trace_accuracy = invalid_correct / invalid_total if invalid_total else 0


    result.consistency_rate = result.valid_trace_accuracy - result.invalid_trace_accuracy

    return result
