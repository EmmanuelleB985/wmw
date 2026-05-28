from __future__ import annotations
import json
from typing import Any

from wmw.evaluation.vlm_caller import ModelConfig, VLMResponse, call_vlm
from wmw.evaluation.prompts import build_prompt, SYSTEM_PROMPTS
from wmw.evaluation.trace_parser import parse_trace, extract_answer
from wmw.verifiers.pipeline import verify_trace
from wmw.datasets.common import EvalExample


def _score_trace(trace_dict: dict | None) -> float:
    if trace_dict is None:
        return -10.0

    result = verify_trace(trace_dict)
    score = 0.0
    if result.schema_ok:
        score += 1.0
    if result.state_ok is True:
        score += 2.0
    elif result.state_ok is None:
        score += 0.5
    if result.transition_ok is True:
        score += 2.0
    elif result.transition_ok is None:
        score += 0.5
    if result.answer_trace_ok is True:
        score += 2.0
    elif result.answer_trace_ok is None:
        score += 0.5

    score -= 0.5 * len(result.labels)
    return score


def _score_trace_llm(trace_dict: dict | None, question: str = "",
                     judge_config: "ModelConfig | None" = None) -> float:
    if trace_dict is None:
        return -10.0

    from wmw.verifiers.llm_judge import call_llm_judge
    vr = call_llm_judge(trace_dict, question=question, model_config=judge_config)

    if vr.abstained:
        return 0.0

    score = 5.0
    score -= 1.0 * len(vr.labels)
    if vr.state_ok is False:
        score -= 1.5
    if vr.transition_ok is False:
        score -= 1.5
    if vr.answer_trace_ok is False:
        score -= 2.0
    return score


def _score_trace_ensemble(trace_dict: dict | None, question: str = "",
                          judge_config: "ModelConfig | None" = None) -> float:
    rules_score = _score_trace(trace_dict)
    judge_score = _score_trace_llm(trace_dict, question, judge_config)
    return rules_score + judge_score


def rerank_traces(
    example: EvalExample,
    config: ModelConfig,
    k: int = 5,
    temperature: float = 0.7,
    scoring: str = "rules",
    judge_config: "ModelConfig | None" = None,
) -> tuple[dict | None, list[dict], list[float]]:
    prompt = build_prompt(example, condition="full_trace")
    system = SYSTEM_PROMPTS["full_trace"]


    sample_config = ModelConfig(
        name=config.name,
        provider=config.provider,
        model_id=config.model_id,
        api_key_env=config.api_key_env,
        base_url=config.base_url,
        max_tokens=config.max_tokens,
        temperature=temperature,
        timeout=config.timeout,
    )

    traces = []
    scores = []
    for _ in range(k):
        resp = call_vlm(sample_config, system, prompt, example.image_path)
        td, status = parse_trace(resp.raw_text)


        if td and "id" not in td:
            td["id"] = example.id
        if td and "scenario_family" not in td:
            td["scenario_family"] = example.topic or "unknown"
        if td and "question" not in td:
            td["question"] = example.question
        if td and "metadata" not in td:
            td["metadata"] = {
                "difficulty": example.difficulty,
                "task_type": example.task_type,
                "source": example.source,
            }


        if scoring == "llm_judge":
            score = _score_trace_llm(td, question=example.question, judge_config=judge_config)
        elif scoring == "ensemble":
            score = _score_trace_ensemble(td, question=example.question, judge_config=judge_config)
        elif scoring == "majority_vote":
            score = 0.0
        else:
            score = _score_trace(td)

        traces.append(td)
        scores.append(score)

    if not traces:
        return None, [], []


    if scoring == "majority_vote":
        from collections import Counter
        answers = [extract_answer(t) for t in traces]
        answer_counts = Counter(str(a) for a in answers if a is not None)
        if answer_counts:
            best_answer = answer_counts.most_common(1)[0][0]

            for i, t in enumerate(traces):
                if str(extract_answer(t)) == best_answer:
                    scores[i] = 10.0
                else:
                    scores[i] = 0.0

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return traces[best_idx], traces, scores


def revise_with_feedback(
    example: EvalExample,
    config: ModelConfig,
    original_trace: dict,
) -> tuple[dict | None, str]:
    result = verify_trace(original_trace)

    if result.all_ok:
        return original_trace, "No errors found; trace unchanged."


    feedback_lines = []
    for detail in result.details:
        if "passed" not in detail.lower():
            feedback_lines.append(f"- {detail}")
    if result.labels:
        feedback_lines.append(f"Error types detected: {', '.join(result.labels)}")

    feedback = "\n".join(feedback_lines) if feedback_lines else "Minor issues detected."

    prompt = build_prompt(example, condition="revise", verifier_feedback=feedback)
    system = SYSTEM_PROMPTS["revise"]
    resp = call_vlm(config, system, prompt, example.image_path)
    revised, status = parse_trace(resp.raw_text)

    return revised, feedback
