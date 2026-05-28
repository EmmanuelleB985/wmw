from __future__ import annotations
import json
from typing import Any

from wmw.schemas.models import VerifierResult, FAILURE_LABELS


LLM_JUDGE_SYSTEM = """\
You are a physics verification judge. You will receive a physical reasoning \
trace and must check it for errors. Respond ONLY with valid JSON, no markdown \
fences, no preamble.

The trace has fields: state_0 (initial physical state), transition (physical \
rule and predicted effect), state_1 (resulting state), derivation (short \
reasoning chain), and answer.

Check each field independently for physical errors, then check whether the \
answer follows from the trace. Use ONLY these error labels:
- object: wrong entity tracked or two entities merged
- state: initial physical state is misstated (wrong value, impossible condition)
- relation: spatial relation (contact, support, containment, order) is wrong
- force: a force is missing, reversed, or assigned to the wrong body
- transition: predicted change is invalid given the state and forces
- intervention: effect of an action or counterfactual is wrong
- temporal: before/after frames or states are swapped
- unit_scale: magnitude, unit, or sign convention is inconsistent
- faithfulness: the answer contradicts the model's own trace

Respond with this exact JSON structure:
{
  "labels": ["label1", "label2"],
  "field_errors": {
    "state_0": "description of error or null",
    "transition": "description of error or null",
    "state_1": "description of error or null",
    "derivation": "description of error or null",
    "answer": "description of error or null"
  },
  "answer_trace_consistent": true or false,
  "confidence": "high" or "medium" or "low",
  "reasoning": "one-sentence explanation of your judgment"
}

If the trace is physically correct and internally consistent, return:
{"labels": [], "field_errors": {"state_0": null, "transition": null, \
"state_1": null, "derivation": null, "answer": null}, \
"answer_trace_consistent": true, "confidence": "high", \
"reasoning": "Trace is physically consistent."}
"""


def build_judge_prompt(
    trace_dict: dict,
    question: str = "",
    gold_answer: Any = None,
    include_gold: bool = False,
) -> str:
    parts = []
    if question:
        parts.append(f"Question: {question}")
    if include_gold and gold_answer is not None:
        parts.append(f"Gold answer: {gold_answer}")
    parts.append(f"Trace to verify:\n{json.dumps(trace_dict, indent=2)}")
    return "\n\n".join(parts)


def parse_judge_response(raw_text: str) -> dict:
    text = raw_text.strip()


    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass


    import re
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass


    start = text.find('{')
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except json.JSONDecodeError:
                        break


    return {
        "labels": [],
        "field_errors": {},
        "answer_trace_consistent": None,
        "confidence": "low",
        "reasoning": "Could not parse judge response.",
        "parse_error": True,
    }


def judge_to_verifier_result(judge_output: dict) -> VerifierResult:
    labels = [l for l in judge_output.get("labels", []) if l in FAILURE_LABELS]
    field_errors = judge_output.get("field_errors", {})
    at_consistent = judge_output.get("answer_trace_consistent", None)
    confidence = judge_output.get("confidence", "low")
    reasoning = judge_output.get("reasoning", "")


    state_ok = field_errors.get("state_0") is None
    transition_ok = (field_errors.get("transition") is None and
                     field_errors.get("state_1") is None)
    answer_trace_ok = at_consistent if at_consistent is not None else True


    abstained = confidence == "low" or judge_output.get("parse_error", False)

    details = []
    for field, err in field_errors.items():
        if err:
            details.append(f"llm_judge:{field}: {err}")
    if reasoning:
        details.append(f"llm_judge_reasoning: {reasoning}")
    if not details:
        details.append("llm_judge: no errors found")

    return VerifierResult(
        schema_ok=True,
        state_ok=state_ok if not abstained else None,
        transition_ok=transition_ok if not abstained else None,
        answer_trace_ok=answer_trace_ok if not abstained else None,
        labels=labels,
        abstained=abstained,
        details=details,
    )


def call_llm_judge(
    trace_dict: dict,
    question: str = "",
    gold_answer: Any = None,
    include_gold: bool = False,
    model_config: Any = None,
) -> VerifierResult:
    from wmw.evaluation.vlm_caller import call_vlm, MODELS

    if model_config is None:
        model_config = MODELS.get("mock")

    user_prompt = build_judge_prompt(trace_dict, question, gold_answer, include_gold)
    resp = call_vlm(model_config, LLM_JUDGE_SYSTEM, user_prompt)

    if resp.error:
        return VerifierResult(
            abstained=True,
            details=[f"llm_judge_error: {resp.error}"],
        )

    parsed = parse_judge_response(resp.raw_text)
    return judge_to_verifier_result(parsed)
