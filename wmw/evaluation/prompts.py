from __future__ import annotations
from wmw.datasets.common import EvalExample


_TRACE_SCHEMA_PROMPT = """\
Respond ONLY with a JSON object in this exact format (no markdown fences, no preamble):
{
  "state_0": {
    "objects": [{"name": "...", "attributes": {"mass": ..., ...}}],
    "relations": [{"type": "on|in|attached|above|below|contact|...", "args": ["obj1", "obj2"]}],
    "forces": [{"name": "...", "target": "...", "direction": "...", "magnitude": ..., "unit": "N"}],
    "variables": {"key": value, ...},
    "assumptions": ["..."]
  },
  "transition": {
    "rule": "Name of physical law",
    "effect": "Qualitative or symbolic description of predicted change",
    "equation": "Optional symbolic equation",
    "evidence": ["fact or rule supporting the transition"]
  },
  "state_1": {
    "predicted_change": "What changes from the initial state",
    "new_variables": {"key": value}
  },
  "derivation": "One-to-three sentence reasoning chain: state → rule → result → answer.",
  "answer": {
    "value": "final answer (letter, number, or expression)",
    "unit": "unit if numeric, else null",
    "explanation": "one-sentence justification"
  }
}"""


def build_prompt(
    example: EvalExample,
    condition: str = "full_trace",
    verifier_feedback: str | None = None,
    edited_field: dict | None = None,
    gold_state: dict | None = None,
    gold_transition: dict | None = None,
) -> str:
    q = example.question
    options_str = ""
    if example.options:
        option_letters = "ABCDEFGHIJ"
        opts = [f"({option_letters[i]}) {o}" for i, o in enumerate(example.options)]
        options_str = "\nOptions:\n" + "\n".join(opts)

    if condition == "answer_only":
        return (
            f"Question: {q}{options_str}\n\n"
            f"Provide ONLY the final answer. "
            f"Respond with a JSON object: {{\"answer\": {{\"value\": \"...\"}}}}"
        )

    elif condition == "state_to_answer":
        return (
            f"Question: {q}{options_str}\n\n"
            f"First, describe the initial physical state of the scene "
            f"(objects, forces, relations, variables). "
            f"Then give the final answer.\n\n"
            f"Respond with a JSON object:\n"
            f'{{"state_0": {{...}}, "answer": {{"value": "..."}}}}'
        )

    elif condition == "full_trace":
        return (
            f"Question: {q}{options_str}\n\n"
            f"Reason step by step through the physics.\n\n"
            f"{_TRACE_SCHEMA_PROMPT}"
        )

    elif condition == "gold_state_answer":
        import json
        state_str = json.dumps(gold_state, indent=2) if gold_state else "{}"
        return (
            f"Question: {q}{options_str}\n\n"
            f"The initial physical state is given:\n{state_str}\n\n"
            f"Using this state, determine the answer.\n"
            f"Respond with: {{\"answer\": {{\"value\": \"...\", \"explanation\": \"...\"}}}}"
        )

    elif condition == "gold_trans_answer":
        import json
        trans_str = json.dumps(gold_transition, indent=2) if gold_transition else "{}"
        return (
            f"Question: {q}{options_str}\n\n"
            f"The physical transition is given:\n{trans_str}\n\n"
            f"Using this transition, determine the answer.\n"
            f"Respond with: {{\"answer\": {{\"value\": \"...\", \"explanation\": \"...\"}}}}"
        )

    elif condition == "revise":
        feedback = verifier_feedback or "No specific errors found."
        return (
            f"Question: {q}{options_str}\n\n"
            f"Your previous trace had the following issues:\n{feedback}\n\n"
            f"Please produce a corrected trace.\n\n"
            f"{_TRACE_SCHEMA_PROMPT}"
        )

    elif condition == "counterfactual":
        import json
        edit_str = json.dumps(edited_field, indent=2) if edited_field else "{}"
        return (
            f"Question: {q}{options_str}\n\n"
            f"Suppose the following is true about the physical scene:\n{edit_str}\n\n"
            f"Given this change, what is the answer?\n"
            f"Respond with: {{\"answer\": {{\"value\": \"...\", \"explanation\": \"...\"}}}}"
        )

    elif condition == "ablation":
        return (
            f"Question: {q}{options_str}\n\n"
            f"Answer directly.\n"
            f"Respond with: {{\"answer\": {{\"value\": \"...\"}}}}"
        )

    else:
        raise ValueError(f"Unknown condition: {condition}")


SYSTEM_PROMPTS = {
    "answer_only": (
        "You are a physics expert. Answer the question directly and concisely. "
        "Respond only with valid JSON."
    ),
    "state_to_answer": (
        "You are a physics expert. First describe the physical state, then answer. "
        "Respond only with valid JSON."
    ),
    "full_trace": (
        "You are a physics expert. Analyze the physical scene step by step: "
        "initial state, transition (physical law + effect), resulting state, "
        "a short derivation, then final answer. Respond only with valid JSON."
    ),
    "gold_state_answer": (
        "You are a physics expert. The initial state is provided. "
        "Determine the answer using the given state. Respond only with valid JSON."
    ),
    "gold_trans_answer": (
        "You are a physics expert. The physical transition is provided. "
        "Determine the answer using the given transition. Respond only with valid JSON."
    ),
    "revise": (
        "You are a physics expert. Your previous answer had errors. "
        "Correct the trace and answer. Respond only with valid JSON."
    ),
    "counterfactual": (
        "You are a physics expert. A change has been made to the physical scene. "
        "Determine the new answer. Respond only with valid JSON."
    ),
    "ablation": (
        "You are a physics expert. Answer directly. Respond only with valid JSON."
    ),
}


PAPER_CONDITIONS = [
    "answer_only",
    "state_to_answer",
    "full_trace",
    "revise",

]
