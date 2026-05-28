from __future__ import annotations
import hashlib
import json
from wmw.schemas.models import Trace, Metadata
from wmw.generators.scenarios import ScenarioSpec


def _content_hash(spec: ScenarioSpec) -> str:
    content = json.dumps({
        "family": spec.family,
        "question": spec.question,
        "params": str(sorted(spec.params.items())),
        "answer": str(spec.answer.value),
    }, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def spec_to_trace(spec: ScenarioSpec, idx: int) -> Trace:
    content_hash = _content_hash(spec)
    trace_id = f"{spec.family}_{idx:04d}_{content_hash}"


    derivation = (
        f"Given {spec.transition.evidence[0] if spec.transition.evidence else 'the initial conditions'}, "
        f"apply {spec.transition.rule}. "
        f"{spec.transition.effect}. "
        f"Therefore the answer is {spec.answer.value}"
        f"{' ' + spec.answer.unit if spec.answer.unit else ''}."
    )

    return Trace(
        id=trace_id,
        scenario_family=spec.family,
        question=spec.question,
        state_0=spec.state,
        transition=spec.transition,
        state_1=spec.result,
        answer=spec.answer,
        derivation=derivation,
        metadata=Metadata(
            difficulty=spec.difficulty,
            task_type=spec.task_type,
            gold_answer=spec.answer.value,
            gold_state_hash=content_hash,
            source="synthetic",
        ),
    )


def generate_traces(specs: list[ScenarioSpec], deduplicate: bool = True) -> list[Trace]:
    traces = []
    seen_questions: set[str] = set()
    counters: dict[str, int] = {}

    for spec in specs:
        if deduplicate:
            q_norm = spec.question.strip().lower()
            if q_norm in seen_questions:
                continue
            seen_questions.add(q_norm)

        counters.setdefault(spec.family, 0)
        counters[spec.family] += 1
        traces.append(spec_to_trace(spec, counters[spec.family]))

    return traces
