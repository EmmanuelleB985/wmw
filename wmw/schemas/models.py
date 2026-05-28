from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PhysicalObject:
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Relation:
    type: str
    args: tuple[str, str] = ("", "")


@dataclass
class Force:
    name: str
    target: str
    direction: str
    magnitude: float | str | None = None
    unit: str = "N"


@dataclass
class State:
    objects: list[PhysicalObject] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    forces: list[Force] = field(default_factory=list)
    variables: dict[str, Any] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)

    def object_names(self) -> set[str]:
        return {o.name for o in self.objects}


@dataclass
class Transition:
    rule: str
    effect: str
    equation: str | None = None
    evidence: list[str] = field(default_factory=list)


@dataclass
class ResultingState:
    predicted_change: str
    new_variables: dict[str, Any] = field(default_factory=dict)


@dataclass
class Answer:
    value: str | float
    unit: str | None = None
    explanation: str | None = None


@dataclass
class Metadata:
    difficulty: str = "medium"
    task_type: str = "transition_prediction"
    gold_answer: Any = None
    gold_state_hash: str | None = None
    source: str = "synthetic"


@dataclass
class VerifierResult:
    schema_ok: bool = True
    state_ok: bool | None = None
    transition_ok: bool | None = None
    answer_trace_ok: bool | None = None
    labels: list[str] = field(default_factory=list)
    abstained: bool = False
    details: list[str] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        checks = [self.schema_ok, self.state_ok, self.transition_ok, self.answer_trace_ok]
        return all(c is True for c in checks if c is not None)

    @property
    def violation_count(self) -> int:
        return len(self.labels)


@dataclass
class Trace:
    id: str
    scenario_family: str
    question: str
    state_0: State
    transition: Transition
    state_1: ResultingState
    answer: Answer
    derivation: str | None = None
    metadata: Metadata = field(default_factory=Metadata)
    verifier: VerifierResult | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "scenario_family": self.scenario_family,
            "question": self.question,
            "state_0": {
                "objects": [{"name": o.name, "attributes": o.attributes} for o in self.state_0.objects],
                "relations": [{"type": r.type, "args": list(r.args)} for r in self.state_0.relations],
                "forces": [
                    {"name": f.name, "target": f.target, "direction": f.direction,
                     "magnitude": f.magnitude, "unit": f.unit}
                    for f in self.state_0.forces
                ],
                "variables": self.state_0.variables,
                "assumptions": self.state_0.assumptions,
            },
            "transition": {
                "rule": self.transition.rule,
                "effect": self.transition.effect,
                "equation": self.transition.equation,
                "evidence": self.transition.evidence,
            },
            "state_1": {
                "predicted_change": self.state_1.predicted_change,
                "new_variables": self.state_1.new_variables,
            },
            "derivation": self.derivation,
            "answer": {
                "value": self.answer.value,
                "unit": self.answer.unit,
                "explanation": self.answer.explanation,
            },
            "metadata": {
                "difficulty": self.metadata.difficulty,
                "task_type": self.metadata.task_type,
                "gold_answer": self.metadata.gold_answer,
                "gold_state_hash": self.metadata.gold_state_hash,
                "source": self.metadata.source,
            },
            "verifier": {
                "schema_ok": self.verifier.schema_ok,
                "state_ok": self.verifier.state_ok,
                "transition_ok": self.verifier.transition_ok,
                "answer_trace_ok": self.verifier.answer_trace_ok,
                "labels": self.verifier.labels,
                "abstained": self.verifier.abstained,
                "details": self.verifier.details,
            } if self.verifier else None,
        }


FAILURE_LABELS = [
    "object", "state", "relation", "force",
    "transition", "intervention", "temporal",
    "unit_scale", "faithfulness",
]

@dataclass
class PreferencePair:
    id: str
    source_trace_id: str
    chosen: dict
    rejected: dict
    perturbation_type: str
    perturbation_field: str
    perturbation_family: str
    description: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_trace_id": self.source_trace_id,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "perturbation_type": self.perturbation_type,
            "perturbation_field": self.perturbation_field,
            "perturbation_family": self.perturbation_family,
            "description": self.description,
        }
