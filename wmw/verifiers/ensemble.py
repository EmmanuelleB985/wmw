from __future__ import annotations
from dataclasses import dataclass, field
from wmw.schemas.models import VerifierResult, FAILURE_LABELS


@dataclass
class EnsembleResult:
    rules: VerifierResult
    llm_judge: VerifierResult
    merged: VerifierResult

    rules_only_labels: list[str] = field(default_factory=list)
    judge_only_labels: list[str] = field(default_factory=list)
    both_labels: list[str] = field(default_factory=list)
    neither_labels: list[str] = field(default_factory=list)


def merge_verifier_results(
    rules: VerifierResult,
    llm_judge: VerifierResult,
) -> EnsembleResult:

    all_labels = list(dict.fromkeys(rules.labels + llm_judge.labels))


    rules_set = set(rules.labels)
    judge_set = set(llm_judge.labels)
    rules_only = [l for l in rules.labels if l not in judge_set]
    judge_only = [l for l in llm_judge.labels if l not in rules_set]
    both = [l for l in all_labels if l in rules_set and l in judge_set]


    def _merge_ok(r, j):
        if r is False or j is False:
            return False
        if r is True and j is True:
            return True
        if r is True or j is True:
            return True
        return None

    merged = VerifierResult(
        schema_ok=rules.schema_ok,
        state_ok=_merge_ok(rules.state_ok, llm_judge.state_ok),
        transition_ok=_merge_ok(rules.transition_ok, llm_judge.transition_ok),
        answer_trace_ok=_merge_ok(rules.answer_trace_ok, llm_judge.answer_trace_ok),
        labels=all_labels,
        abstained=rules.abstained and llm_judge.abstained,
        details=rules.details + llm_judge.details,
    )

    return EnsembleResult(
        rules=rules,
        llm_judge=llm_judge,
        merged=merged,
        rules_only_labels=rules_only,
        judge_only_labels=judge_only,
        both_labels=both,
    )


@dataclass
class EnsembleStats:
    n_traces: int = 0
    rules_only_count: int = 0
    judge_only_count: int = 0
    both_count: int = 0
    neither_count: int = 0
    ensemble_detection_gain: float = 0.0


    by_label: dict[str, dict[str, int]] = field(default_factory=dict)


def compute_ensemble_stats(
    ensemble_results: list[EnsembleResult],
    human_labels: list[set[str]] | None = None,
) -> EnsembleStats:
    stats = EnsembleStats(n_traces=len(ensemble_results))

    for lbl in FAILURE_LABELS:
        stats.by_label[lbl] = {
            "rules_only": 0, "judge_only": 0, "both": 0, "neither": 0,
        }

    for i, er in enumerate(ensemble_results):
        stats.rules_only_count += len(er.rules_only_labels)
        stats.judge_only_count += len(er.judge_only_labels)
        stats.both_count += len(er.both_labels)

        for lbl in er.rules_only_labels:
            if lbl in stats.by_label:
                stats.by_label[lbl]["rules_only"] += 1
        for lbl in er.judge_only_labels:
            if lbl in stats.by_label:
                stats.by_label[lbl]["judge_only"] += 1
        for lbl in er.both_labels:
            if lbl in stats.by_label:
                stats.by_label[lbl]["both"] += 1


        if human_labels and i < len(human_labels):
            ensemble_set = set(er.merged.labels)
            for lbl in human_labels[i]:
                if lbl not in ensemble_set:
                    stats.neither_count += 1
                    if lbl in stats.by_label:
                        stats.by_label[lbl]["neither"] += 1


    rules_total = stats.rules_only_count + stats.both_count
    ensemble_total = rules_total + stats.judge_only_count
    if rules_total > 0:
        stats.ensemble_detection_gain = stats.judge_only_count / rules_total
    elif stats.judge_only_count > 0:
        stats.ensemble_detection_gain = 1.0

    return stats
