from __future__ import annotations
from dataclasses import dataclass, field
from wmw.schemas.models import VerifierResult, FAILURE_LABELS


@dataclass
class DiagnosticReport:
    n_traces: int = 0
    answer_accuracy: float = 0.0
    state_accuracy: float = 0.0
    transition_accuracy: float = 0.0
    hidden_inconsistency_rate: float = 0.0
    trace_answer_consistency: float = 0.0
    abstention_rate: float = 0.0
    failure_counts: dict[str, int] = field(default_factory=dict)
    failure_rates: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"═══ WMW Diagnostic Report ({self.n_traces} traces) ═══",
            f"  Answer accuracy:         {self.answer_accuracy:.1%}",
            f"  State accuracy:          {self.state_accuracy:.1%}",
            f"  Transition accuracy:     {self.transition_accuracy:.1%}",
            f"  Hidden inconsistency:    {self.hidden_inconsistency_rate:.1%}",
            f"  Trace-answer consistency:{self.trace_answer_consistency:.1%}",
            f"  Abstention rate:         {self.abstention_rate:.1%}",
            "",
            "  Failure decomposition:",
        ]
        for lbl in FAILURE_LABELS:
            count = self.failure_counts.get(lbl, 0)
            rate = self.failure_rates.get(lbl, 0)
            lines.append(f"    {lbl:<16s}  {count:>4d}  ({rate:.1%})")
        return "\n".join(lines)


def compute_diagnostics(
    results: list[VerifierResult],
    answer_correct: list[bool] | None = None,
) -> DiagnosticReport:
    n = len(results)
    if n == 0:
        return DiagnosticReport()

    state_ok_count = sum(1 for r in results if r.state_ok is True)
    trans_ok_count = sum(1 for r in results if r.transition_ok is True)
    at_ok_count = sum(1 for r in results if r.answer_trace_ok is True)
    abstain_count = sum(1 for r in results if r.abstained)


    failure_counts: dict[str, int] = {lbl: 0 for lbl in FAILURE_LABELS}
    for r in results:
        for lbl in r.labels:
            if lbl in failure_counts:
                failure_counts[lbl] += 1


    answer_acc = 0.0
    hidden_incon = 0.0
    if answer_correct is not None:
        correct_count = sum(answer_correct)
        answer_acc = correct_count / n


        hidden = 0
        for i, r in enumerate(results):
            if answer_correct[i]:
                trace_invalid = (r.state_ok is False) or (r.transition_ok is False)
                if trace_invalid:
                    hidden += 1
        hidden_incon = hidden / n if n > 0 else 0.0

    report = DiagnosticReport(
        n_traces=n,
        answer_accuracy=answer_acc,
        state_accuracy=state_ok_count / n,
        transition_accuracy=trans_ok_count / n,
        hidden_inconsistency_rate=hidden_incon,
        trace_answer_consistency=at_ok_count / n,
        abstention_rate=abstain_count / n,
        failure_counts=failure_counts,
        failure_rates={lbl: c / n for lbl, c in failure_counts.items()},
    )
    return report


def visual_state_gap(
    acc_gold_state_to_answer: float,
    acc_image_to_state_to_answer: float,
) -> float:
    return acc_gold_state_to_answer - acc_image_to_state_to_answer


def transition_gap(
    acc_gold_transition_to_answer: float,
    acc_state_to_transition_to_answer: float,
) -> float:
    return acc_gold_transition_to_answer - acc_state_to_transition_to_answer


@dataclass
class AgreementMetrics:
    label: str
    precision: float
    recall: float
    f1: float
    n_human: int
    n_verifier: int


def verifier_human_agreement(
    verifier_labels: list[set[str]],
    human_labels: list[set[str]],
    target_label: str,
) -> AgreementMetrics:
    tp = fp = fn = 0
    for v_set, h_set in zip(verifier_labels, human_labels):
        v_has = target_label in v_set
        h_has = target_label in h_set
        if v_has and h_has:
            tp += 1
        elif v_has and not h_has:
            fp += 1
        elif not v_has and h_has:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return AgreementMetrics(
        label=target_label,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        n_human=tp + fn,
        n_verifier=tp + fp,
    )


def rerank_gain(
    baseline_accuracy: float,
    reranked_accuracy: float,
) -> float:
    return reranked_accuracy - baseline_accuracy


def ensemble_detection_gain(
    rules_only_errors: int,
    ensemble_errors: int,
) -> float:
    if rules_only_errors == 0:
        return 1.0 if ensemble_errors > 0 else 0.0
    return (ensemble_errors - rules_only_errors) / rules_only_errors


def preference_pair_stats(pairs: list[dict]) -> dict:
    stats = {
        "total": len(pairs),
        "by_label": {},
        "by_family": {"seen": 0, "held_out": 0},
        "by_field": {},
    }
    for p in pairs:
        lbl = p.get("perturbation_type", "unknown")
        stats["by_label"][lbl] = stats["by_label"].get(lbl, 0) + 1
        fam = p.get("perturbation_family", "unknown")
        stats["by_family"][fam] = stats["by_family"].get(fam, 0) + 1
        fld = p.get("perturbation_field", "unknown")
        stats["by_field"][fld] = stats["by_field"].get(fld, 0) + 1
    return stats
