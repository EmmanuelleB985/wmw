from __future__ import annotations
from wmw.schemas.models import Trace, VerifierResult
from wmw.verifiers.schema_verifier import verify_schema
from wmw.verifiers.state_verifier import verify_state
from wmw.verifiers.transition_verifier import verify_transition


def verify_trace(trace: Trace | dict, gold_state: dict | None = None) -> VerifierResult:
    td = trace.to_dict() if isinstance(trace, Trace) else trace


    schema_result = verify_schema(td)
    if not schema_result.schema_ok:

        return VerifierResult(
            schema_ok=False,
            state_ok=None,
            transition_ok=None,
            answer_trace_ok=None,
            labels=schema_result.labels,
            abstained=False,
            details=["SCHEMA FAILED — skipping deeper checks"] + schema_result.details,
        )


    state_result = verify_state(td, gold=gold_state)


    transition_result = verify_transition(td)


    all_labels = list(dict.fromkeys(
        state_result.labels + transition_result.labels
    ))
    all_details = (
        schema_result.details +
        state_result.details +
        transition_result.details
    )
    abstained = state_result.abstained or transition_result.abstained

    return VerifierResult(
        schema_ok=True,
        state_ok=state_result.state_ok,
        transition_ok=transition_result.transition_ok,
        answer_trace_ok=transition_result.answer_trace_ok,
        labels=all_labels,
        abstained=abstained,
        details=all_details,
    )


def verify_traces(traces: list[Trace | dict], gold_states: list[dict] | None = None) -> list[VerifierResult]:
    golds = gold_states or [None] * len(traces)
    return [verify_trace(t, g) for t, g in zip(traces, golds)]
