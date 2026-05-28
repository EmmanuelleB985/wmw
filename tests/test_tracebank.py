import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wmw.generators.scenarios import (
    generate_scenario, generate_balanced, SCENARIO_FAMILIES,
)
from wmw.generators.trace_generator import spec_to_trace, generate_traces
from wmw.generators.perturbation import (
    generate_preference_pairs, get_perturbations, perturb_trace, perturbation_stats,
)
from wmw.verifiers.schema_verifier import verify_schema
from wmw.verifiers.state_verifier import verify_state
from wmw.verifiers.transition_verifier import verify_transition
from wmw.verifiers.pipeline import verify_trace
from wmw.schemas.models import Trace, VerifierResult, PreferencePair, FAILURE_LABELS
from wmw.metrics import compute_diagnostics, visual_state_gap, transition_gap


def test_all_families_generate():
    random.seed(42)
    for family in SCENARIO_FAMILIES:
        spec = generate_scenario(family)
        assert spec.family == family
        assert spec.question
        assert spec.state.objects
        assert spec.transition.rule
        assert spec.answer.value is not None


def test_balanced_generation():
    random.seed(42)
    specs = generate_balanced(64)
    assert len(specs) == 64
    families_seen = {s.family for s in specs}
    assert families_seen == set(SCENARIO_FAMILIES.keys())


def test_generate_at_scale():
    random.seed(42)
    specs = generate_balanced(500)
    assert len(specs) == 500


def test_spec_to_trace():
    random.seed(42)
    spec = generate_scenario("inclined_plane")
    trace = spec_to_trace(spec, 1)
    assert trace.id.startswith("inclined_plane_0001_")
    assert len(trace.id.split("_")) >= 3
    assert trace.scenario_family == "inclined_plane"
    assert trace.metadata.gold_state_hash is not None
    d = trace.to_dict()
    assert "state_0" in d
    assert "transition" in d
    assert "answer" in d

    json.dumps(d)


def test_trace_serialization_roundtrip():
    random.seed(42)
    specs = generate_balanced(32)
    traces = generate_traces(specs)
    for t in traces:
        d = t.to_dict()
        s = json.dumps(d)
        recovered = json.loads(s)
        assert recovered["id"] == t.id
        assert recovered["scenario_family"] == t.scenario_family


def test_positive_traces_pass_schema():
    random.seed(42)
    specs = generate_balanced(32)
    traces = generate_traces(specs)
    for t in traces:
        r = verify_schema(t.to_dict())
        assert r.schema_ok, f"Schema failed for {t.id}: {r.details}"


def test_broken_trace_fails_schema():
    r = verify_schema({"id": "bad_trace"})
    assert not r.schema_ok


def test_positive_traces_pass_state():
    random.seed(42)
    specs = generate_balanced(32)
    traces = generate_traces(specs)
    pass_count = 0
    for t in traces:
        r = verify_state(t.to_dict())
        if r.state_ok:
            pass_count += 1

    assert pass_count >= 24, f"Only {pass_count}/32 passed state verification"


def test_positive_traces_pass_transition():
    random.seed(42)
    specs = generate_balanced(32)
    traces = generate_traces(specs)
    pass_count = 0
    for t in traces:
        r = verify_transition(t.to_dict())
        if r.transition_ok:
            pass_count += 1
    assert pass_count >= 24, f"Only {pass_count}/32 passed transition verification"


def test_full_pipeline_on_positive():
    random.seed(42)
    specs = generate_balanced(16)
    traces = generate_traces(specs)
    for t in traces:
        r = verify_trace(t)
        assert r.schema_ok


def test_perturbation_registry():
    all_p = get_perturbations()
    assert len(all_p) >= 20
    seen = get_perturbations(family="seen")
    held = get_perturbations(family="held_out")
    assert len(seen) >= 10
    assert len(held) >= 5

    labels_covered = {p.label for p in all_p}
    assert labels_covered == set(FAILURE_LABELS)


def test_perturb_trace():
    random.seed(42)
    spec = generate_scenario("collision")
    trace = spec_to_trace(spec, 1)
    pair = perturb_trace(trace)
    assert isinstance(pair, PreferencePair)
    assert pair.source_trace_id == trace.id
    assert pair.perturbation_type in FAILURE_LABELS
    assert pair.perturbation_family in ("seen", "held_out")

    json.dumps(pair.to_dict())


def test_preference_pair_generation_at_scale():
    random.seed(42)
    specs = generate_balanced(200)
    traces = generate_traces(specs)
    pairs = generate_preference_pairs(traces, pairs_per_trace=16)

    assert len(pairs) >= 3200 * 0.95, f"Too few pairs: {len(pairs)} (expected ~3200)"
    assert len(pairs) <= 3200, f"Too many pairs: {len(pairs)}"


    stats = perturbation_stats(pairs)
    for lbl in FAILURE_LABELS:
        count = stats["by_label"].get(lbl, 0)
        assert count > 0, f"No pairs for label '{lbl}'"

        mean = len(pairs) / len(FAILURE_LABELS)
        assert count > mean * 0.2, f"Label '{lbl}' severely underrepresented: {count}"


def test_preference_pair_serialization():
    random.seed(42)
    spec = generate_scenario("collision")
    trace = spec_to_trace(spec, 1)
    pair = perturb_trace(trace)
    d = pair.to_dict()
    s = json.dumps(d)
    recovered = json.loads(s)
    assert recovered["perturbation_type"] == pair.perturbation_type


def test_compute_diagnostics():
    results = [
        VerifierResult(schema_ok=True, state_ok=True, transition_ok=True, answer_trace_ok=True),
        VerifierResult(schema_ok=True, state_ok=False, transition_ok=True, answer_trace_ok=True, labels=["state"]),
        VerifierResult(schema_ok=True, state_ok=True, transition_ok=False, answer_trace_ok=True, labels=["transition"]),
    ]
    answer_correct = [True, True, False]
    report = compute_diagnostics(results, answer_correct)
    assert report.n_traces == 3
    assert 0 <= report.answer_accuracy <= 1
    assert 0 <= report.hidden_inconsistency_rate <= 1
    assert report.failure_counts["state"] == 1
    assert report.failure_counts["transition"] == 1


def test_gap_metrics():
    assert abs(visual_state_gap(0.9, 0.7) - 0.2) < 1e-9
    assert abs(transition_gap(0.85, 0.65) - 0.2) < 1e-9


def test_verifier_detects_some_perturbations():
    random.seed(42)
    specs = generate_balanced(20)
    traces = generate_traces(specs)
    pairs = generate_preference_pairs(traces, pairs_per_trace=8)

    detected = 0
    for pair in pairs:
        r = verify_trace(pair.rejected)
        if not r.all_ok:
            detected += 1

    rate = detected / len(pairs)
    assert rate > 0.1, f"Verifier only detected {rate:.1%} of perturbations"


def test_no_noop_pairs():
    random.seed(42)
    specs = generate_balanced(50)
    traces = generate_traces(specs)
    pairs = generate_preference_pairs(traces, pairs_per_trace=8)
    noop = 0
    for p in pairs:
        if json.dumps(p.chosen, sort_keys=True) == json.dumps(p.rejected, sort_keys=True):
            noop += 1
    noop_rate = noop / len(pairs) if pairs else 0
    assert noop_rate < 0.02, f"{noop}/{len(pairs)} ({noop_rate:.1%}) pairs are no-ops"


def test_no_duplicate_questions():
    random.seed(42)
    specs = generate_balanced(220)
    traces = generate_traces(specs, deduplicate=True)
    questions = [t.question.strip().lower() for t in traces]
    assert len(questions) == len(set(questions)),\
        f"Duplicate questions found: {len(questions)} total, {len(set(questions))} unique"


def test_content_addressed_ids():
    random.seed(42)
    spec = generate_scenario("collision")
    t1 = spec_to_trace(spec, 1)
    t2 = spec_to_trace(spec, 1)

    hash1 = t1.id.split("_")[-1]
    hash2 = t2.id.split("_")[-1]
    assert hash1 == hash2, f"Same content should produce same hash: {hash1} != {hash2}"
    assert len(hash1) == 12, f"Hash should be 12 hex chars, got {len(hash1)}"


def test_ids_globally_unique():
    random.seed(42)
    specs = generate_balanced(200)
    traces = generate_traces(specs)
    ids = [t.id for t in traces]
    assert len(ids) == len(set(ids)), "Duplicate trace IDs found"


def test_perturbation_applicability_guard():
    from wmw.generators.perturbation import _applicable_perturbations
    random.seed(42)

    spec = generate_scenario("wave")
    trace = spec_to_trace(spec, 1)
    applicable = _applicable_perturbations(trace)
    all_perts = get_perturbations()
    assert len(applicable) < len(all_perts),\
        "Wave scenario should not support all perturbations"

    spec2 = generate_scenario("collision")
    trace2 = spec_to_trace(spec2, 1)
    applicable2 = _applicable_perturbations(trace2)
    assert len(applicable2) > len(applicable),\
        "Collision should support more perturbations than wave"


def test_gold_answer_matches_trace_answer():
    random.seed(42)
    specs = generate_balanced(100)
    traces = generate_traces(specs)
    for t in traces:
        gold = t.metadata.gold_answer
        trace_answer = t.answer.value
        assert str(gold) == str(trace_answer),\
            f"{t.id}: gold={gold} != trace_answer={trace_answer}"


if __name__ == "__main__":

    test_fns = [v for k, v in globals().items() if k.startswith("test_")]
    passed = failed = 0
    for fn in test_fns:
        try:
            fn()
            print(f"  ✓ {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {fn.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")
    sys.exit(1 if failed else 0)
