import json
import math
import os
import random
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


class TestSchemaIntegrity:

    @pytest.fixture(autouse=True)
    def setup(self):
        random.seed(2026)
        from wmw.generators.scenarios import generate_balanced
        from wmw.generators.trace_generator import generate_traces
        self.specs = generate_balanced(100)
        self.traces = generate_traces(self.specs, deduplicate=True)

    def test_all_traces_have_required_fields(self):
        required = ["id", "scenario_family", "question", "state_0",
                     "transition", "state_1", "answer", "derivation", "metadata"]
        for t in self.traces:
            td = t.to_dict()
            for field in required:
                assert field in td, f"Missing {field} in {t.id}"

    def test_all_traces_validate_schema(self):
        from wmw.verifiers.schema_verifier import verify_schema
        for t in self.traces:
            td = t.to_dict()
            result = verify_schema(td)
            assert result.schema_ok, f"Schema fail: {t.id}: {result.details}"

    def test_state_0_has_objects(self):
        for t in self.traces:
            assert len(t.state_0.objects) > 0, f"No objects in {t.id}"

    def test_transition_has_rule(self):
        for t in self.traces:
            assert t.transition.rule, f"Empty rule in {t.id}"

    def test_answer_has_value(self):
        for t in self.traces:
            assert t.answer.value is not None, f"No answer value in {t.id}"

    def test_derivation_is_nonempty(self):
        for t in self.traces:
            assert t.derivation and len(t.derivation) > 10,\
                f"Empty/short derivation in {t.id}"

    def test_metadata_gold_answer_matches(self):
        for t in self.traces:
            assert str(t.metadata.gold_answer) == str(t.answer.value),\
                f"Gold mismatch in {t.id}: {t.metadata.gold_answer} != {t.answer.value}"


class TestPhysicsCorrectness:

    def test_free_fall_time(self):
        from wmw.generators.scenarios import generate_scenario
        random.seed(42)
        for _ in range(10):
            spec = generate_scenario("free_fall")
            h = spec.params["h"]
            t_gold = spec.answer.value

            if spec.answer.unit == "s":
                t_expected = round(math.sqrt(2 * h / 9.8), 2)
                assert abs(t_gold - t_expected) < 0.1,\
                    f"Free fall time: h={h}, gold={t_gold}, expected={t_expected}"
            elif spec.answer.unit == "m/s":
                v_expected = round(math.sqrt(2 * 9.8 * h), 2)
                assert abs(t_gold - v_expected) < 0.1,\
                    f"Free fall velocity: h={h}, gold={t_gold}, expected={v_expected}"

    def test_projectile_range(self):
        from wmw.generators.scenarios import generate_scenario
        random.seed(42)
        for _ in range(10):
            spec = generate_scenario("projectile")
            v0 = spec.params["v0"]
            theta = spec.params.get("angle", spec.params.get("theta", 45))
            val = spec.answer.value

            assert val > 0, f"Projectile answer ≤ 0: {val}"

            max_possible = v0**2 / 9.8
            assert val <= max_possible * 1.1,\
                f"Projectile answer exceeds max range: {val} > {max_possible}"

    def test_spring_energy_or_period(self):
        from wmw.generators.scenarios import generate_scenario
        random.seed(42)
        for _ in range(10):
            spec = generate_scenario("spring")
            k = spec.params["k"]
            mass = spec.params["mass"]
            x = spec.params.get("x", 0)
            val = spec.answer.value
            assert val > 0, f"Spring answer ≤ 0: {val}"

            if spec.answer.unit == "J":
                pe_expected = round(0.5 * k * x**2, 4)
                assert abs(val - pe_expected) < 0.1,\
                    f"Spring PE: k={k}, x={x}, gold={val}, expected={pe_expected}"
            elif spec.answer.unit == "s":
                T_expected = round(2 * math.pi * math.sqrt(mass / k), 4)
                assert abs(val - T_expected) < 0.01,\
                    f"Spring T: m={mass}, k={k}, gold={val}, expected={T_expected}"

    def test_circuit_current(self):
        from wmw.generators.scenarios import generate_scenario
        random.seed(42)
        for _ in range(10):
            spec = generate_scenario("circuit")
            V = spec.params["V"]
            R1 = spec.params["R1"]
            R2 = spec.params["R2"]
            is_series = spec.params.get("series", spec.params.get("config", "series"))
            if is_series is True or is_series == "series":
                R_total = R1 + R2
            else:
                R_total = (R1 * R2) / (R1 + R2)
            I_expected = round(V / R_total, 4)
            I_gold = spec.answer.value
            assert abs(I_gold - I_expected) < 0.01,\
                f"Circuit: V={V}, R1={R1}, R2={R2}, gold={I_gold}, expected={I_expected}"


class TestDiagramGeneration:

    def test_all_families_render(self):
        from wmw.generators.scenarios import generate_scenario, SCENARIO_FAMILIES
        from wmw.generators.trace_generator import spec_to_trace
        from wmw.diagrams.renderer import render_trace_diagram

        random.seed(42)
        for family in SCENARIO_FAMILIES:
            spec = generate_scenario(family)
            trace = spec_to_trace(spec, 1)
            td = trace.to_dict()
            path = render_trace_diagram(td, output_dir="/tmp/wmw_diagram_test")
            assert path is not None, f"No diagram for {family}"
            assert path.exists(), f"Diagram file missing for {family}"
            assert path.stat().st_size > 1000, f"Diagram too small for {family}: {path.stat().st_size}B"

    def test_diagram_is_valid_png(self):
        from wmw.generators.scenarios import generate_scenario
        from wmw.generators.trace_generator import spec_to_trace
        from wmw.diagrams.renderer import render_trace_diagram

        random.seed(42)
        spec = generate_scenario("inclined_plane")
        trace = spec_to_trace(spec, 1)
        path = render_trace_diagram(trace.to_dict(), output_dir="/tmp/wmw_diagram_test")
        with open(path, "rb") as f:
            header = f.read(8)

        assert header[:4] == b'\x89PNG', "Not a valid PNG file"


class TestQuestionDiversity:

    def test_paraphrase_produces_variants(self):
        from wmw.generators.paraphrases import paraphrase_question
        random.seed(42)
        params = {"mass_kg": 5, "incline_angle_deg": 30, "friction_coeff": 0}
        questions = set()
        for _ in range(20):
            q = paraphrase_question("inclined_plane", params)
            questions.add(q)
        assert len(questions) >= 3, f"Only {len(questions)} variants for inclined_plane"

    def test_paraphrases_across_families(self):
        from wmw.generators.paraphrases import _TEMPLATE_REGISTRY

        assert len(_TEMPLATE_REGISTRY) >= 9

    def test_500_traces_high_uniqueness(self):
        from wmw.generators.scenarios import generate_balanced
        from wmw.generators.trace_generator import generate_traces
        from wmw.generators.paraphrases import paraphrase_question

        random.seed(2026)
        specs = generate_balanced(575)
        traces = generate_traces(specs, deduplicate=True)[:500]

        for t in traces:
            params = t.state_0.variables.copy()
            for obj in t.state_0.objects:
                for k, v in obj.attributes.items():
                    if k not in params:
                        params[k] = v
            alt = paraphrase_question(t.scenario_family, params)
            if alt:
                t.question = alt

        questions = [t.question for t in traces]
        unique = len(set(questions))
        ratio = unique / len(questions)
        assert ratio >= 0.80, f"Only {ratio:.1%} unique questions at 500 traces"


class TestPreferencePairQuality:

    @pytest.fixture(autouse=True)
    def setup(self):
        random.seed(2026)
        from wmw.generators.scenarios import generate_balanced
        from wmw.generators.trace_generator import generate_traces
        from wmw.generators.perturbation import generate_preference_pairs
        specs = generate_balanced(50)
        self.traces = generate_traces(specs, deduplicate=True)
        self.pairs = generate_preference_pairs(self.traces, pairs_per_trace=8)

    def test_zero_noop_pairs(self):
        noop = sum(1 for p in self.pairs
                   if json.dumps(p.chosen, sort_keys=True) == json.dumps(p.rejected, sort_keys=True))
        assert noop == 0, f"{noop} no-op pairs found"

    def test_all_labels_represented(self):
        from wmw.schemas.models import FAILURE_LABELS
        from wmw.generators.perturbation import perturbation_stats
        stats = perturbation_stats(self.pairs)
        for lbl in FAILURE_LABELS:
            assert stats["by_label"].get(lbl, 0) > 0,\
                f"Label '{lbl}' not represented in pairs"

    def test_pair_ids_unique(self):
        ids = [p.id for p in self.pairs]
        assert len(ids) == len(set(ids)), "Duplicate pair IDs"

    def test_chosen_has_gold_structure(self):
        for p in self.pairs[:20]:
            assert "state_0" in p.chosen
            assert "transition" in p.chosen
            assert "answer" in p.chosen

    def test_rejected_differs_from_chosen(self):
        for p in self.pairs[:20]:
            assert json.dumps(p.chosen, sort_keys=True) != json.dumps(p.rejected, sort_keys=True)


class TestSplitIntegrity:

    def test_no_trace_leakage(self):
        random.seed(2026)
        from wmw.generators.scenarios import generate_balanced
        from wmw.generators.trace_generator import generate_traces

        specs = generate_balanced(110)
        traces = generate_traces(specs, deduplicate=True)[:100]

        rng = random.Random(2026 + 7)
        ids = [t.id for t in traces]
        rng.shuffle(ids)
        n = len(ids)
        train = set(ids[:int(n*0.6)])
        val = set(ids[int(n*0.6):int(n*0.8)])
        test = set(ids[int(n*0.8):])

        assert len(train & val) == 0, "Train/val overlap"
        assert len(train & test) == 0, "Train/test overlap"
        assert len(val & test) == 0, "Val/test overlap"
        assert len(train) + len(val) + len(test) == n


class TestVerifierCoverage:

    def test_detection_rate_on_rejected(self):
        random.seed(42)
        from wmw.generators.scenarios import generate_balanced
        from wmw.generators.trace_generator import generate_traces
        from wmw.generators.perturbation import generate_preference_pairs
        from wmw.verifiers.pipeline import verify_trace

        specs = generate_balanced(30)
        traces = generate_traces(specs)
        pairs = generate_preference_pairs(traces, pairs_per_trace=4)

        detected = 0
        for p in pairs:
            vr = verify_trace(p.rejected)
            if not vr.all_ok:
                detected += 1

        rate = detected / len(pairs)
        assert rate > 0.10, f"Detection rate too low: {rate:.1%}"

    def test_gold_traces_mostly_pass(self):
        random.seed(42)
        from wmw.generators.scenarios import generate_balanced
        from wmw.generators.trace_generator import generate_traces
        from wmw.verifiers.pipeline import verify_traces

        specs = generate_balanced(50)
        traces = generate_traces(specs)
        results = verify_traces(traces)
        pass_rate = sum(1 for r in results if r.all_ok) / len(results)
        assert pass_rate > 0.85, f"Gold pass rate too low: {pass_rate:.1%}"


class TestLLMJudge:

    def test_parse_valid_json(self):
        from wmw.verifiers.llm_judge import parse_judge_response
        raw = json.dumps({
            "labels": ["transition"],
            "field_errors": {"state_0": None, "transition": "wrong law", "state_1": None,
                             "derivation": None, "answer": None},
            "answer_trace_consistent": True,
            "confidence": "high",
            "reasoning": "Wrong law applied.",
        })
        result = parse_judge_response(raw)
        assert result["labels"] == ["transition"]
        assert result["confidence"] == "high"

    def test_parse_malformed_falls_back(self):
        from wmw.verifiers.llm_judge import parse_judge_response
        result = parse_judge_response("this is not json at all")
        assert "parse_error" in result

    def test_judge_to_verifier_result(self):
        from wmw.verifiers.llm_judge import judge_to_verifier_result
        judge_output = {
            "labels": ["force", "transition"],
            "field_errors": {"state_0": "wrong force", "transition": "invalid",
                             "state_1": None, "derivation": None, "answer": None},
            "answer_trace_consistent": True,
            "confidence": "high",
            "reasoning": "test",
        }
        vr = judge_to_verifier_result(judge_output)
        assert vr.state_ok is False
        assert vr.transition_ok is False
        assert "force" in vr.labels
        assert "transition" in vr.labels


class TestEnsemble:

    def test_union_of_labels(self):
        from wmw.schemas.models import VerifierResult
        from wmw.verifiers.ensemble import merge_verifier_results

        rules = VerifierResult(schema_ok=True, state_ok=False, labels=["force"])
        judge = VerifierResult(schema_ok=True, transition_ok=False, labels=["transition"])
        ens = merge_verifier_results(rules, judge)

        assert set(ens.merged.labels) == {"force", "transition"}
        assert ens.rules_only_labels == ["force"]
        assert ens.judge_only_labels == ["transition"]
        assert ens.both_labels == []

    def test_both_flag_same_label(self):
        from wmw.schemas.models import VerifierResult
        from wmw.verifiers.ensemble import merge_verifier_results

        rules = VerifierResult(schema_ok=True, labels=["unit_scale"])
        judge = VerifierResult(schema_ok=True, labels=["unit_scale"])
        ens = merge_verifier_results(rules, judge)

        assert ens.both_labels == ["unit_scale"]
        assert ens.rules_only_labels == []
        assert ens.judge_only_labels == []

    def test_ensemble_stats(self):
        from wmw.schemas.models import VerifierResult
        from wmw.verifiers.ensemble import merge_verifier_results, compute_ensemble_stats

        results = []
        for _ in range(5):
            r = VerifierResult(schema_ok=True, labels=["force"])
            j = VerifierResult(schema_ok=True, labels=["transition"])
            results.append(merge_verifier_results(r, j))

        stats = compute_ensemble_stats(results)
        assert stats.rules_only_count == 5
        assert stats.judge_only_count == 5
        assert stats.both_count == 0
        assert stats.ensemble_detection_gain == 1.0


class TestReproducibility:

    def test_deterministic_traces(self):
        from wmw.generators.scenarios import generate_balanced
        from wmw.generators.trace_generator import generate_traces

        random.seed(2026)
        t1 = generate_traces(generate_balanced(20))
        random.seed(2026)
        t2 = generate_traces(generate_balanced(20))

        assert len(t1) == len(t2)
        for a, b in zip(t1, t2):
            assert a.id == b.id
            assert a.answer.value == b.answer.value

    def test_deterministic_pairs(self):
        from wmw.generators.scenarios import generate_balanced
        from wmw.generators.trace_generator import generate_traces
        from wmw.generators.perturbation import generate_preference_pairs

        random.seed(2026)
        traces = generate_traces(generate_balanced(20))
        pairs1 = generate_preference_pairs(traces, pairs_per_trace=4)

        random.seed(2026)
        traces = generate_traces(generate_balanced(20))
        pairs2 = generate_preference_pairs(traces, pairs_per_trace=4)

        assert len(pairs1) == len(pairs2)
        for a, b in zip(pairs1, pairs2):
            assert a.perturbation_type == b.perturbation_type
