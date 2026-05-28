import json
import random
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


class TestCommonDataset:
    def test_eval_example_roundtrip(self):
        from wmw.datasets.common import EvalExample, save_examples, load_examples
        ex = EvalExample(
            id="test_001", source="test", question="What is 2+2?",
            gold_answer="4", options=["3", "4", "5"],
        )
        path = Path("/tmp/test_examples.jsonl")
        save_examples([ex], path)
        loaded = load_examples(path)
        assert len(loaded) == 1
        assert loaded[0].id == "test_001"
        assert loaded[0].gold_answer == "4"
        path.unlink()

    def test_eval_example_to_dict(self):
        from wmw.datasets.common import EvalExample
        ex = EvalExample(id="x", source="s", question="q")
        d = ex.to_dict()
        assert d["id"] == "x"
        assert d["source"] == "s"
        assert d["image_path"] is None


class TestSyntheticDataset:
    def test_prepare_synthetic(self):
        from wmw.datasets.prepare import prepare_synthetic
        out = Path("/tmp/test_synth")
        out.mkdir(exist_ok=True)
        exs = prepare_synthetic(out, n_scenarios=10, seed=42)
        assert len(exs) == 10
        assert all(ex.source == "synthetic" for ex in exs)
        assert all(ex.extra.get("trace") is not None for ex in exs)

        for f in out.glob("*"):
            f.unlink()
        out.rmdir()


class TestClevrerPlaceholder:
    def test_placeholder_generation(self):
        from wmw.datasets.clevrer import _create_clevrer_placeholder
        path = Path("/tmp/test_clevrer.json")
        _create_clevrer_placeholder(path, 20)
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) > 0
        assert "questions" in data[0]
        path.unlink()


class TestPromptBuilder:
    def _make_example(self):
        from wmw.datasets.common import EvalExample
        return EvalExample(
            id="test_001", source="test",
            question="A 5kg block slides down a 30° incline. What is the acceleration?",
            options=["2.5 m/s²", "4.9 m/s²", "9.8 m/s²"],
            gold_answer="4.9 m/s²",
        )

    def test_answer_only(self):
        from wmw.evaluation.prompts import build_prompt
        ex = self._make_example()
        p = build_prompt(ex, "answer_only")
        assert "Question:" in p
        assert "final answer" in p.lower()
        assert "(A)" in p

    def test_full_trace(self):
        from wmw.evaluation.prompts import build_prompt
        ex = self._make_example()
        p = build_prompt(ex, "full_trace")
        assert "state_0" in p
        assert "transition" in p
        assert "JSON" in p

    def test_all_conditions(self):
        from wmw.evaluation.prompts import build_prompt, SYSTEM_PROMPTS
        ex = self._make_example()
        for cond in ["answer_only", "full_trace", "state_to_answer",
                     "ablation", "revise", "counterfactual",
                     "gold_state_answer", "gold_trans_answer"]:
            p = build_prompt(ex, cond)
            assert len(p) > 10, f"Empty prompt for {cond}"
            assert cond in SYSTEM_PROMPTS

    def test_unknown_condition_raises(self):
        from wmw.evaluation.prompts import build_prompt
        ex = self._make_example()
        with pytest.raises(ValueError):
            build_prompt(ex, "nonexistent_condition")


class TestTraceParser:
    def test_direct_json(self):
        from wmw.evaluation.trace_parser import parse_trace
        raw = json.dumps({"answer": {"value": "42"}})
        td, status = parse_trace(raw)
        assert td is not None
        assert status == "ok"
        assert td["answer"]["value"] == "42"

    def test_json_fence(self):
        from wmw.evaluation.trace_parser import parse_trace
        raw = "Here is the trace:\n```json\n{\"answer\": {\"value\": \"B\"}}\n```"
        td, status = parse_trace(raw)
        assert td is not None
        assert status == "json_fence"

    def test_embedded_json(self):
        from wmw.evaluation.trace_parser import parse_trace
        raw = "Let me analyze this. {\"answer\": {\"value\": 3.14}} That's the result."
        td, status = parse_trace(raw)
        assert td is not None
        assert status == "extracted"

    def test_answer_only_fallback(self):
        from wmw.evaluation.trace_parser import parse_trace
        td, status = parse_trace("The answer is B")
        assert td is not None
        assert status == "answer_only"
        assert td["answer"]["value"] == "B"

    def test_bare_letter(self):
        from wmw.evaluation.trace_parser import parse_trace
        td, status = parse_trace("A")
        assert td is not None
        assert td["answer"]["value"] == "A"

    def test_empty_fails(self):
        from wmw.evaluation.trace_parser import parse_trace
        td, status = parse_trace("")
        assert td is None
        assert status == "failed"

    def test_answer_matching(self):
        from wmw.evaluation.trace_parser import answers_match
        assert answers_match("4.9", "4.9")
        assert answers_match("4.90", 4.9)
        assert answers_match("B", "b")
        assert answers_match(4.9, 4.85, tolerance=0.05)
        assert not answers_match("A", "B")
        assert not answers_match(None, "B")

    def test_normalize_answer(self):
        from wmw.evaluation.trace_parser import normalize_answer
        assert normalize_answer("(B)") == "b"
        assert normalize_answer("Option A") == "a"
        assert normalize_answer("3.1400") == "3.14"
        assert normalize_answer(None) == ""


class TestVLMCaller:
    def test_mock_caller(self):
        from wmw.evaluation.vlm_caller import MODELS, call_vlm
        config = MODELS["mock"]
        resp = call_vlm(config, "system", "test prompt")
        assert resp.raw_text
        assert resp.error is None
        assert resp.latency_ms > 0

        d = json.loads(resp.raw_text)
        assert "state_0" in d
        assert "answer" in d

    def test_model_configs_exist(self):
        from wmw.evaluation.vlm_caller import MODELS
        assert "mock" in MODELS
        assert "gpt4o" in MODELS
        assert "claude_sonnet" in MODELS

    def test_missing_api_key(self):
        from wmw.evaluation.vlm_caller import MODELS, call_vlm, ModelConfig
        config = ModelConfig(
            name="test", provider="openai", model_id="test",
            api_key_env="NONEXISTENT_KEY_12345",
        )
        resp = call_vlm(config, "system", "test")
        assert resp.error is not None
        assert "Missing API key" in resp.error


class TestReranker:
    def test_score_trace(self):
        from wmw.evaluation.reranker import _score_trace

        assert _score_trace(None) == -10.0

        good_trace = {
            "id": "t1", "scenario_family": "free_fall", "question": "q",
            "state_0": {
                "objects": [{"name": "ball", "attributes": {"mass": 1}}],
                "relations": [], "forces": [], "variables": {}, "assumptions": [],
            },
            "transition": {"rule": "gravity", "effect": "falls", "equation": None, "evidence": []},
            "state_1": {"predicted_change": "falls", "new_variables": {}},
            "answer": {"value": "9.8", "unit": "m/s²", "explanation": "g"},
            "metadata": {"difficulty": "easy", "task_type": "calc", "source": "test"},
        }
        score = _score_trace(good_trace)
        assert score > 0


class TestStressTests:
    def test_held_out_eval(self):
        from wmw.evaluation.stress_tests import run_held_out_eval
        from wmw.generators.scenarios import generate_balanced
        from wmw.generators.trace_generator import generate_traces
        random.seed(42)
        specs = generate_balanced(10)
        traces = generate_traces(specs)
        result = run_held_out_eval(traces, n_pairs=20)
        assert result.test_name == "held_out_perturbations"
        assert result.n_examples > 0
        assert 0 <= result.seen_detection_rate <= 1
        assert 0 <= result.held_out_detection_rate <= 1

    def test_natural_rejected(self):
        from wmw.evaluation.stress_tests import run_natural_rejected_eval
        from wmw.datasets.common import EvalExample
        examples = [
            EvalExample(id=f"ex_{i}", source="test", question=f"q{i}",
                       gold_answer=str(i))
            for i in range(5)
        ]

        traces = [
            {"id": f"ex_{i}", "scenario_family": "test", "question": f"q{i}",
             "state_0": {"objects": [{"name": "a", "attributes": {}}],
                        "relations": [], "forces": [], "variables": {}, "assumptions": []},
             "transition": {"rule": "r", "effect": "e", "equation": None, "evidence": []},
             "state_1": {"predicted_change": "c", "new_variables": {}},
             "answer": {"value": str(i), "unit": None, "explanation": ""},
             "metadata": {"source": "test"}}
            for i in range(5)
        ]
        golds = [str(i) for i in range(5)]
        result = run_natural_rejected_eval(examples, traces, golds)
        assert result.test_name == "natural_rejected"
        assert result.n_examples > 0

    def test_stress_result_serialization(self):
        from wmw.evaluation.stress_tests import StressTestResult
        r = StressTestResult(test_name="test", n_examples=10, direct_accuracy=0.8)
        d = r.to_dict()
        assert d["test_name"] == "test"
        assert d["direct_accuracy"] == 0.8


class TestLatexTables:
    def test_table_main(self):
        from wmw.evaluation.latex_tables import table_main
        rows = [
            {"model": "GPT-4o", "answer_acc": 75.0, "state_acc": 60.0,
             "transition_acc": 55.0, "hidden_incons": 12.0,
             "revise_gain": 3.0, "rerank_gain": 5.0},
        ]
        tex = table_main(rows)
        assert "\\begin{table" in tex
        assert "GPT-4o" in tex
        assert "75.0" in tex

    def test_table_failures(self):
        from wmw.evaluation.latex_tables import table_failures
        results = {"GPT-4o": {"object": 5, "transition": 10}}
        totals = {"GPT-4o": 100}
        tex = table_failures(results, totals)
        assert "object" in tex
        assert "transition" in tex

    def test_table_stress(self):
        from wmw.evaluation.latex_tables import table_stress
        rows = [{"test_name": "ablation", "direct_accuracy": 70.0,
                 "trace_accuracy": 75.0, "consistency_rate": 5.0,
                 "answer_change_rate": 20.0,
                 "invalid_trace_accuracy": 60.0,
                 "valid_trace_accuracy": 80.0}]
        tex = table_stress(rows)
        assert "ablation" in tex

    def test_write_tables(self):
        from wmw.evaluation.latex_tables import write_all_tables
        tables = {"test": "\\begin{table}\ntest\n\\end{table}"}
        path = "/tmp/test_tables.tex"
        write_all_tables(tables, path)
        with open(path) as f:
            content = f.read()
        assert "test" in content
        os.unlink(path)


class TestIntegration:
    def test_mock_end_to_end(self):
        from wmw.datasets.prepare import prepare_synthetic
        from wmw.evaluation.prompts import build_prompt, SYSTEM_PROMPTS
        from wmw.evaluation.vlm_caller import MODELS, call_vlm
        from wmw.evaluation.trace_parser import parse_trace, extract_answer
        from wmw.verifiers.pipeline import verify_trace


        out = Path("/tmp/test_e2e")
        out.mkdir(exist_ok=True)
        examples = prepare_synthetic(out, n_scenarios=5, seed=42)


        config = MODELS["mock"]
        results = []
        for ex in examples:
            prompt = build_prompt(ex, "full_trace")
            resp = call_vlm(config, SYSTEM_PROMPTS["full_trace"], prompt)
            td, status = parse_trace(resp.raw_text)
            assert td is not None, f"Parse failed for {ex.id}"
            results.append(td)


        for td in results:
            td.setdefault("id", "test")
            td.setdefault("scenario_family", "free_fall")
            td.setdefault("question", "test")
            td.setdefault("metadata", {"difficulty": "easy", "task_type": "transition_prediction", "source": "test"})
            vr = verify_trace(td)

            assert vr.schema_ok


        for f in out.rglob("*"):
            if f.is_file():
                f.unlink()
        for f in sorted(out.rglob("*"), reverse=True):
            if f.is_dir():
                f.rmdir()
        out.rmdir()
