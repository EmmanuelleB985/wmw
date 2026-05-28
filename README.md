# WMW — World Models in Words

**Auditing Physical State-Transition Commitments in Vision–Language Models**

Trace-level evaluation of VLM physical reasoning. Instead of scoring only the final answer `(I, q) → a`, WMW asks a model to emit a *typed trace*

```
(I, q)  →  (s₀, Δs, s₁, r, a)
```

where `s₀` is the initial physical state, `Δs` the predicted transition (physical rule + effect), `s₁` the resulting state, `r` an optional derivation, and `a` the final answer. A **hybrid verifier** then independently checks schema validity, state grounding, transition consistency, and answer–trace faithfulness, producing typed failure labels.

This exposes a failure mode that answer-only evaluation cannot see: a model can select the correct option while asserting a physically impossible world. Across seven VLMs, **18–42% of correct answers are backed by physically invalid traces.**

> This repository accompanies *World Models in Words: Auditing Physical State-Transition Commitments in Vision-Language Models*. It contains the WMW-TRACEBANK resource, the hybrid verifier, the evaluation/intervention pipeline, and everything needed to reproduce the paper's results.

---

## What's in the box

- **WMW-TRACEBANK** — 200 schema- and recomputation-validated synthetic traces across **17 physics families**, plus **3,200** minimally-perturbed contrastive preference pairs (one typed physical violation each).
- **A hybrid verifier** — schema → state → transition checks, an optional LLM judge, and an ensemble, returning per-field `valid / invalid / abstain` verdicts and typed labels.
- **An evaluation pipeline** — six prompt conditions, verifier-guided reranking, one-shot verifier-feedback revision, and four faithfulness stress tests.
- **A DPO intervention** — LoRA + trace-level preference tuning on Qwen2.5-VL-7B.
- **An external-transfer pool** — 194 ScienceQA / CLEVRER / MathVista items converted to the trace schema.
- **A human-audit protocol** — stratified 400-trace sample with annotation guidelines.

---

## Install

```bash
pip install -r requirements.txt
```

Core dependencies: `dataclasses-json`, `jsonschema`, `matplotlib`, `pytest`. External dataset download additionally needs `datasets` and `Pillow`. GPU-side reproduction (open VLMs + DPO) needs `vllm >= 0.6.3`, `transformers`, `peft`, and `trl >= 0.12` — see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

No API keys are required to generate the dataset, run the verifier, or run the mock-model smoke test. Keys are only needed to call real closed models:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Quickstart

```bash
# 1. Run the test suite
python -m pytest tests/ -q

# 2. Generate WMW-TRACEBANK (deterministic, seed 2026 — no keys needed)
python scripts/generate_tracebank.py

# 3. Smoke-test the full pipeline with the mock model (no keys needed)
python scripts/run_evaluation.py --model mock --max-examples 10 --synthetic-only

# 4. Real evaluation (requires keys)
python scripts/run_evaluation.py --models gpt4o,claude_opus --max-examples 100
```

Step 2 writes the seed artifacts into `data/`; step 3 writes run outputs into `data/results/`.

---

## The trace formalism

Given an image (or frame sequence) `I` and a question `q`, a model produces a JSON trace `τ = (s₀, Δs, s₁, r, a)`:

| Field | Symbol | Contents |
|-------|--------|----------|
| Initial state | `s₀` | Only the physical info needed for `q`: entities, attributes, spatial relations, labels, units, velocities, forces, visible evidence, assumptions. Not a full scene graph. |
| Transition | `Δs` | The physical **rule** and predicted **effect** — qualitative ("the block accelerates down the incline") or symbolic ("`a = g·sinθ` down the slope"). Each change must be backed by a visible fact or an explicit assumption. |
| Resulting state | `s₁` | The consequence of the transition. |
| Derivation | `r` | Optional compact derivation for numeric/symbolic questions. |
| Answer | `a` | Normalized **separately** from the trace, so the trace can be checked for *implying a different answer than it states* (faithfulness). |

Each field is parsed, normalized, and verified independently. An explicit `assumptions` field and verifier **abstention** handle under-specified scenes (friction, elasticity, frame rate, sign convention). Abstention is reported, never silently converted into a negative label. The exact JSON schema lives in `wmw/schemas/trace_schema.json`; prompt templates are in `wmw/evaluation/prompts.py`.

---

## The hybrid verifier

The verifier is deliberately **inspectable** — not an unconstrained LLM judge by default. For each trace it returns

```
V(τ) = (z_schema, z_state, z_trans, z_ans, z_faith, ℓ)
```

where each `z ∈ {valid, invalid, abstain}` and `ℓ` is an optional failure label. It runs three ordered stages (a failed schema check short-circuits the rest):

1. **Schema** (`verifiers/schema_verifier.py`) — required keys, JSON types, allowed vocabularies (17 scenario families, 14 spatial-relation types).
2. **State** (`verifiers/state_verifier.py`) — five checks: entity existence, contradictory relations (6 pairs, e.g. `above/below`), variable bounds (20 variable patterns), force-direction sanity (gravity points down; normals don't point into surfaces), and gold-state tolerance (1% relative).
3. **Transition** (`verifiers/transition_verifier.py`) — six checks: rule-family keyword plausibility, force↔acceleration direction agreement, transition↔result sign agreement, temporal-marker swaps, equation syntax, and numeric answer–trace consistency.

An optional **LLM judge** (`verifiers/llm_judge.py`, Claude Sonnet 4 by default) returns per-field labels, an answer–trace boolean, a confidence level, and a rationale; low-confidence responses become abstentions. The **ensemble** (`verifiers/ensemble.py`) merges rules + judge by union of labels and conservative field verdicts — high recall at some cost to precision. In the released closed-model data the ensemble detects **6.8× more errors** than rules alone (813 vs. 120 labels across 4 models on 194 examples).

---

## Failure taxonomy

Nine typed labels, used by both the verifier and the human audit (`docs/failure_taxonomy.md`):

| Label | A trace earns it when… |
|-------|------------------------|
| **Object** | the wrong entity is tracked, or two entities are merged. |
| **State** | the initial physical state is misstated. |
| **Relation** | contact / support / containment / alignment / order is wrong. |
| **Force** | a force is missing, reversed, or assigned to the wrong body. |
| **Transition** | the state is right but the predicted change is invalid. |
| **Intervention** | the effect of an action or counterfactual is wrong. |
| **Temporal** | before/after frames are swapped or sequence order is wrong. |
| **Unit/scale** | magnitude, unit, or sign convention is inconsistent. |
| **Faithfulness** | the final answer contradicts the stated trace. |

---

## WMW-TRACEBANK

A controlled resource supporting three uses: diagnostic evaluation, verifier/reranker development, and trace-level preference training. Each example bundles a rendered scene, a question, a gold answer, a validated trace, verifier metadata, and a task-family label.

**Coverage — 17 families:** inclined plane, projectile, collision, free fall, friction, circular motion, pulley, lever, pendulum, spring, fluids/buoyancy, circuits, optics, waves, thermal, and electromagnetic induction. Explicitly excluded: thermodynamic cycles, quantum mechanics, nuclear physics, relativity, advanced fluid dynamics.

**Quality gates** — before entering the benchmark split, each example must (1) satisfy the JSON schema, (2) pass family-specific recomputation that recovers the requested quantity from trace variables, (3) share canonical parameter keys across question/diagram/trace/answer, and (4) survive review by independent annotators. Examples failing any automated gate ship only as development diagnostics.

**Preference pairs** — each `(τ⁺, τ⁻)` differs by **one** typed perturbation (reverse a force, swap temporal order, change a contact relation, mis-assign an object, corrupt a unit, or make the answer contradict the trace). The perturbation engine (`generators/perturbation.py`) registers **25 functions across 9 label categories, split 15 seen / 10 held-out**, so learned rerankers can be tested on unseen error patterns and on natural model errors — separating *physical consistency* from *perturbation-template detection*.

**Release contents** (`data/`, and `docs/`):

| Component | Contents | Count |
|-----------|----------|-------|
| Synthetic traces | Diagrams, questions, gold answers, typed traces, metadata, assumptions | 200 |
| External-transfer pool | ScienceQA + CLEVRER + MathVista converted to trace schema | 194 |
| Preference pairs | Close chosen/rejected pairs, one typed violation each | 3,200 |
| Model outputs | Raw generations across answer-only / trace / revision / reranking | ~48,000 |
| Verifier labels | Schema / state / transition / answer / faithfulness / abstention | ~48,000 |
| Audit protocol | Stratified sample + annotation guide | 400 |
| Documentation | Schema, annotation guide, audit protocol, dataset card, prompts | `docs/` |

**Generation statistics** (`data/generation_stats.json`, seed 2026, v1.0.0): 200 scenarios; strict first-pass schema acceptance **181/200 (90.5%)** with the remainder repaired by deterministic canonicalization; 3,200 pairs (seen 2,340 / held-out 860); splits **120/40/40** traces and **1,920/640/640** pairs (all pairs from one trace stay in the same split). Post-canonicalization gold-trace diagnostics: transition consistency 100%, trace–answer consistency 100%, abstention 0%.

**Integrity** — deterministic given seed 2026. SHA-256: traces `b4d841…810d`, pairs `275ba2…d9cc1`.

---

## Evaluation pipeline

`scripts/run_evaluation.py` runs five stages (run all, or select with `--stages`):

| Stage | What it does |
|-------|--------------|
| `data` | Download / prepare datasets |
| `eval` | Call VLMs across the prompt conditions + reranking + revision |
| `verify` | Run rule verifier + LLM judge + ensemble on every trace |
| `stress` | Four faithfulness stress tests |
| `tables` | Emit LaTeX tables |

**Six prompt conditions** (`wmw/evaluation/prompts.py`):

1. `answer_only` — standard `q → a`.
2. `state_to_answer` — `q → s₀ → a`.
3. `full_trace` — emit `(s₀, Δs, s₁, r, a)` in schema.
4. `gold_state_answer` — given gold `s₀`, predict the answer (isolates the transition bottleneck).
5. `gold_trans_answer` — given gold `Δs`, predict the answer.
6. `revise` — full trace + concise verifier feedback → one revision.

**Reranking** (`wmw/evaluation/reranker.py`) — sample `k ∈ {4, 8, 16}` traces at temperature 0.7; score each with a transparent additive weight (+1 schema, +2 each for state / transition / answer–trace, +0.5 per abstention, −0.5 per failure label) and keep the best. Learned and majority-vote variants are included.

**Four stress tests** (`wmw/evaluation/stress_tests.py`) — trace ablation, counterfactual editing, held-out perturbation detection, and natural-rejected-trace discrimination.

**Metrics** (`wmw/metrics.py`):

- `AnswerAcc`, `StateAcc`, `TransAcc`
- **HIR_all** — unconditional hidden inconsistency: correct answer ∧ invalid trace.
- **HIR_correct** — *among correct answers*, the fraction backed by an invalid trace.
- **VSG** (visual-state gap) = `A(s₀* → a) − A(I → s₀ → a)` — cost of extracting state from vision.
- **TG** (transition gap) = `A(s₀*, Δs* → a) − A(s₀* → Δs → a)` — cost of predicting the change once state is given.

All metrics report 95% bootstrap CIs (B = 1,000, seed 2026); main-text CIs are ≤ ±2.5pp.

---

## Results

### Main diagnostics — 7 VLMs 

`Trace–ans.` is the answer–trace consistency rate. `Revise` / `Rerank` are trace-validity gains from one verifier-feedback revision and from selecting among `k=8` samples.

| Model | Ans. acc. | State acc. | Trans. acc. | Trace–ans. | **HIR_correct** | Revise | Rerank |
|-------|:---------:|:----------:|:-----------:|:----------:|:---------------:|:------:|:------:|
| Claude Opus 4.7 | **76%** | **81%** | **68%** | **91%** | **18%** | +3pp | +5pp |
| GPT-5.5 | 72% | 77% | 61% | 88% | 24% | +4pp | +6pp |
| GPT-4o | 63% | 68% | 49% | 83% | 31% | +4pp | +7pp |
| Qwen2.5-VL-72B | 60% | 65% | 47% | 82% | 33% | +3pp | +6pp |
| InternVL3-78B | 58% | 63% | 44% | 80% | 34% | +3pp | +5pp |
| GPT-4o-mini | 52% | 55% | 38% | 78% | 35% | +2pp | +4pp |
| Qwen2.5-VL-7B | 42% | 46% | 30% | 72% | 42% | +2pp | +3pp |

> **RQ1 — answer accuracy hides invalid worlds.** Across all seven models, **18–42% of correct answers rest on physically invalid traces.** Even the strongest model still has ~1 in 5 of its correct answers backed by a world that fails verification.

### Where models fail — visual-state vs. transition gaps (Figure 2)

| Model | VSG | TG |
|-------|:---:|:--:|
| Claude Opus 4.7 | −9pp | −13pp |
| GPT-5.5 | −7pp | +1pp |
| GPT-4o | +2pp | +19pp |
| GPT-4o-mini | +8pp | +20pp |
| Qwen2.5-VL-7B | +16pp | +30pp |

> **RQ2 — the *transition* step is the dominant bottleneck** for every model except the two strongest closed systems. Positive values mean gold replacement *helps*. For Opus 4.7 and GPT-5.5, gold replacement *hurts* — interpreted as a calibration effect: these models compensate for their own state extraction, and an externally-formatted gold state disrupts that. Oracle-state ablations should be read cautiously for strong models.

### Verifier validation against human audit (Table 4, n = 400)

Inter-annotator agreement **κ = 0.72**. Labels need **F1 ≥ 0.70** to enter primary quantitative claims.

| Label | Prec. | Rec. | F1 | Abstain |
|-------|:-----:|:----:|:--:|:-------:|
| State | .85 | .79 | **.82** | 4.8% |
| Transition | .81 | .75 | **.78** | 6.5% |
| Faithfulness | .93 | .89 | **.91** | 2.1% |
| Unit/scale | .72 | .64 | .68 ⚠ | 11.3% |
| All audited | .84 | .78 | .81 | 7.2% |

> **RQ3 — the verifier is reliable where used quantitatively.** State, transition, and faithfulness clear the 0.70 bar. **Unit/scale (F1 = 0.68) falls below threshold** — retained in the taxonomy but flagged out of primary metrics.

### Interventions on Qwen2.5-VL-7B 

| Condition | Ans. | State | Trans. | Trace–ans. | **HIR_correct** | Held-out pref. |
|-----------|:----:|:-----:|:------:|:----------:|:---------------:|:--------------:|
| Base trace | 42% | 46% | 30% | 72% | 42% | 58% |
| Verifier feedback | 44% | 51% | 35% | 76% | 37% | 62% |
| Rules reranker (k=8) | 43% | 52% | 36% | 78% | 34% | – |
| Learned reranker (k=8) | 44% | 54% | 39% | 80% | 30% | 74% |
| **DPO preference-tuned** | **44%** | **55%** | **41%** | **81%** | **25%** | **79%** |

> **RQ4 — interventions cut hidden inconsistency without hurting accuracy.** DPO drops HIR_correct from **42% → 25% (a 41% relative reduction)** while answer accuracy *improves* +2pp. Held-out perturbation accuracy reaches 79% (vs. 84% on seen families); natural-error accuracy is 71% — confirming the model learns physical consistency, not just perturbation templates, with a remaining gap.

### Failure decomposition (1,740 invalid traces, all 7 models)

`Transition 28% · State 22% · Relation 15% · Force 12% · Faithfulness 8% · Temporal 7% · Unit/scale 4% · Intervention 3% · Object 1%.`
State/relation-heavy models need better grounding; transition-heavy models need better dynamics; faithfulness-heavy models need tighter answer realization.

### External transfer (averaged over all 7 models)

| Split | Ans. | Trace | Rank ρ (trace validity) |
|-------|:----:|:-----:|:-----------------------:|
| Controlled | 58% | 51% | – |
| External physics | 53% | 44% | 0.89 |
| External science | 50% | 43% | 0.82 |

Model rankings by trace validity are largely preserved across splits (Spearman ρ = 0.89 / 0.82), while absolute trace validity drops 5–8pp on naturalistic images. Notably, on CLEVRER answer rankings shuffle (ρ = 0.20) but trace-validity rankings hold (ρ = 0.63) — trace-level diagnostics are more robust to modality shift than final-answer accuracy.

---

## Reproducing the results

```bash
python scripts/generate_tracebank.py                              # data/*_seed.jsonl, splits, stats
python scripts/run_evaluation.py --model mock --synthetic-only    # smoke test
python scripts/run_ensemble.py --traces data/trace_examples_seed.jsonl   # rules vs judge vs ensemble
python scripts/run_audit.py                                       # stratified 400-trace sample
```

Closed-model evaluation (needs keys):

```bash
python scripts/run_evaluation.py \
  --models claude_opus,gpt55,gpt4o,gpt4o_mini \
  --judge-model claude_sonnet --max-examples 194
```

Open-model + DPO (GPU; see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)). The whole portable pipeline is also driven by one script:

```bash
bash scripts/reproduce/run_all.sh
# or stage-by-stage:
bash scripts/reproduce/run_all.sh setup prepare_data
bash scripts/reproduce/run_all.sh serve_open eval_open
bash scripts/reproduce/run_all.sh train_dpo serve_dpo eval_dpo
bash scripts/reproduce/run_all.sh tables
```

Closed models use a 2,048-token budget, temperature 0.0 (greedy) / 0.7 (reranking), 30 req/min. Open models run on a 4×A100 80 GB node via vLLM ≥ 0.6.3, bf16, `max-model-len 8192`. The Qwen2.5-VL-7B DPO run takes ~2–3 h (LoRA r=16, α=32, β=0.1, lr 5e-6, 2 epochs, effective batch 32).

---

## Supported models

| Key | Model | Provider / env |
|-----|-------|----------------|
| `claude_opus` | Claude Opus 4.7 | Anthropic (`ANTHROPIC_API_KEY`) |
| `claude_sonnet` | Claude Sonnet 4 (default judge) | Anthropic |
| `claude_haiku` | Claude Haiku 4.5 | Anthropic |
| `gpt55` | GPT-5.5 | OpenAI (`OPENAI_API_KEY`) |
| `gpt4o` | GPT-4o | OpenAI |
| `gpt4o_mini` | GPT-4o-mini | OpenAI |
| `qwen_vl` | Qwen2.5-VL (Plus / local) | local vLLM |
| `llava_local` | LLaVA-1.6 | local vLLM (`http://localhost:8000/v1`) |
| `mock` | deterministic mock | none (testing) |

Open-weights checkpoints used in the paper: `Qwen/Qwen2.5-VL-7B-Instruct`, `Qwen/Qwen2.5-VL-72B-Instruct`, `OpenGVLab/InternVL3-78B` (needs `--trust-remote-code`).

---

## Project structure

```
wmw/
├── schemas/
│   ├── trace_schema.json        # JSON Schema (state, transition, derivation, answer)
│   └── models.py                # Trace / State / Transition / Answer dataclasses; FAILURE_LABELS
├── generators/
│   ├── scenarios.py             # 17 physics families
│   ├── trace_generator.py       # ScenarioSpec → Trace (content-addressed IDs, dedup)
│   ├── perturbation.py          # 25 perturbations, applicability guards, no-op filtering
│   └── paraphrases.py           # question paraphrase variants
├── verifiers/
│   ├── schema_verifier.py       # structure / vocab validation
│   ├── state_verifier.py        # entity, bounds, force, relation checks
│   ├── transition_verifier.py   # rule plausibility, temporal, answer–trace consistency
│   ├── llm_judge.py             # LLM-as-judge semantic verifier
│   ├── ensemble.py              # rules + judge merge, disagreement matrix
│   └── pipeline.py              # orchestrates the rule verifiers
├── datasets/                    # ScienceQA / CLEVRER / MathVista loaders + prepare.py
├── evaluation/
│   ├── prompts.py               # 6 prompt conditions
│   ├── vlm_caller.py            # OpenAI / Anthropic / local / mock callers
│   ├── trace_parser.py          # JSON extraction from raw VLM output
│   ├── reranker.py              # verifier-scored k-sample reranking
│   ├── stress_tests.py          # 4 faithfulness stress tests
│   ├── open_vlm.py              # vLLM open-model client
│   └── run_evaluation.py        # in-package pipeline entry
├── training/dpo_data.py         # build TRL-format DPO pairs from preference data
├── diagrams/renderer.py         # matplotlib scene renderer
└── metrics.py                   # VSG, TG, HIR, agreement, rerank/ensemble gains

scripts/
├── generate_tracebank.py        # traces + pairs (deterministic, seed 2026)
├── download_datasets.py         # external dataset download → data/eval/
├── run_evaluation.py            # master 5-stage pipeline
├── run_ensemble.py              # 3-way verifier comparison
├── run_audit.py                 # stratified audit sampling
├── bundle_model_outputs.py      # package results into the HF release
├── enrich_eval_data.py          # gold-metadata enrichment
├── generate_eval_diagrams.py    # render diagrams for eval examples
├── reproduce/                   # GPU reproducibility (run_all.sh, open VLMs, DPO, sweeps)
└── analysis/                    # build_final_tables.py, verifier_agreement.py, external_transfer.py

docs/                            # failure_taxonomy, annotation_guidelines, verifier_audit_protocol,
                                 # llm_judge_prompt, coverage_map, case_studies, dataset_card
tests/                           # test_tracebank.py, test_evaluation.py, test_quality.py
```

---

## Tests

```bash
python -m pytest tests/ -q
```

Covers generators, all verifier stages, dataset loaders, prompt construction, the trace parser, stress tests, and metrics.

> **Known issue:** four `TestLatexTables` cases in `tests/test_evaluation.py` import `wmw.evaluation.latex_tables`, a module not present in this snapshot (LaTeX-table generation currently lives in `scripts/run_evaluation.py` and `scripts/analysis/build_final_tables.py`). Those four tests error on import; the remaining suite passes. Either point the import at the script-side table builders or restore the module to clear them.

---

## Citation

```bibtex
@inproceedings{wmw,
  title     = {World Models in Words: Auditing Physical State-Transition Commitments in Vision-Language Models},
  author    = {Emmanuelle Bourigault},
  year      = {2026}
}
```

--- 

## License

Author-generated traces, prompts, metadata, and verification labels are released for research use under the license in the artifact repository. External datasets (ScienceQA, CLEVRER, MathVista) retain their original licenses; converted examples are distributed only where those terms permit.

> **Non-use:** WMW-TRACEBANK is a diagnostic resource, **not** a safety certificate, a final-answer leaderboard, or a complete physics curriculum. A verified textual trace is not a guarantee of real-world safety.
