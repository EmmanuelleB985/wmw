# Dataset Card: WMW-TraceBank

## Overview

**WMW-TraceBank** is a seed resource for trace-level evaluation of
vision-language model physical reasoning. It provides the data interface
(schemas, traces, preference pairs, verifiers, and audit protocols) for
the *World Models in Words* framework.

## Contents

| File | Description |
|---|---|
| `trace_schema.json` | JSON Schema for traces: state, transition, result, derivation, answer, verifier, preference-pair fields |
| `trace_examples_seed.jsonl` | 32 positive traces across 17 physics families |
| `preference_pairs_seed.jsonl` | 256 close chosen/rejected pairs (8 per trace) spanning 9 failure labels |
| `splits.json` | Train/val/test split assignments (trace-level, no leakage) |
| `generation_stats.json` | Generation config, counts, SHA-256 hashes |
| `failure_taxonomy.md` | 9 error labels: object, state, relation, force, transition, intervention, temporal, unit_scale, faithfulness |
| `annotation_guidelines.md` | Human labeling instructions |
| `verifier_audit_protocol.md` | Stratified audit protocol with target sample sizes |
| `llm_judge_prompt.md` | JSON-only prompt for semantic trace verification |
| `coverage_map.md` | Mapping to introductory physics curriculum |
| `case_studies.md` | Templates for hidden-inconsistency and verifier-disagreement analysis |

## Intended Use

- Diagnostic evaluation of VLM physical reasoning
- Verifier development and calibration
- Preference-pair construction for DPO-style post-training
- Reranker training for trace selection at inference time
- Benchmarking rule-based vs LLM-judge vs ensemble verifiers

## Non-Use Cases

- **Not a safety certificate.** A verified trace is not evidence of real-world physical understanding.
- **Not a leaderboard.** 32 traces seed the interface; use 200+ for statistical power.
- **Not complete physics.** Coverage limited to introductory mechanics, circuits, optics, pressure.

## Limitations

- Synthetic data only. All traces generated from parameterized templates.
- 17 families covering ~60% of AP Physics C. No thermo cycles, quantum, relativity, nuclear.
- Rule verifier is shallow. Catches structure, unit, sign errors but not deep semantic physics.
- Perturbation artifacts. Rejected traces are single-field edits; real errors may be more diffuse.

## Data Splits

60/20/20 train/val/test at the trace level. All pairs from one trace go to the same split.

## License

Research use. External datasets retain original licenses.

## Version

v1.0.0 — seed release. Deterministic given seed 2026. SHA-256 hashes in generation_stats.json.
