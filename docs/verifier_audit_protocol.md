# WMW Verifier-Audit Protocol

## Purpose

The verifier is a measurement instrument, not an oracle. This protocol defines how to measure its reliability against human judgments.

## Audit Design

### Sampling Strategy

Stratified sampling across three dimensions:
1. **Scenario family** — proportional to family distribution in the dataset
2. **Model** — equal allocation across evaluated models
3. **Predicted label** — oversampling rare labels to ensure statistical power

Target: 200 traces minimum (50 positive, 150 from preference pairs spanning all 9 labels).

### Annotation Procedure

1. Two trained annotators independently label each trace (see `annotation_guidelines.md`)
2. Annotators see the trace but not the verifier's output
3. Disagreements are adjudicated by a third annotator
4. Adjudicated labels are treated as ground truth

### Metrics

For each verifier label (state, transition, faithfulness, etc.):

| Metric | Formula | Target |
|--------|---------|--------|
| Precision | TP / (TP + FP) | ≥ 0.80 |
| Recall | TP / (TP + FN) | ≥ 0.70 |
| F1 | 2·P·R / (P+R) | ≥ 0.75 |
| Abstention rate | abstained / total | Report only |
| Cohen's κ | (agreement - chance) / (1 - chance) | ≥ 0.60 |

### Decision Rules

- Labels with F1 ≥ 0.75: use as primary quantitative evidence
- Labels with 0.50 ≤ F1 < 0.75: report with caveats
- Labels with F1 < 0.50: report qualitatively only, do not use for model comparison

### False Positive Analysis

Run the verifier on all gold-standard positive traces. Any label fired on a positive trace is a false positive. Report per-label false positive rate; target < 5%.

### Detection Rate Analysis

Run the verifier on all rejected traces from preference pairs. For each injected perturbation label, measure whether the verifier correctly identifies that label. Report per-label detection rate; target > 60%.

## Running the Audit

```bash
# Generate the data
python scripts/generate_tracebank.py --num-scenarios 200 --pairs-per-trace 16

# Run the automated audit
python scripts/run_audit.py --data-dir data --sample-size 50
```

The audit script produces `data/audit_sample.jsonl` for human annotation.
