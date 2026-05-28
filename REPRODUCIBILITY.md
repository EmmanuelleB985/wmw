# Reproducibility Guide

This document describes how to reproduce all experiments in the paper on
a multi-GPU machine (4×A100 80GB recommended). Read it together with the
main `README.md`.

## Prerequisites

- Python ≥ 3.10
- CUDA-capable GPUs (4× A100 80GB for the full suite; smaller configs
  work for individual 7B models on a single GPU)
- API keys for closed-model evaluation:
  `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`

## Quick Start

```bash
pip install -r requirements.txt
bash scripts/reproduce/run_all.sh          # full pipeline
```

Or run individual stages:

```bash
bash scripts/reproduce/run_all.sh setup prep
bash scripts/reproduce/run_all.sh serve_qwen7 eval_qwen7 stop_vllm
bash scripts/reproduce/run_all.sh dpo_train serve_dpo dpo_eval
bash scripts/reproduce/run_all.sh tables
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `WMW_ROOT` | `$PWD` | Repository root |
| `DATA_DIR` | `$WMW_ROOT/data` | Dataset directory |
| `CKPT_DIR` | `$WMW_ROOT/checkpoints` | DPO adapter output |
| `VLLM_PORT` | `8000` | Port for vLLM server |
| `N_EVAL` | `200` | Examples per source |
| `N_RERANK` | `50` | Examples for reranking sweep |
| `JUDGE_MODEL` | `claude_sonnet` | LLM judge for verifier ensemble |

## What the Pipeline Runs

| Paper Element | Script |
|---|---|
| Open-suite VLM evaluation (§6.1) | `wmw/evaluation/open_vlm.py` + `scripts/reproduce/run_open_vlms.py` |
| Reranking sweep k ∈ {1, 4, 8, 16} — Figure 2 | `scripts/reproduce/run_rerank_sweep.py` |
| LoRA + DPO preference training (§6.3) | `wmw/training/dpo_data.py`, `scripts/reproduce/prepare_dpo_data.py`, `scripts/reproduce/train_dpo.py` |
| DPO-tuned model evaluation | `scripts/reproduce/eval_dpo_model.py` |
| Verifier vs. judge agreement — Table 5 | `scripts/analysis/verifier_agreement.py` |
| External transfer — Table 8 | `scripts/analysis/external_transfer.py` |
| Final paper tables 3–8 + Figure 2 | `scripts/analysis/build_final_tables.py` |
| End-to-end orchestrator | `scripts/reproduce/run_all.sh` |

## Per-Model Serving

vLLM ≥ 0.6.3 is required. Tensor-parallel sizes are tuned for 80 GB GPUs:

| Key | HF id | TP | Memory (bf16) |
|---|---|---|---|
| `qwen25_vl_7b` | `Qwen/Qwen2.5-VL-7B-Instruct` | 1 | ~17 GB |
| `qwen25_vl_32b` | `Qwen/Qwen2.5-VL-32B-Instruct` | 2 | ~70 GB / GPU |
| `qwen25_vl_72b` | `Qwen/Qwen2.5-VL-72B-Instruct` | 4 | ~38 GB / GPU |
| `internvl3_8b` | `OpenGVLab/InternVL3-8B` | 1 | ~18 GB |
| `internvl3_14b` | `OpenGVLab/InternVL3-14B` | 2 | ~32 GB / GPU |
| `internvl3_38b` | `OpenGVLab/InternVL3-38B` | 4 | ~22 GB / GPU |
| `llava_onevision_7b` | `lmms-lab/llava-onevision-qwen2-7b-ov` | 1 | ~17 GB |
| `molmo_7b` | `allenai/Molmo-7B-D-0924` | 1 | ~16 GB |

## DPO Training Notes

- Trainer: HuggingFace TRL `DPOTrainer` ≥ 0.12 with `peft` LoRA adapters.
- Default hyperparameters (paper §6.3): r=16, α=32, dropout 0.05; β=0.1,
  lr 5×10⁻⁶, 2 epochs, batch 1 × accumulate 8.
- `prepare_dpo_data.py` enforces zero held-out perturbation family leakage
  into the train set.
- On 4×A100 80GB with bf16 + gradient checkpointing, Qwen2.5-VL-7B fits
  comfortably; expect ~2–3 h for the 1413-pair train set × 2 epochs.

## Output Layout

After a full run, `data/results/` contains:

```
{model}_full_trace.json              # accuracy, parse_rate, latency
{model}_answer_only.json             # baseline (no trace)
{model}_state_to_answer.json         # state → answer
{model}_gold_state_answer.json       # s₀* → a (for VSG)
{model}_gold_trans_answer.json       # Δs* → a (for TG)
{model}_verification.json            # rule + judge + ensemble labels
{model}_rerank_sweep.json            # rerank curve at k ∈ {1,4,8,16}
{model}_stress.json                  # 4 stress tests
verifier_agreement.json              # per-label rules vs judge P/R/F1
external_transfer.json               # per-source decomposition
model_summary.json                   # all numbers for every model
paper_tables_all.tex                 # All LaTeX tables concatenated
```
