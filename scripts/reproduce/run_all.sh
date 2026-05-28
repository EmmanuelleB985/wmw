#!/usr/bin/env bash

set -euo pipefail

WMW_ROOT="${WMW_ROOT:-$PWD}"
DATA_DIR="${DATA_DIR:-$WMW_ROOT/data}"
CKPT_DIR="${CKPT_DIR:-$WMW_ROOT/checkpoints}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_URL="${OPEN_VLM_URL:-http://localhost:$VLLM_PORT/v1}"
N_EVAL="${N_EVAL:-200}"
N_RERANK="${N_RERANK:-50}"
JUDGE_MODEL="${JUDGE_MODEL:-claude_sonnet}"

mkdir -p "$DATA_DIR/results" "$CKPT_DIR" "$WMW_ROOT/logs"

cd "$WMW_ROOT"
export PYTHONPATH="$WMW_ROOT:${PYTHONPATH:-}"
export OPEN_VLM_URL="$VLLM_URL"
export OPEN_VLM_KEY="EMPTY"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$WMW_ROOT/logs/reproduce_master.log"; }

stage_setup() {
    log "── setup: installing dependencies"
    pip install --quiet \
        vllm==0.6.3 \
        transformers>=4.46.3 \
        peft==0.13.0 \
        trl==0.12.0 \
        accelerate==1.0.0 \
        datasets==3.0.0 \
        pillow matplotlib jsonschema dataclasses-json
    log "── setup: done"
}

stage_prep() {
    log "── prep: verifying TraceBank artefacts on disk"
    test -f "$DATA_DIR/trace_examples_seed.jsonl" \
        || python scripts/generate_tracebank.py --num-scenarios 200 --pairs-per-trace 16
    test -f "$DATA_DIR/eval_data/merged_eval.jsonl" \
        || python scripts/download_datasets.py --skip-external
    log "── prep: building DPO splits"
    python scripts/reproduce/prepare_dpo_data.py \
        --pairs "$DATA_DIR/preference_pairs_seed.jsonl" \
        --splits "$DATA_DIR/splits.json" \
        --output-dir "$DATA_DIR/dpo" \
        --image-root "$DATA_DIR/eval_data/diagrams"
    log "── prep: done"
}

_serve_vllm() {
    local hf_id="$1"
    local tp="${2:-1}"
    local extra="${3:-}"
    log "── vllm: starting $hf_id (tp=$tp)"
    pkill -f "bin/vllm serve" 2>/dev/null || true
    sleep 3
    nohup vllm serve "$hf_id" \
        --port "$VLLM_PORT" \
        --tensor-parallel-size "$tp" \
        --max-model-len 8192 --enforce-eager --enforce-eager \
       \
        $extra \
        > "$WMW_ROOT/logs/vllm_$(date +%s).log" 2>&1 &
    echo $! > "$WMW_ROOT/logs/vllm.pid"
    log "── vllm: waiting for $VLLM_URL to come up"
    python -c "
from wmw.evaluation.open_vlm import wait_for_server, check_server_model
ok = wait_for_server('$VLLM_URL', timeout_s=900)
print('READY' if ok else 'TIMEOUT')
print('Served:', check_server_model('$VLLM_URL'))
"
}

stop_vllm() {
    log "── vllm: stopping"
    if [[ -f "$WMW_ROOT/logs/vllm.pid" ]]; then
        kill "$(cat "$WMW_ROOT/logs/vllm.pid")" 2>/dev/null || true
    fi
    pkill -f "bin/vllm serve" 2>/dev/null || true
    sleep 3
}

_eval_open_vlm() {
    local model_key="$1"
    log "── eval: $model_key full pipeline"
    python scripts/reproduce/run_open_vlms.py \
        --model "$model_key" \
        --conditions all \
        --max-examples "$N_EVAL" \
        --data "$DATA_DIR/eval_data/merged_eval.jsonl"
}

_rerank_open_vlm() {
    local model_key="$1"
    log "── rerank sweep: $model_key (k=1,4,8,16) on $N_RERANK examples"
    python scripts/reproduce/run_rerank_sweep.py \
        --model "$model_key" \
        --ks 1,4,8,16 \
        --max-examples "$N_RERANK" \
        --data "$DATA_DIR/eval_data/merged_eval.jsonl"
}

_verify_open_vlm() {
    local model_key="$1"
    log "── verify: $model_key"
    python scripts/run_evaluation.py \
        --model "$model_key" \
        --stages verify \
        --judge-model "$JUDGE_MODEL" \
        --output-dir "$DATA_DIR"
}

stage_serve_qwen7()   { _serve_vllm "Qwen/Qwen2.5-VL-7B-Instruct" 1; }
stage_eval_qwen7()    { _eval_open_vlm qwen25_vl_7b; _rerank_open_vlm qwen25_vl_7b; _verify_open_vlm qwen25_vl_7b; }

stage_serve_internvl(){ _serve_vllm "OpenGVLab/InternVL3-8B" 1 "--trust-remote-code"; }
stage_eval_internvl() { _eval_open_vlm internvl3_8b;    _rerank_open_vlm internvl3_8b;    _verify_open_vlm internvl3_8b; }

stage_serve_llava()   { _serve_vllm "lmms-lab/llava-onevision-qwen2-7b-ov" 1; }
stage_eval_llava()    { _eval_open_vlm llava_onevision_7b; _rerank_open_vlm llava_onevision_7b; _verify_open_vlm llava_onevision_7b; }

stage_serve_molmo()   { _serve_vllm "allenai/Molmo-7B-D-0924" 1 "--trust-remote-code"; }
stage_eval_molmo()    { _eval_open_vlm molmo_7b;     _rerank_open_vlm molmo_7b;     _verify_open_vlm molmo_7b; }

stage_serve_qwen32()  { _serve_vllm "Qwen/Qwen2.5-VL-32B-Instruct" 2; }
stage_eval_qwen32()   { _eval_open_vlm qwen25_vl_32b; _rerank_open_vlm qwen25_vl_32b; _verify_open_vlm qwen25_vl_32b; }

stage_serve_qwen72()  { _serve_vllm "Qwen/Qwen2.5-VL-72B-Instruct" 4; }
stage_eval_qwen72()   { _eval_open_vlm qwen25_vl_72b; _verify_open_vlm qwen25_vl_72b; }

stage_dpo_train() {
    stop_vllm
    log "── DPO: training adapter on Qwen2.5-VL-7B"
    mkdir -p "$CKPT_DIR/qwen25vl_7b_wmw_dpo"
    accelerate launch \
        --num_processes 4 \
        --mixed_precision bf16 \
        scripts/reproduce/train_dpo.py \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --train "$DATA_DIR/dpo/train.jsonl" \
        --val "$DATA_DIR/dpo/val.jsonl" \
        --output-dir "$CKPT_DIR/qwen25vl_7b_wmw_dpo" \
        --epochs 2 \
        --per-device-bs 1 \
        --grad-accum 8 \
        --beta 0.1 \
        --lr 5e-6 \
        --bf16
}

stage_serve_dpo() {
    log "── vllm: serving Qwen2.5-VL-7B + DPO adapter"
    pkill -f "bin/vllm serve" 2>/dev/null || true
    sleep 3
    nohup vllm serve "Qwen/Qwen2.5-VL-7B-Instruct" \
        --port "$VLLM_PORT" \
        --tensor-parallel-size 1 \
        --max-model-len 8192 --enforce-eager --enforce-eager \
        --enable-lora \
        --lora-modules "wmw_dpo=$CKPT_DIR/qwen25vl_7b_wmw_dpo" \
        > "$WMW_ROOT/logs/vllm_dpo.log" 2>&1 &
    echo $! > "$WMW_ROOT/logs/vllm.pid"
    python -c "
from wmw.evaluation.open_vlm import wait_for_server
print('READY' if wait_for_server('$VLLM_URL', timeout_s=900) else 'TIMEOUT')
"
}

stage_dpo_eval() {
    log "── DPO: evaluating adapter"
    python scripts/reproduce/eval_dpo_model.py \
        --base-model qwen25_vl_7b \
        --adapter-path "$CKPT_DIR/qwen25vl_7b_wmw_dpo" \
        --max-examples "$N_EVAL" \
        --data "$DATA_DIR/eval_data/merged_eval.jsonl" \
        --pairs "$DATA_DIR/preference_pairs_seed.jsonl"
}

stage_stress() {
    log "── stress: running stress tests for each open model"
    for m in qwen25_vl_7b internvl3_8b llava_onevision_7b molmo_7b; do
        if [[ -f "$DATA_DIR/results/${m}_full_trace.json" ]]; then
            python scripts/run_evaluation.py \
                --model "$m" \
                --stages stress \
                --output-dir "$DATA_DIR" || log "stress failed for $m (continuing)"
        fi
    done
}

stage_tables() {
    log "── tables: computing final paper tables"
    python scripts/analysis/verifier_agreement.py \
        --results-dir "$DATA_DIR/results" \
        --output "$DATA_DIR/results/verifier_agreement.json" \
        --tex-output "$DATA_DIR/results/table_judge_agreement.tex"
    python scripts/analysis/external_transfer.py \
        --results-dir "$DATA_DIR/results" \
        --output "$DATA_DIR/results/external_transfer.json" \
        --tex-output "$DATA_DIR/results/table_external_transfer.tex"
    python scripts/analysis/build_final_tables.py \
        --results-dir "$DATA_DIR/results" \
        --out-prefix "$DATA_DIR/results" \
        --judge-agreement "$DATA_DIR/results/verifier_agreement.json"
    log "── tables: done. See $DATA_DIR/results/paper_tables_all.tex"
}

run_default() {
    stage_setup
    stage_prep

    stage_serve_qwen7;   stage_eval_qwen7
    stage_serve_internvl; stage_eval_internvl
    stage_serve_llava;   stage_eval_llava
    stage_serve_molmo;   stage_eval_molmo
    stage_serve_qwen32;  stage_eval_qwen32

    stage_dpo_train
    stage_serve_dpo;     stage_dpo_eval

    stop_vllm
    stage_stress
    stage_tables
}

if [[ $# -eq 0 ]]; then
    run_default
else
    for stage in "$@"; do
        if declare -f "stage_$stage" >/dev/null; then
            "stage_$stage"
        elif declare -f "$stage" >/dev/null; then
            "$stage"
        else
            log "Unknown stage: $stage"
            exit 1
        fi
    done
fi
log "── run_all.sh: COMPLETE"
