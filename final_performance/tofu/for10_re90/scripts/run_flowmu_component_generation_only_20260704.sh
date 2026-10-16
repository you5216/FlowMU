#!/usr/bin/env bash
set -euo pipefail

AGENT_DIR="/disk8T/lkh/python_projects/agent_unlearning_zhx/agent"
ENV52_PY="/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/env52/bin/python"
ROOT="$AGENT_DIR/final_performance/tofu/for10_re90"
LOG_ROOT="$ROOT/logs/flowmu_component_ablation_for10_20260704"
WRAPPER="$ROOT/scripts/run_step6_component_ablation.py"

BASE_MODEL="$AGENT_DIR/merge_model/native_llm"
GA_MODEL="$ROOT/merged_model/ga/tofu_for10_re90_ga_native_top20rr09_ce60_20260702"
RAG_GRAPH="$ROOT/phase1_gat_retrain_gatselect_rag0925_memory0925_20260702/rag_unlearning_graph.pkl"
MEMORY_GRAPH="$ROOT/phase1_gat_retrain_gatselect_rag0925_memory0925_20260702/memory_unlearning_graph.pkl"
RETRIEVAL_CSV="$ROOT/retrieval_cache/final_gatselect_rag0925_memory0925_forget10_retain90_top50_20260702/graph_path_eval_results.csv"
FORGET_PATH="/disk8T/lkh/python_projects/datasets/TOFU/re_split/forget10.json"
RETAIN_PATH="/disk8T/lkh/python_projects/datasets/TOFU/re_split/retain90.json"

ANSWER_PORT="${ANSWER_PORT:-8002}"
NUM_WORKERS="${NUM_WORKERS:-8}"

mkdir -p "$LOG_ROOT"
cd "$AGENT_DIR"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG_ROOT/generation_only_runner.log"
}

require_ready() {
  curl -fsS "http://127.0.0.1:$ANSWER_PORT/v1/models" >/dev/null || {
    log "ERROR answer vLLM is not ready on port $ANSWER_PORT"
    exit 2
  }
}

run_one() {
  local label="$1"
  local run_name="$2"
  shift 2
  local out_dir="$ROOT/full_pipeline_eval/$run_name"
  local log_dir="$LOG_ROOT/$run_name"
  mkdir -p "$out_dir" "$log_dir"
  log "START $label -> $out_dir"
  VLLM_API_BASE="http://127.0.0.1:$ANSWER_PORT/v1" VLLM_MODEL_NAME="ga_ce60" QWEN_INFER_BACKEND="vllm_api" VLLM_API_TIMEOUT="${VLLM_API_TIMEOUT:-180}" \
  "$ENV52_PY" "$WRAPPER" \
    --base-model-path "$BASE_MODEL" \
    --lora-path "$GA_MODEL" \
    --rag-graph-path "$RAG_GRAPH" \
    --memory-graph-path "$MEMORY_GRAPH" \
    --retrieval-results-csv "$RETRIEVAL_CSV" \
    --forget-path "$FORGET_PATH" \
    --retain-path "$RETAIN_PATH" \
    --forget-limit 0 \
    --retain-limit 0 \
    --output-csv "$out_dir/results.csv" \
    --output-json "$out_dir/results.json" \
    --backend vllm_api \
    --max-new-tokens 128 \
    --temperature 0.0 \
    --num-workers "$NUM_WORKERS" \
    --rag-fallback-top-k 3 \
    --memory-max-hits 3 \
    --checkpoint-every 100 \
    "$@" \
    2>&1 | tee "$log_dir/generation.log"
  log "DONE $label"
}

main() {
  require_ready
  run_one "FlowMU-R remove RAG, keep Memory" "flowmu_r_no_rag_ga_ce60_full4000_component_ablation_20260704" --no-rag
  run_one "FlowMU-M remove Memory, keep RAG" "flowmu_m_no_memory_ga_ce60_full4000_component_ablation_20260704" --no-memory
  run_one "FlowMU-R&M remove both RAG and Memory" "flowmu_rm_no_rag_no_memory_ga_ce60_full4000_component_ablation_20260704" --no-rag --no-memory
  log "ALL GENERATION DONE"
}

main "$@"
