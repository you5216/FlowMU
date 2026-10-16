#!/usr/bin/env bash
set -euo pipefail

AGENT_DIR="/disk8T/lkh/python_projects/agent_unlearning_zhx/agent"
ENV52_PY="/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/env52/bin/python"
VLLM_BIN="/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/52vllm/bin/vllm"
ROOT="$AGENT_DIR/final_performance/tofu/for10_re90"
LOG_ROOT="$ROOT/logs/flowmu_llm_ablation_for10_20260704"

BASE_MODEL="$AGENT_DIR/merge_model/native_llm"
RAG_GRAPH="$ROOT/phase1_gat_retrain_gatselect_rag0925_memory0925_20260702/rag_unlearning_graph.pkl"
MEMORY_GRAPH="$ROOT/phase1_gat_retrain_gatselect_rag0925_memory0925_20260702/memory_unlearning_graph.pkl"
RETRIEVAL_CSV="$ROOT/retrieval_cache/final_gatselect_rag0925_memory0925_forget10_retain90_top50_20260702/graph_path_eval_results.csv"
FORGET_PATH="/disk8T/lkh/python_projects/datasets/TOFU/re_split/forget10.json"
RETAIN_PATH="/disk8T/lkh/python_projects/datasets/TOFU/re_split/retain90.json"
OUT_DIR="$ROOT/full_pipeline_eval/flowmu_llm_native_no_param_edit_full4000_component_ablation_20260704"

ANSWER_PORT="${ANSWER_PORT:-8002}"
GPU_ID="${GPU_ID:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"

mkdir -p "$LOG_ROOT" "$OUT_DIR"
cd "$AGENT_DIR"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG_ROOT/runner.log"
}

stop_port() {
  local port="$1"
  local pids
  pids="$(lsof -ti tcp:"$port" || true)"
  if [[ -n "$pids" ]]; then
    log "stopping existing server on port=$port pids=$pids"
    kill $pids || true
    sleep 5
  fi
}

wait_server() {
  local port="$1"
  local pid_file="$2"
  local waited=0
  while (( waited < 900 )); do
    if curl -fsS "http://127.0.0.1:$port/v1/models" >/dev/null; then
      log "vLLM ready port=$port waited=${waited}s"
      return 0
    fi
    if [[ -s "$pid_file" ]] && ! kill -0 "$(cat "$pid_file")" 2>/dev/null; then
      log "ERROR vLLM exited before ready on port=$port"
      tail -120 "$LOG_ROOT/native_vllm.log" || true
      return 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
  log "ERROR timeout waiting for vLLM port=$port"
  tail -120 "$LOG_ROOT/native_vllm.log" || true
  return 1
}

for path in "$ENV52_PY" "$VLLM_BIN" "$BASE_MODEL" "$RAG_GRAPH" "$MEMORY_GRAPH" "$RETRIEVAL_CSV" "$FORGET_PATH" "$RETAIN_PATH"; do
  [[ -e "$path" ]] || { log "ERROR missing path: $path"; exit 2; }
done

log "RUNNER START FlowMU-LLM native/base LLM ablation"
stop_port "$ANSWER_PORT"

nohup bash -lc "exec env CUDA_VISIBLE_DEVICES='$GPU_ID' VLLM_USE_V1=0 '$VLLM_BIN' serve '$BASE_MODEL' \
  --host 0.0.0.0 \
  --port '$ANSWER_PORT' \
  --served-model-name native_llm \
  --max-model-len 2048 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --trust-remote-code" > "$LOG_ROOT/native_vllm.log" 2>&1 &
echo $! > "$LOG_ROOT/native_vllm.pid"
wait_server "$ANSWER_PORT" "$LOG_ROOT/native_vllm.pid"

log "START FlowMU-LLM generation -> $OUT_DIR"
VLLM_API_BASE="http://127.0.0.1:$ANSWER_PORT/v1" \
VLLM_MODEL_NAME="native_llm" \
QWEN_INFER_BACKEND="vllm_api" \
VLLM_API_TIMEOUT="${VLLM_API_TIMEOUT:-180}" \
"$ENV52_PY" -m phase2.pipeline.step6_full_pipeline_eval \
  --base-model-path "$BASE_MODEL" \
  --lora-path "" \
  --rag-graph-path "$RAG_GRAPH" \
  --memory-graph-path "$MEMORY_GRAPH" \
  --retrieval-results-csv "$RETRIEVAL_CSV" \
  --forget-path "$FORGET_PATH" \
  --retain-path "$RETAIN_PATH" \
  --forget-limit 0 \
  --retain-limit 0 \
  --output-csv "$OUT_DIR/results.csv" \
  --output-json "$OUT_DIR/results.json" \
  --backend vllm_api \
  --max-new-tokens 128 \
  --temperature 0.0 \
  --num-workers "$NUM_WORKERS" \
  --rag-fallback-top-k 3 \
  --memory-max-hits 3 \
  --checkpoint-every 100 \
  2>&1 | tee "$LOG_ROOT/generation.log"

log "DONE FlowMU-LLM generation; native vLLM remains alive pid=$(cat "$LOG_ROOT/native_vllm.pid") port=$ANSWER_PORT"
