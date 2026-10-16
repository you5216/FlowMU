#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 7 ]]; then
  echo "Usage: $0 <label> <model_path> <split> <forget_path> <retain_path> <retrieval_csv> <eval_dir>" >&2
  exit 2
fi

LABEL="$1"
MODEL_PATH="$2"
SPLIT="$3"
FORGET_PATH="$4"
RETAIN_PATH="$5"
RETRIEVAL_CSV="$6"
EVAL_DIR="$7"

AGENT_DIR="/disk8T/lkh/python_projects/agent_unlearning_zhx/agent"
PY="/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/env52/bin/python"
VLLM="/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/52vllm/bin/vllm"

RAG_GRAPH="${AGENT_DIR}/final_performance/literaryQA/${SPLIT}/rag_memory_unlearning_graph/rag_unlearning_graph.pkl"
MEMORY_GRAPH="${AGENT_DIR}/final_performance/literaryQA/${SPLIT}/rag_memory_unlearning_graph/memory_unlearning_graph.pkl"
LOG_DIR="${EVAL_DIR}/logs"
MODEL_PORT="${MODEL_PORT:-18180}"

for p in "${MODEL_PATH}/config.json" "${FORGET_PATH}" "${RETAIN_PATH}" "${RETRIEVAL_CSV}" "${RAG_GRAPH}" "${MEMORY_GRAPH}"; do
  if [[ ! -e "${p}" ]]; then
    echo "Missing required path: ${p}" >&2
    exit 1
  fi
done

mkdir -p "${EVAL_DIR}" "${LOG_DIR}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

kill_port_vllm() {
  local port="$1"
  local pids
  pids="$(pgrep -af "vllm serve .*--port ${port}" | awk '{print $1}' || true)"
  if [[ -n "${pids}" ]]; then
    log "Stopping vLLM on port ${port}: ${pids}"
    kill ${pids} 2>/dev/null || true
    sleep 5
  fi
}

wait_server() {
  local port="$1"
  local name="$2"
  local log_file="$3"
  local waited=0
  local max_wait="${4:-900}"
  log "Waiting for ${name} on port ${port}"
  while true; do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      log "${name} ready on port ${port}"
      return 0
    fi
    if [[ -f "${log_file}.pid" ]]; then
      local pid
      pid="$(cat "${log_file}.pid")"
      if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
        log "ERROR: ${name} vLLM exited before ready; tail ${log_file}"
        tail -n 120 "${log_file}" || true
        exit 1
      fi
    fi
    if (( waited >= max_wait )); then
      log "ERROR: timeout waiting for ${name}; tail ${log_file}"
      tail -n 120 "${log_file}" || true
      exit 1
    fi
    sleep 5
    waited=$((waited + 5))
  done
}

start_vllm() {
  local model_path="$1"
  local served_name="$2"
  local port="$3"
  local log_file="$4"
  local gpu_memory="${5:-0.86}"
  kill_port_vllm "${port}"
  log "Starting vLLM: ${served_name}, port=${port}, model=${model_path}"
  CUDA_VISIBLE_DEVICES=0 "${VLLM}" serve "${model_path}" \
    --host 0.0.0.0 \
    --port "${port}" \
    --served-model-name "${served_name}" \
    --max-model-len 1024 \
    --gpu-memory-utilization "${gpu_memory}" \
    --enforce-eager \
    --trust-remote-code \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --mm-processor-cache-gb 0 \
    > "${log_file}" 2>&1 &
  echo $! > "${log_file}.pid"
  wait_server "${port}" "${served_name}" "${log_file}"
}

rm -f "${EVAL_DIR}/results.csv" "${EVAL_DIR}/results.json"

start_vllm "${MODEL_PATH}" "${LABEL}" "${MODEL_PORT}" "${LOG_DIR}/vllm_generate.log" "0.86"
log "Generating full pipeline: label=${LABEL}, split=${SPLIT}"
VLLM_API_BASE="http://127.0.0.1:${MODEL_PORT}/v1" \
VLLM_MODEL_NAME="${LABEL}" \
QWEN_INFER_BACKEND="vllm_api" \
QWEN_EVAL_NUM_WORKERS="${QWEN_EVAL_NUM_WORKERS:-24}" \
QWEN_TOP_K=1 \
"${PY}" -m phase2.pipeline.step6_full_pipeline_eval \
  --base-model-path "${MODEL_PATH}" \
  --lora-path "" \
  --rag-graph-path "${RAG_GRAPH}" \
  --memory-graph-path "${MEMORY_GRAPH}" \
  --retrieval-results-csv "${RETRIEVAL_CSV}" \
  --forget-path "${FORGET_PATH}" \
  --retain-path "${RETAIN_PATH}" \
  --forget-limit 0 \
  --retain-limit 0 \
  --output-csv "${EVAL_DIR}/results.csv" \
  --output-json "${EVAL_DIR}/results.json" \
  --backend vllm_api \
  --max-new-tokens 128 \
  --temperature 0.0 \
  --rag-fallback-top-k 3 \
  --memory-max-hits 3 \
  --no-resume \
  > "${LOG_DIR}/generate.log" 2>&1
kill_port_vllm "${MODEL_PORT}"

log "Generation finished: ${EVAL_DIR}/results.csv"
