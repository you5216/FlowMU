#!/usr/bin/env bash
set -euo pipefail

AGENT_DIR="/disk8T/lkh/python_projects/agent_unlearning_zhx/agent"
PY="/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/env52/bin/python"
VLLM="/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/52vllm/bin/vllm"

MODEL_PATH="${AGENT_DIR}/final_performance/literaryQA/for10_re90/merged_model/rmu/literaryQA_for10_re90_rmu_nokl_ce13_20260701"
JUDGE_MODEL="/disk8T/lkh/python_projects/llms/Qwen2.5-VL-7B-Instruct"
FORGET_PATH="/disk8T/lkh/python_projects/datasets/LiteraryQA/literaryqa_annotations/rag/forget10.json"
EMPTY_RETAIN="${AGENT_DIR}/final_performance/literaryQA/for10_re90/empty_retain_for_forget_only.json"
RETRIEVAL_CSV="${AGENT_DIR}/final_performance/literaryQA/for10_re90/retrieval_cache/final_graph_full_forget1970_retain17091_top50_20260629/graph_path_eval_results.csv"
RAG_GRAPH="${AGENT_DIR}/final_performance/literaryQA/for10_re90/rag_memory_unlearning_graph/rag_unlearning_graph.pkl"
MEMORY_GRAPH="${AGENT_DIR}/final_performance/literaryQA/for10_re90/rag_memory_unlearning_graph/memory_unlearning_graph.pkl"

EVAL_DIR="${AGENT_DIR}/final_performance/literaryQA/for10_re90/full_pipeline_eval/rmu_ce13_forget10_graphhybrid_20260701"
LOG_DIR="${EVAL_DIR}/logs"
MODEL_PORT="${MODEL_PORT:-18080}"
JUDGE_PORT="${JUDGE_PORT:-18081}"

mkdir -p "${LOG_DIR}" "${EVAL_DIR}"

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
rm -rf "${EVAL_DIR}/judge_qwen25vl"

start_vllm "${MODEL_PATH}" "rmu_ce13" "${MODEL_PORT}" "${LOG_DIR}/vllm_generate.log" "0.86"
log "Generating forget10 full-pipeline outputs with precomputed retrieval cache"
VLLM_API_BASE="http://127.0.0.1:${MODEL_PORT}/v1" \
VLLM_MODEL_NAME="rmu_ce13" \
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
  --retain-path "${EMPTY_RETAIN}" \
  --forget-limit 1970 \
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

start_vllm "${JUDGE_MODEL}" "judge" "${JUDGE_PORT}" "${LOG_DIR}/vllm_judge.log" "0.86"
log "Judging forget10 full-pipeline outputs"
JUDGE_NUM_WORKERS="${JUDGE_NUM_WORKERS:-24}" \
"${PY}" evaluation/llm_judge.py \
  --files "${EVAL_DIR}/results.csv" \
  --labels "rmu_ce13_full_forget10" \
  --api-url "http://127.0.0.1:${JUDGE_PORT}/v1" \
  --model judge \
  --output-dir "${EVAL_DIR}/judge_qwen25vl" \
  > "${LOG_DIR}/judge.log" 2>&1
kill_port_vllm "${JUDGE_PORT}"

log "Summary:"
cat "${EVAL_DIR}/judge_qwen25vl/judge_summary_custom.json"
