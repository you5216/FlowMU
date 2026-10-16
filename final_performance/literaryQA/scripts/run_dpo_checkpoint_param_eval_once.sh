#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <for05_re95|for10_re90> <step> [limit]" >&2
  exit 2
fi

SPLIT="$1"
STEP="$2"
LIMIT="${3:-200}"

AGENT_DIR="/disk8T/lkh/python_projects/agent_unlearning_zhx/agent"
PY="/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/env52/bin/python"
VLLM="/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/52vllm/bin/vllm"

BASE_MODEL="${AGENT_DIR}/final_performance/literaryQA/base/lora/merged/literary_neative_model"
JUDGE_MODEL="/disk8T/lkh/python_projects/llms/Qwen2.5-VL-7B-Instruct"
DATA_ROOT="/disk8T/lkh/python_projects/datasets/LiteraryQA/literaryqa_annotations/rag"

case "${SPLIT}" in
  for05_re95)
    FORGET_PATH="${DATA_ROOT}/forget05.json"
    DPO_DIR="${AGENT_DIR}/final_performance/literaryQA/for05_re95/lora/dpo_enhanced_strong_literarynative_graphqa2217_eqretain_b02_lr8e5_as080_key025_e5_save300_20260701"
    ;;
  for10_re90)
    FORGET_PATH="${DATA_ROOT}/forget10.json"
    DPO_DIR="${AGENT_DIR}/final_performance/literaryQA/for10_re90/lora/dpo_enhanced_strong_literarynative_graphqa2230_eqretain_b02_lr8e5_as080_key025_e5_save300_20260701"
    ;;
  *)
    echo "Unsupported split: ${SPLIT}" >&2
    exit 2
    ;;
esac

CHECKPOINT_DIR="${DPO_DIR}/checkpoint-step${STEP}"
if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT_DIR}" >&2
  exit 1
fi

EMPTY_RETAIN="${AGENT_DIR}/final_performance/literaryQA/${SPLIT}/empty_retain_for_forget_only.json"
mkdir -p "$(dirname "${EMPTY_RETAIN}")"
[[ -f "${EMPTY_RETAIN}" ]] || printf '[]\n' > "${EMPTY_RETAIN}"

MERGED_DIR="${AGENT_DIR}/final_performance/literaryQA/${SPLIT}/merged_model/dpo/literaryQA_${SPLIT}_dpo_step${STEP}_20260702"
EVAL_DIR="${AGENT_DIR}/final_performance/literaryQA/${SPLIT}/param_eval/dpo_step${STEP}_forget_${LIMIT}_20260702"
LOG_DIR="${EVAL_DIR}/logs"

MODEL_PORT="${MODEL_PORT:-18080}"
JUDGE_PORT="${JUDGE_PORT:-18081}"
MODEL_NAME="dpo_step${STEP}"
JUDGE_NAME="judge"

mkdir -p "${LOG_DIR}" "${MERGED_DIR}" "${EVAL_DIR}"

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

if [[ ! -f "${MERGED_DIR}/config.json" ]]; then
  log "Merging ${SPLIT} DPO step${STEP}: ${CHECKPOINT_DIR}"
  "${PY}" -m phase2.pipeline.merge_lora_for_vllm \
    --base-model-path "${BASE_MODEL}" \
    --lora-path "${CHECKPOINT_DIR}" \
    --output-dir "${MERGED_DIR}" \
    --torch-dtype float16 \
    --device-map cpu \
    --max-shard-size 4GB \
    --label "literaryQA_${SPLIT}_dpo_step${STEP}" \
    --force \
    > "${LOG_DIR}/merge.log" 2>&1
else
  log "Merged model already exists: ${MERGED_DIR}"
fi

rm -f "${EVAL_DIR}/results.csv" "${EVAL_DIR}/results.json"
rm -rf "${EVAL_DIR}/judge_qwen25vl"

start_vllm "${MERGED_DIR}" "${MODEL_NAME}" "${MODEL_PORT}" "${LOG_DIR}/vllm_generate.log" "0.86"
log "Generating pure-param forget answers: split=${SPLIT}, step=${STEP}, limit=${LIMIT}"
VLLM_API_BASE="http://127.0.0.1:${MODEL_PORT}/v1" \
VLLM_MODEL_NAME="${MODEL_NAME}" \
QWEN_INFER_BACKEND="vllm_api" \
QWEN_EVAL_NUM_WORKERS="${QWEN_EVAL_NUM_WORKERS:-16}" \
QWEN_TOP_K=1 \
"${PY}" -m phase2.pipeline.parametric_eval \
  --backend vllm_api \
  --base-model-path "${MERGED_DIR}" \
  --lora-path "" \
  --forget-path "${FORGET_PATH}" \
  --retain-path "${EMPTY_RETAIN}" \
  --forget-limit "${LIMIT}" \
  --retain-limit 0 \
  --max-new-tokens 96 \
  --temperature 0.0 \
  --output-csv "${EVAL_DIR}/results.csv" \
  --output-json "${EVAL_DIR}/results.json" \
  --no-resume \
  > "${LOG_DIR}/generate.log" 2>&1
kill_port_vllm "${MODEL_PORT}"

start_vllm "${JUDGE_MODEL}" "${JUDGE_NAME}" "${JUDGE_PORT}" "${LOG_DIR}/vllm_judge.log" "0.86"
log "Judging pure-param forget answers: split=${SPLIT}, step=${STEP}, limit=${LIMIT}"
JUDGE_NUM_WORKERS="${JUDGE_NUM_WORKERS:-16}" \
"${PY}" evaluation/llm_judge.py \
  --files "${EVAL_DIR}/results.csv" \
  --labels "dpo_step${STEP}_${SPLIT}_${LIMIT}" \
  --api-url "http://127.0.0.1:${JUDGE_PORT}/v1" \
  --model "${JUDGE_NAME}" \
  --output-dir "${EVAL_DIR}/judge_qwen25vl" \
  > "${LOG_DIR}/judge.log" 2>&1
kill_port_vllm "${JUDGE_PORT}"

log "Summary:"
cat "${EVAL_DIR}/judge_qwen25vl/judge_summary_custom.json"
