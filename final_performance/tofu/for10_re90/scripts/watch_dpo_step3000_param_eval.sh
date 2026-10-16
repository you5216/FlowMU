#!/usr/bin/env bash
set -euo pipefail

AGENT_DIR="/disk8T/lkh/python_projects/agent_unlearning_zhx/agent"
ENV52_PY="/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/env52/bin/python"
VLLM_BIN="/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/52vllm/bin/vllm"

GPU_ID="${GPU_ID:-0}"
MIN_FREE_MB="${MIN_FREE_MB:-19000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
ANSWER_PORT="${ANSWER_PORT:-8002}"
JUDGE_PORT="${JUDGE_PORT:-8003}"
NUM_WORKERS="${NUM_WORKERS:-8}"
JUDGE_NUM_WORKERS="${JUDGE_NUM_WORKERS:-8}"

BASE_MODEL="${AGENT_DIR}/merge_model/native_llm"
MERGED_MODEL="${AGENT_DIR}/final_performance/tofu/for10_re90/merged_model/dpo/tofu_for10_re90_dpo_native_top20rr09_step3000_20260702"
JUDGE_MODEL="/home/likunhao/likunhao52shared/llms/Qwen2.5-VL-7B-Instruct"
FORGET_PATH="/disk8T/lkh/python_projects/datasets/TOFU/forget10.json"

RUN_NAME="dpo_native_top20rr09_step3000_forget400_20260702"
EVAL_DIR="${AGENT_DIR}/final_performance/tofu/for10_re90/param_eval/${RUN_NAME}"
LOG_DIR="${AGENT_DIR}/final_performance/tofu/for10_re90/logs/dpo_step3000_param_eval_20260702"

mkdir -p "${EVAL_DIR}" "${LOG_DIR}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_DIR}/watcher.log"
}

gpu_free_mb() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU_ID}" | head -1 | tr -d ' '
}

port_busy() {
  ss -ltn "sport = :$1" | grep -q LISTEN
}

wait_for_gpu() {
  while true; do
    local free_mb
    free_mb="$(gpu_free_mb || echo 0)"
    log "GPU${GPU_ID} free=${free_mb}MB need>=${MIN_FREE_MB}MB"
    if [[ "${free_mb}" =~ ^[0-9]+$ ]] && (( free_mb >= MIN_FREE_MB )); then
      return 0
    fi
    sleep "${POLL_SECONDS}"
  done
}

wait_openai_server() {
  local port="$1"
  local pid_file="$2"
  local vllm_log="$3"
  local served_name="${4:-}"
  local waited=0
  while true; do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      log "vLLM ready port=${port} waited=${waited}s"
      return 0
    fi
    if ! ps -p "$(cat "${pid_file}")" >/dev/null 2>&1 && ! pgrep -af "vllm serve" | grep -F -- "--port ${port}" >/dev/null 2>&1; then
      log "ERROR vLLM exited before ready on port=${port}"
      tail -120 "${vllm_log}" | tee -a "${LOG_DIR}/watcher.log" || true
      return 1
    fi
    if (( waited > 0 && waited % 30 == 0 )); then
      log "waiting vLLM port=${port} served=${served_name} waited=${waited}s"
      tail -10 "${vllm_log}" | tee -a "${LOG_DIR}/watcher.log" || true
    fi
    sleep 5
    waited=$((waited + 5))
  done
}

start_vllm() {
  local model_path="$1"
  local served_name="$2"
  local port="$3"
  local pid_file="$4"
  local vllm_log="$5"

  if port_busy "${port}"; then
    log "ERROR port ${port} is already busy. Set ANSWER_PORT/JUDGE_PORT to another free port."
    ss -ltn "sport = :${port}" | tee -a "${LOG_DIR}/watcher.log" || true
    return 1
  fi

  log "starting vLLM served=${served_name} port=${port} model=${model_path}"
  nohup bash -lc "
    exec env CUDA_VISIBLE_DEVICES='${GPU_ID}' VLLM_USE_V1=0 '${VLLM_BIN}' serve '${model_path}' \
      --host 0.0.0.0 --port '${port}' \
      --served-model-name '${served_name}' \
      --max-model-len 2048 \
      --max-num-seqs 8 \
      --gpu-memory-utilization 0.85 \
      --enforce-eager --trust-remote-code
  " > "${vllm_log}" 2>&1 &
  echo $! > "${pid_file}"
  wait_openai_server "${port}" "${pid_file}" "${vllm_log}" "${served_name}"
}

stop_pid_file() {
  local pid_file="$1"
  if [[ -f "${pid_file}" ]] && ps -p "$(cat "${pid_file}")" >/dev/null 2>&1; then
    log "stopping pid $(cat "${pid_file}")"
    kill "$(cat "${pid_file}")" || true
    sleep 8
  fi
}

main() {
  cd "${AGENT_DIR}"
  log "watcher start run=${RUN_NAME}"
  log "will keep judge vLLM alive after judge; do not close it automatically"

  if [[ ! -f "${MERGED_MODEL}/config.json" ]]; then
    log "ERROR missing merged model: ${MERGED_MODEL}"
    exit 2
  fi
  if [[ ! -f "${FORGET_PATH}" ]]; then
    log "ERROR missing forget path: ${FORGET_PATH}"
    exit 2
  fi

  wait_for_gpu

  start_vllm "${MERGED_MODEL}" "dpo_step3000" "${ANSWER_PORT}" "${LOG_DIR}/answer_vllm.pid" "${LOG_DIR}/answer_vllm.log"

  log "running parametric generation"
  VLLM_API_BASE="http://127.0.0.1:${ANSWER_PORT}/v1" \
  VLLM_MODEL_NAME="dpo_step3000" \
  QWEN_INFER_BACKEND="vllm_api" \
  "${ENV52_PY}" -m phase2.pipeline.parametric_eval \
    --base-model-path "${BASE_MODEL}" \
    --lora-path "${MERGED_MODEL}" \
    --forget-path "${FORGET_PATH}" \
    --retain-path "${FORGET_PATH}" \
    --forget-limit 400 \
    --retain-limit 0 \
    --output-csv "${EVAL_DIR}/results.csv" \
    --output-json "${EVAL_DIR}/results.json" \
    --backend vllm_api \
    --num-workers "${NUM_WORKERS}" \
    --max-new-tokens 128 \
    --temperature 0.0 \
    --no-resume \
    > "${LOG_DIR}/parametric_eval.log" 2>&1
  log "parametric generation done"

  stop_pid_file "${LOG_DIR}/answer_vllm.pid"
  wait_for_gpu

  start_vllm "${JUDGE_MODEL}" "judge" "${JUDGE_PORT}" "${LOG_DIR}/judge_vllm.pid" "${LOG_DIR}/judge_vllm.log"

  log "running judge"
  "${ENV52_PY}" evaluation/llm_judge.py \
    --files "${EVAL_DIR}/results.csv" \
    --labels "${RUN_NAME}" \
    --api-url "http://127.0.0.1:${JUDGE_PORT}/v1" \
    --model judge \
    --output-dir "${EVAL_DIR}/judge_qwen25" \
    --num-workers "${JUDGE_NUM_WORKERS}" \
    > "${LOG_DIR}/judge.log" 2>&1
  log "judge done"

  local judge_csv
  judge_csv="$(find "${EVAL_DIR}/judge_qwen25" -name 'judge_*.csv' | head -1)"
  if [[ -z "${judge_csv}" ]]; then
    log "ERROR judge csv not found"
    exit 3
  fi

  log "running final metrics"
  "${ENV52_PY}" evaluation/final_metrics.py \
    --files "${judge_csv}" \
    --labels "${RUN_NAME}" \
    --output-dir "${EVAL_DIR}/final_metrics" \
    > "${LOG_DIR}/final_metrics.log" 2>&1
  cat "${EVAL_DIR}/final_metrics/final_metrics_summary.json" | tee -a "${LOG_DIR}/watcher.log"

  log "DONE. Judge vLLM remains alive pid=$(cat "${LOG_DIR}/judge_vllm.pid") port=${JUDGE_PORT}"
  log "To stop later: kill $(cat "${LOG_DIR}/judge_vllm.pid")"
  while ps -p "$(cat "${LOG_DIR}/judge_vllm.pid")" >/dev/null 2>&1; do
    sleep 300
  done
}

main "$@"
