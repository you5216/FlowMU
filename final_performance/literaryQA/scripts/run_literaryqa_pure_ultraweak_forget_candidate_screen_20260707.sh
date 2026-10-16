#!/usr/bin/env bash
set -euo pipefail

AGENT_DIR="/disk8T/lkh/python_projects/agent_unlearning_zhx/agent"
PY="${PY:-/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/env52/bin/python}"
VLLM="${VLLM:-/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/52vllm/bin/vllm}"

BASE_MODEL="${AGENT_DIR}/final_performance/literaryQA/base/lora/merged/literary_neative_model"
JUDGE_MODEL="${JUDGE_MODEL:-/disk8T/lkh/python_projects/llms/Qwen2.5-VL-7B-Instruct}"
DATA_ROOT="/disk8T/lkh/python_projects/datasets/LiteraryQA/literaryqa_annotations/rag"
RUN_ROOT="${AGENT_DIR}/final_performance/literaryQA/pure_ultraweak_candidate_screen_20260707"
MODEL_PORT="${MODEL_PORT:-18080}"
JUDGE_PORT="${JUDGE_PORT:-18081}"
GEN_WORKERS="${GEN_WORKERS:-16}"
JUDGE_WORKERS="${JUDGE_WORKERS:-16}"

FOR01_LORA="${AGENT_DIR}/final_performance/literaryQA/for01_re99/lora/ga_pure_ultraweak_literarynative_top50rr06_lr3e4_gn1_ce3_15_20260707/ce_threshold_checkpoints"
FOR05_LORA="${AGENT_DIR}/final_performance/literaryQA/for05_re95/lora/ga_pure_ultraweak_literarynative_top50rr09_lr3e4_gn1_ce3_15_20260707/ce_threshold_checkpoints"

mkdir -p "${RUN_ROOT}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

kill_port_vllm() {
  local port="$1"
  local pids
  pids="$(pgrep -af "vllm.*serve.*--port ${port}" | awk '{print $1}' || true)"
  if [[ -n "${pids}" ]]; then
    log "Stopping vLLM on port ${port}: ${pids}"
    kill ${pids} 2>/dev/null || true
    sleep 8
  fi
}

wait_server() {
  local port="$1"
  local name="$2"
  local log_file="$3"
  local waited=0
  local max_wait="${4:-1200}"
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
        log "ERROR: ${name} exited before ready; tail ${log_file}"
        tail -n 160 "${log_file}" || true
        exit 1
      fi
    fi
    if (( waited >= max_wait )); then
      log "ERROR: timeout waiting for ${name}; tail ${log_file}"
      tail -n 160 "${log_file}" || true
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
  log "Starting vLLM: name=${served_name}, port=${port}, model=${model_path}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${VLLM}" serve "${model_path}" \
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

merge_one() {
  local label="$1"
  local lora_path="$2"
  local merged_dir="$3"
  local log_dir="$4"
  if [[ -f "${merged_dir}/config.json" ]]; then
    log "Merged model exists: ${merged_dir}"
    return 0
  fi
  mkdir -p "${merged_dir}" "${log_dir}"
  log "Merging ${label}: ${lora_path}"
  "${PY}" -m phase2.pipeline.merge_lora_for_vllm \
    --base-model-path "${BASE_MODEL}" \
    --lora-path "${lora_path}" \
    --output-dir "${merged_dir}" \
    --torch-dtype float16 \
    --device-map cpu \
    --max-shard-size 4GB \
    --label "${label}" \
    --force \
    > "${log_dir}/merge.log" 2>&1
}

generate_one() {
  local label="$1"
  local merged_dir="$2"
  local forget_path="$3"
  local limit="$4"
  local eval_dir="$5"
  local empty_retain="${eval_dir}/empty_retain.json"
  local log_dir="${eval_dir}/logs"
  mkdir -p "${log_dir}"
  printf '[]\n' > "${empty_retain}"
  rm -f "${eval_dir}/results.csv" "${eval_dir}/results.json"
  start_vllm "${merged_dir}" "${label}" "${MODEL_PORT}" "${log_dir}/vllm_generate.log" "0.86"
  log "Generating forget-only answers: ${label}, limit=${limit}"
  VLLM_API_BASE="http://127.0.0.1:${MODEL_PORT}/v1" \
  VLLM_MODEL_NAME="${label}" \
  QWEN_INFER_BACKEND="vllm_api" \
  QWEN_EVAL_NUM_WORKERS="${GEN_WORKERS}" \
  "${PY}" -m phase2.pipeline.parametric_eval \
    --backend vllm_api \
    --base-model-path "${merged_dir}" \
    --lora-path "" \
    --forget-path "${forget_path}" \
    --retain-path "${empty_retain}" \
    --forget-limit "${limit}" \
    --retain-limit 0 \
    --max-new-tokens 96 \
    --temperature 0.0 \
    --num-workers "${GEN_WORKERS}" \
    --output-csv "${eval_dir}/results.csv" \
    --output-json "${eval_dir}/results.json" \
    --no-resume \
    > "${log_dir}/generate.log" 2>&1
  kill_port_vllm "${MODEL_PORT}"
}

declare -a SPECS=(
  "for01_re99|pure_ultraweak_for01_ce7p5_step181|${FOR01_LORA}/ce7.5_step181|${DATA_ROOT}/forget01.json|204"
  "for01_re99|pure_ultraweak_for01_ce10_step256|${FOR01_LORA}/ce10_step256|${DATA_ROOT}/forget01.json|204"
  "for05_re95|pure_ultraweak_for05_ce7p5_step149|${FOR05_LORA}/ce7.5_step149|${DATA_ROOT}/forget05.json|992"
  "for05_re95|pure_ultraweak_for05_ce10_step233|${FOR05_LORA}/ce10_step233|${DATA_ROOT}/forget05.json|992"
)

declare -a RESULT_FILES=()
declare -a LABELS=()
for spec in "${SPECS[@]}"; do
  IFS='|' read -r cell label lora_path forget_path limit <<< "${spec}"
  [[ -d "${lora_path}" ]] || { echo "Missing LoRA checkpoint: ${lora_path}" >&2; exit 1; }
  merged_dir="${RUN_ROOT}/${cell}/merged/${label}"
  eval_dir="${RUN_ROOT}/${cell}/eval/${label}"
  merge_one "${label}" "${lora_path}" "${merged_dir}" "${eval_dir}/logs"
  generate_one "${label}" "${merged_dir}" "${forget_path}" "${limit}" "${eval_dir}"
  RESULT_FILES+=("${eval_dir}/results.csv")
  LABELS+=("${label}")
done

JUDGE_OUT="${RUN_ROOT}/judge_qwen25vl"
rm -rf "${JUDGE_OUT}"
mkdir -p "${JUDGE_OUT}/logs"
start_vllm "${JUDGE_MODEL}" "judge" "${JUDGE_PORT}" "${JUDGE_OUT}/logs/vllm_judge.log" "0.86"
log "Judging candidate outputs"
JUDGE_NUM_WORKERS="${JUDGE_WORKERS}" \
"${PY}" evaluation/llm_judge.py \
  --files "${RESULT_FILES[@]}" \
  --labels "${LABELS[@]}" \
  --api-url "http://127.0.0.1:${JUDGE_PORT}/v1" \
  --model "judge" \
  --num-workers "${JUDGE_WORKERS}" \
  --output-dir "${JUDGE_OUT}" \
  > "${JUDGE_OUT}/logs/judge.log" 2>&1
kill_port_vllm "${JUDGE_PORT}"

"${PY}" evaluation/final_metrics.py \
  --files "${JUDGE_OUT}"/judge_*.csv \
  --output-dir "${RUN_ROOT}/final_metrics" \
  | tee "${RUN_ROOT}/candidate_screen_metrics.txt"

log "Done: ${RUN_ROOT}/candidate_screen_metrics.txt"
