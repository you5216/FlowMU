#!/usr/bin/env bash
set -euo pipefail

AGENT_DIR="/disk8T/lkh/python_projects/agent_unlearning_zhx/agent"
PY="${PY:-/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/env52/bin/python}"
VLLM="${VLLM:-/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/52vllm/bin/vllm}"

BASE_MODEL="${AGENT_DIR}/final_performance/literaryQA/base/lora/merged/literary_neative_model"
JUDGE_MODEL="${JUDGE_MODEL:-/disk8T/lkh/python_projects/llms/Qwen2.5-VL-7B-Instruct}"
DATA_ROOT="/disk8T/lkh/python_projects/datasets/LiteraryQA/literaryqa_annotations/rag"

OUT_ROOT="${AGENT_DIR}/final_performance/literaryQA/pure_ce10_full_pipeline_20260707"
MERGED_ROOT="${OUT_ROOT}/merged"
JUDGE_DIR="${OUT_ROOT}/judge_qwen25vl"
METRICS_DIR="${OUT_ROOT}/final_metrics"
LOG_ROOT="${OUT_ROOT}/logs"

MODEL_PORT="${MODEL_PORT:-18180}"
JUDGE_PORT="${JUDGE_PORT:-18181}"
GEN_WORKERS="${GEN_WORKERS:-24}"
JUDGE_WORKERS="${JUDGE_WORKERS:-24}"

mkdir -p "${MERGED_ROOT}" "${JUDGE_DIR}" "${METRICS_DIR}" "${LOG_ROOT}"

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

validate_csv() {
  local path="$1"
  local expect_forget="$2"
  local expect_retain="$3"
  if [[ ! -f "${path}" ]]; then
    return 1
  fi
  "${PY}" - "$path" "$expect_forget" "$expect_retain" <<'PY'
import csv, sys
path, ef, er = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(path, encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
f = sum(1 for r in rows if (r.get("split") or "").strip().lower() == "forget")
r = sum(1 for r in rows if (r.get("split") or "").strip().lower() == "retain")
bad = sum(1 for row in rows if "[EVAL_ERROR]" in str(row.get("prediction") or ""))
if f != ef or r != er or bad:
    print(f"invalid {path}: forget={f}/{ef} retain={r}/{er} eval_errors={bad}")
    raise SystemExit(1)
print(f"valid {path}: forget={f} retain={r}")
PY
}

merge_one() {
  local label="$1"
  local lora_path="$2"
  local merged_dir="$3"
  local log_file="$4"
  if [[ -f "${merged_dir}/config.json" ]]; then
    log "Merged model already exists: ${merged_dir}"
    return 0
  fi
  [[ -d "${lora_path}" ]] || { echo "Missing LoRA checkpoint: ${lora_path}" >&2; exit 1; }
  mkdir -p "${merged_dir}" "$(dirname "${log_file}")"
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
    > "${log_file}" 2>&1
}

generate_one() {
  local ratio="$1"
  local cell="$2"
  local label="$3"
  local model_path="$4"
  local forget_path="$5"
  local retain_path="$6"
  local retrieval_csv="$7"
  local expect_forget="$8"
  local expect_retain="$9"
  local eval_dir="${OUT_ROOT}/${cell}"
  local log_dir="${eval_dir}/logs"
  local rag_graph="${AGENT_DIR}/final_performance/literaryQA/${cell}/rag_memory_unlearning_graph/rag_unlearning_graph.pkl"
  local memory_graph="${AGENT_DIR}/final_performance/literaryQA/${cell}/rag_memory_unlearning_graph/memory_unlearning_graph.pkl"

  for p in "${model_path}/config.json" "${forget_path}" "${retain_path}" "${retrieval_csv}" "${rag_graph}" "${memory_graph}"; do
    [[ -e "${p}" ]] || { echo "Missing required path: ${p}" >&2; exit 1; }
  done

  mkdir -p "${log_dir}"
  if validate_csv "${eval_dir}/results.csv" "${expect_forget}" "${expect_retain}"; then
    log "Generation already valid: ${eval_dir}/results.csv"
    return 0
  fi

  rm -f "${eval_dir}/results.csv" "${eval_dir}/results.json"
  start_vllm "${model_path}" "${label}" "${MODEL_PORT}" "${log_dir}/vllm_generate.log" "0.86"
  log "Generating full pipeline ${label}: forget=${expect_forget}, retain=${expect_retain}"
  VLLM_API_BASE="http://127.0.0.1:${MODEL_PORT}/v1" \
  VLLM_MODEL_NAME="${label}" \
  QWEN_INFER_BACKEND="vllm_api" \
  QWEN_EVAL_NUM_WORKERS="${GEN_WORKERS}" \
  QWEN_TOP_K=1 \
  "${PY}" -m phase2.pipeline.step6_full_pipeline_eval \
    --base-model-path "${model_path}" \
    --lora-path "" \
    --rag-graph-path "${rag_graph}" \
    --memory-graph-path "${memory_graph}" \
    --retrieval-results-csv "${retrieval_csv}" \
    --forget-path "${forget_path}" \
    --retain-path "${retain_path}" \
    --forget-limit 0 \
    --retain-limit 0 \
    --output-csv "${eval_dir}/results.csv" \
    --output-json "${eval_dir}/results.json" \
    --backend vllm_api \
    --max-new-tokens 128 \
    --temperature 0.0 \
    --num-workers "${GEN_WORKERS}" \
    --rag-fallback-top-k 3 \
    --memory-max-hits 3 \
    --checkpoint-every 500 \
    --no-resume \
    > "${log_dir}/generate.log" 2>&1
  kill_port_vllm "${MODEL_PORT}"
  validate_csv "${eval_dir}/results.csv" "${expect_forget}" "${expect_retain}"
}

run_judge_and_metrics() {
  local files=(
    "${OUT_ROOT}/for01_re99/results.csv"
    "${OUT_ROOT}/for05_re95/results.csv"
    "${OUT_ROOT}/for10_re90/results.csv"
  )
  local labels=(
    "pure_ce10_for01"
    "pure_ce10_for05"
    "pure_ce10_for10"
  )
  for f in "${files[@]}"; do
    [[ -f "${f}" ]] || { echo "Missing generation CSV for judge: ${f}" >&2; exit 1; }
  done

  rm -rf "${JUDGE_DIR}" "${METRICS_DIR}"
  mkdir -p "${JUDGE_DIR}/logs" "${METRICS_DIR}"
  start_vllm "${JUDGE_MODEL}" "judge" "${JUDGE_PORT}" "${JUDGE_DIR}/logs/vllm_judge.log" "0.86"
  log "Judging three LiteraryQA PURE ce10 full-pipeline result files"
  JUDGE_NUM_WORKERS="${JUDGE_WORKERS}" \
  "${PY}" evaluation/llm_judge.py \
    --files "${files[@]}" \
    --labels "${labels[@]}" \
    --api-url "http://127.0.0.1:${JUDGE_PORT}/v1" \
    --model judge \
    --num-workers "${JUDGE_WORKERS}" \
    --output-dir "${JUDGE_DIR}" \
    > "${JUDGE_DIR}/logs/judge.log" 2>&1
  kill_port_vllm "${JUDGE_PORT}"

  "${PY}" evaluation/final_metrics.py \
    --files \
      "${JUDGE_DIR}/judge_pure_ce10_for01.csv" \
      "${JUDGE_DIR}/judge_pure_ce10_for05.csv" \
      "${JUDGE_DIR}/judge_pure_ce10_for10.csv" \
    --labels "${labels[@]}" \
    --output-dir "${METRICS_DIR}" \
    > "${LOG_ROOT}/final_metrics.log" 2>&1
}

cd "${AGENT_DIR}"

FOR01_LORA="${AGENT_DIR}/final_performance/literaryQA/for01_re99/lora/ga_pure_ultraweak_literarynative_top50rr06_lr3e4_gn1_ce3_15_20260707/ce_threshold_checkpoints/ce10_step256"
FOR05_LORA="${AGENT_DIR}/final_performance/literaryQA/for05_re95/lora/ga_pure_ultraweak_literarynative_top50rr09_lr3e4_gn1_ce3_15_20260707/ce_threshold_checkpoints/ce10_step233"
FOR10_LORA="${AGENT_DIR}/final_performance/literaryQA/for10_re90/lora/ga_pure_ultraweak_literarynative_top50rr09_lr3e4_gn1_ce3_15_20260707/ce_threshold_checkpoints/ce10_step196"

FOR01_MODEL="${MERGED_ROOT}/literaryQA_for01_re99_pure_ce10_step256_20260707"
FOR05_MODEL="${MERGED_ROOT}/literaryQA_for05_re95_pure_ce10_step233_20260707"
FOR10_MODEL="${MERGED_ROOT}/literaryQA_for10_re90_pure_ce10_step196_20260707"

merge_one "literaryQA_for01_re99_pure_ce10_step256" "${FOR01_LORA}" "${FOR01_MODEL}" "${LOG_ROOT}/merge_for01.log"
merge_one "literaryQA_for05_re95_pure_ce10_step233" "${FOR05_LORA}" "${FOR05_MODEL}" "${LOG_ROOT}/merge_for05.log"
merge_one "literaryQA_for10_re90_pure_ce10_step196" "${FOR10_LORA}" "${FOR10_MODEL}" "${LOG_ROOT}/merge_for10.log"

generate_one \
  "01" "for01_re99" "pure_ce10_for01" "${FOR01_MODEL}" \
  "${DATA_ROOT}/forget01.json" "${DATA_ROOT}/retain99.json" \
  "${AGENT_DIR}/final_performance/literaryQA/for01_re99/retrieval_cache/final_graph_full_forget204_retain18857_top50_20260629/graph_path_eval_results.csv" \
  204 18857

generate_one \
  "05" "for05_re95" "pure_ce10_for05" "${FOR05_MODEL}" \
  "${DATA_ROOT}/forget05.json" "${DATA_ROOT}/retain95.json" \
  "${AGENT_DIR}/final_performance/literaryQA/for05_re95/retrieval_cache/final_graph_full_forget992_retain18069_top50_20260629/graph_path_eval_results.csv" \
  992 18069

generate_one \
  "10" "for10_re90" "pure_ce10_for10" "${FOR10_MODEL}" \
  "${DATA_ROOT}/forget10.json" "${DATA_ROOT}/retain90.json" \
  "${AGENT_DIR}/final_performance/literaryQA/for10_re90/retrieval_cache/final_graph_full_forget1970_retain17091_top50_20260629/graph_path_eval_results.csv" \
  1970 17091

run_judge_and_metrics

log "Done. Metrics: ${METRICS_DIR}/final_metrics_summary.csv"
cat "${METRICS_DIR}/final_metrics_summary.csv"
