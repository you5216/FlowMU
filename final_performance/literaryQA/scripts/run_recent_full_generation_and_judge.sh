#!/usr/bin/env bash
set -euo pipefail

AGENT_DIR="/disk8T/lkh/python_projects/agent_unlearning_zhx/agent"
PY="/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/env52/bin/python"
VLLM="/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/52vllm/bin/vllm"
JUDGE_MODEL="/disk8T/lkh/python_projects/llms/Qwen2.5-VL-7B-Instruct"
GEN_SCRIPT="${AGENT_DIR}/final_performance/literaryQA/scripts/run_full_generation_once.sh"
OUT_ROOT="${AGENT_DIR}/final_performance/literaryQA/full_pipeline_eval_recent_20260702"
LOG_DIR="${OUT_ROOT}/logs"
JUDGE_DIR="${OUT_ROOT}/judge_qwen25vl"
METRICS_DIR="${OUT_ROOT}/metrics"
JUDGE_PORT="${JUDGE_PORT:-18181}"
JUDGE_WORKERS="${JUDGE_WORKERS:-24}"
MODEL_PORT="${MODEL_PORT:-18180}"

mkdir -p "${OUT_ROOT}" "${LOG_DIR}" "${JUDGE_DIR}" "${METRICS_DIR}"

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

start_judge_vllm() {
  local log_file="${LOG_DIR}/vllm_judge.log"
  kill_port_vllm "${JUDGE_PORT}"
  log "Starting judge vLLM on port ${JUDGE_PORT}"
  CUDA_VISIBLE_DEVICES=0 "${VLLM}" serve "${JUDGE_MODEL}" \
    --host 0.0.0.0 \
    --port "${JUDGE_PORT}" \
    --served-model-name judge \
    --max-model-len 1024 \
    --gpu-memory-utilization 0.86 \
    --enforce-eager \
    --trust-remote-code \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --mm-processor-cache-gb 0 \
    > "${log_file}" 2>&1 &
  echo $! > "${log_file}.pid"
  wait_server "${JUDGE_PORT}" "judge" "${log_file}" 900
}

validate_csv() {
  local csv_path="$1"
  local expect_forget="$2"
  local expect_retain="$3"
  "${PY}" - "$csv_path" "$expect_forget" "$expect_retain" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
expect_forget = int(sys.argv[2])
expect_retain = int(sys.argv[3])
if not path.exists():
    print(f"MISSING {path}")
    sys.exit(1)

with path.open("r", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

def split_type(row):
    value = (row.get("split_type") or row.get("split") or "").strip().lower()
    return value if value in {"forget", "retain"} else "retain"

forget = sum(1 for row in rows if split_type(row) == "forget")
retain = sum(1 for row in rows if split_type(row) == "retain")
empty = sum(1 for row in rows if not (row.get("prediction") or "").strip())
errors = sum(1 for row in rows if "[ERROR" in (row.get("prediction") or ""))
think = sum(
    1
    for row in rows
    if "<think" in (row.get("prediction") or "").lower()
    or "thinking process" in (row.get("prediction") or "").lower()
)
ok = (
    len(rows) == expect_forget + expect_retain
    and forget == expect_forget
    and retain == expect_retain
    and empty == 0
    and errors == 0
    and think == 0
)
print(
    f"rows={len(rows)} forget={forget}/{expect_forget} retain={retain}/{expect_retain} "
    f"empty={empty} errors={errors} think={think}"
)
sys.exit(0 if ok else 1)
PY
}

wait_for_existing_generation() {
  local eval_dir="$1"
  while pgrep -af "run_full_generation_once.sh .*${eval_dir}|step6_full_pipeline_eval .*${eval_dir}" >/dev/null 2>&1; do
    log "Existing generation still running for ${eval_dir}; waiting 300s"
    sleep 300
  done
}

run_missing_generation() {
  local label="$1"
  local model_path="$2"
  local split="$3"
  local forget_path="$4"
  local retain_path="$5"
  local retrieval_csv="$6"
  local eval_dir="$7"
  local expect_forget="$8"
  local expect_retain="$9"

  mkdir -p "${eval_dir}/logs"
  wait_for_existing_generation "${eval_dir}"
  if validate_csv "${eval_dir}/results.csv" "${expect_forget}" "${expect_retain}"; then
    log "Generation already valid: ${eval_dir}/results.csv"
    return 0
  fi

  log "Running missing generation: ${label} ${split}"
  MODEL_PORT="${MODEL_PORT}" QWEN_EVAL_NUM_WORKERS="${QWEN_EVAL_NUM_WORKERS:-24}" \
    "${GEN_SCRIPT}" "${label}" "${model_path}" "${split}" "${forget_path}" "${retain_path}" "${retrieval_csv}" "${eval_dir}" \
    > "${eval_dir}/logs/orchestrator_generate.stdout" 2>&1

  validate_csv "${eval_dir}/results.csv" "${expect_forget}" "${expect_retain}"
  log "Generation valid after run: ${eval_dir}/results.csv"
}

log "Validating existing 20260701/20260702 full generation CSVs"

FOR01_GA="${AGENT_DIR}/final_performance/literaryQA/for01_re99/full_pipeline_eval/ga_pure_literarynative_top50rr06_ce60_full_forget204_retain18857_graphhybrid_len1024_20260701/results.csv"
FOR01_RMU="${AGENT_DIR}/final_performance/literaryQA/for01_re99/full_pipeline_eval/rmu_nokl_ce80_full_forget204_retain18857_graphhybrid_genonly_20260702/results.csv"
FOR01_DPO="${AGENT_DIR}/final_performance/literaryQA/for01_re99/full_pipeline_eval/dpo_strong_resume_step1101_full_forget204_retain18857_graphhybrid_len1024_20260701/results.csv"
FOR05_GA="${AGENT_DIR}/final_performance/literaryQA/for05_re95/full_pipeline_eval/ga_ce75_full_forget992_retain18069_graphhybrid_genonly_20260701/results.csv"
FOR05_RMU="${AGENT_DIR}/final_performance/literaryQA/for05_re95/full_pipeline_eval/rmu_ce65_full_forget992_retain18069_graphhybrid_genonly_20260701/results.csv"
FOR05_DPO="${AGENT_DIR}/final_performance/literaryQA/for05_re95/full_pipeline_eval/dpo_step5100_full_forget992_retain18069_graphhybrid_genonly_20260702/results.csv"
FOR10_GA="${AGENT_DIR}/final_performance/literaryQA/for10_re90/full_pipeline_eval/ga_ce12_full_forget1970_retain17091_graphhybrid_genonly_20260701/results.csv"
FOR10_RMU="${AGENT_DIR}/final_performance/literaryQA/for10_re90/full_pipeline_eval/rmu_ce13_full_forget1970_retain17091_graphhybrid_genonly_20260701/results.csv"
FOR10_DPO="${AGENT_DIR}/final_performance/literaryQA/for10_re90/full_pipeline_eval/dpo_step1395_full_forget1970_retain17091_graphhybrid_genonly_20260702/results.csv"

validate_csv "${FOR01_GA}" 204 18857
validate_csv "${FOR01_DPO}" 204 18857
validate_csv "${FOR05_GA}" 992 18069
validate_csv "${FOR05_RMU}" 992 18069
validate_csv "${FOR10_GA}" 1970 17091
validate_csv "${FOR10_RMU}" 1970 17091

run_missing_generation \
  rmu_ce80 \
  "${AGENT_DIR}/final_performance/literaryQA/for01_re99/merged_model/rmu/literaryQA_for01_re99_rmu_nokl_ce80_20260630" \
  for01_re99 \
  /disk8T/lkh/python_projects/datasets/LiteraryQA/literaryqa_annotations/rag/forget01.json \
  /disk8T/lkh/python_projects/datasets/LiteraryQA/literaryqa_annotations/rag/retain99.json \
  "${AGENT_DIR}/final_performance/literaryQA/for01_re99/retrieval_cache/final_graph_full_forget204_retain18857_top50_20260629/graph_path_eval_results.csv" \
  "${AGENT_DIR}/final_performance/literaryQA/for01_re99/full_pipeline_eval/rmu_nokl_ce80_full_forget204_retain18857_graphhybrid_genonly_20260702" \
  204 18857

run_missing_generation \
  dpo_step5100 \
  "${AGENT_DIR}/final_performance/literaryQA/for05_re95/merged_model/dpo/literaryQA_for05_re95_dpo_step5100_20260702" \
  for05_re95 \
  /disk8T/lkh/python_projects/datasets/LiteraryQA/literaryqa_annotations/rag/forget05.json \
  /disk8T/lkh/python_projects/datasets/LiteraryQA/literaryqa_annotations/rag/retain95.json \
  "${AGENT_DIR}/final_performance/literaryQA/for05_re95/retrieval_cache/final_graph_full_forget992_retain18069_top50_20260629/graph_path_eval_results.csv" \
  "${AGENT_DIR}/final_performance/literaryQA/for05_re95/full_pipeline_eval/dpo_step5100_full_forget992_retain18069_graphhybrid_genonly_20260702" \
  992 18069

run_missing_generation \
  dpo_step1395 \
  "${AGENT_DIR}/final_performance/literaryQA/for10_re90/merged_model/dpo/literaryQA_for10_re90_dpo_step1395_20260702" \
  for10_re90 \
  /disk8T/lkh/python_projects/datasets/LiteraryQA/literaryqa_annotations/rag/forget10.json \
  /disk8T/lkh/python_projects/datasets/LiteraryQA/literaryqa_annotations/rag/retain90.json \
  "${AGENT_DIR}/final_performance/literaryQA/for10_re90/retrieval_cache/final_graph_full_forget1970_retain17091_top50_20260629/graph_path_eval_results.csv" \
  "${AGENT_DIR}/final_performance/literaryQA/for10_re90/full_pipeline_eval/dpo_step1395_full_forget1970_retain17091_graphhybrid_genonly_20260702" \
  1970 17091

log "All 9 generation CSVs are valid. Starting one unified judge run."
start_judge_vllm

JUDGE_NUM_WORKERS="${JUDGE_WORKERS}" "${PY}" evaluation/llm_judge.py \
  --files \
    "${FOR01_GA}" "${FOR01_RMU}" "${FOR01_DPO}" \
    "${FOR05_GA}" "${FOR05_RMU}" "${FOR05_DPO}" \
    "${FOR10_GA}" "${FOR10_RMU}" "${FOR10_DPO}" \
  --labels \
    for01_ga_ce60 for01_rmu_ce80 for01_dpo_step1101 \
    for05_ga_ce75 for05_rmu_ce65 for05_dpo_step5100 \
    for10_ga_ce12 for10_rmu_ce13 for10_dpo_step1395 \
  --api-url "http://127.0.0.1:${JUDGE_PORT}/v1" \
  --model judge \
  --output-dir "${JUDGE_DIR}" \
  --num-workers "${JUDGE_WORKERS}" \
  > "${LOG_DIR}/judge_all.log" 2>&1

kill_port_vllm "${JUDGE_PORT}"

"${PY}" evaluation/final_metrics.py \
  --files \
    "${JUDGE_DIR}/judge_for01_ga_ce60.csv" "${JUDGE_DIR}/judge_for01_rmu_ce80.csv" "${JUDGE_DIR}/judge_for01_dpo_step1101.csv" \
    "${JUDGE_DIR}/judge_for05_ga_ce75.csv" "${JUDGE_DIR}/judge_for05_rmu_ce65.csv" "${JUDGE_DIR}/judge_for05_dpo_step5100.csv" \
    "${JUDGE_DIR}/judge_for10_ga_ce12.csv" "${JUDGE_DIR}/judge_for10_rmu_ce13.csv" "${JUDGE_DIR}/judge_for10_dpo_step1395.csv" \
  --labels \
    for01_ga_ce60 for01_rmu_ce80 for01_dpo_step1101 \
    for05_ga_ce75 for05_rmu_ce65 for05_dpo_step5100 \
    for10_ga_ce12 for10_rmu_ce13 for10_dpo_step1395 \
  --output-dir "${METRICS_DIR}" \
  > "${LOG_DIR}/final_metrics.log" 2>&1

log "Finished. Judge summary: ${JUDGE_DIR}/judge_summary_custom.json"
log "Finished. Final metrics: ${METRICS_DIR}/final_metrics_summary.csv"
