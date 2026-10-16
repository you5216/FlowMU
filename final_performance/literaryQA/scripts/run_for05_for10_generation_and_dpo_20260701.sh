#!/usr/bin/env bash
set -euo pipefail

AGENT_DIR="/disk8T/lkh/python_projects/agent_unlearning_zhx/agent"
PY="/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/env52/bin/python"
VLLM="/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/52vllm/bin/vllm"
DATA_ROOT="/disk8T/lkh/python_projects/datasets/LiteraryQA/literaryqa_annotations/rag"

BASE_MODEL="${AGENT_DIR}/final_performance/literaryQA/base/lora/merged/literary_neative_model"
MODEL_PORT="${MODEL_PORT:-18080}"
GPU_ID="${GPU_ID:-0}"
GPU_WAIT_MAX_USED_MB="${GPU_WAIT_MAX_USED_MB:-1500}"

RUN_ROOT="${AGENT_DIR}/final_performance/literaryQA/nightly_for05_for10_gen_dpo_20260701"
STATUS_LOG="${RUN_ROOT}/status.log"
mkdir -p "${RUN_ROOT}"

export HOME=/disk8T/lkh/runtime_home
export XDG_CACHE_HOME=/disk8T/lkh/runtime_cache
export HF_HOME=/disk8T/lkh/hf_cache
export TRANSFORMERS_CACHE=/disk8T/lkh/hf_cache
export TORCH_HOME=/disk8T/lkh/torch_cache
export TMPDIR=/disk8T/lkh/tmp

cd "${AGENT_DIR}"

log() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "${msg}" | tee -a "${STATUS_LOG}"
}

trap 'rc=$?; log "ERROR line=${LINENO} rc=${rc} cmd=${BASH_COMMAND}"; exit ${rc}' ERR

mark_done() {
  local name="$1"
  mkdir -p "${RUN_ROOT}/done"
  date '+%Y-%m-%d %H:%M:%S' > "${RUN_ROOT}/done/${name}.done"
}

is_done() {
  [[ -f "${RUN_ROOT}/done/$1.done" ]]
}

gpu_used_mb() {
  local used
  if ! used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU_ID}" 2>/dev/null | tr -dc '0-9')"; then
    return 1
  fi
  [[ -n "${used}" ]] || return 1
  printf '%s\n' "${used}"
}

wait_gpu_free() {
  local label="$1"
  local waited=0
  while true; do
    local used
    if ! used="$(gpu_used_mb)"; then
      log "Waiting GPU for ${label}: nvidia-smi unavailable"
      sleep 60
      waited=$((waited + 60))
      continue
    fi
    if [[ -n "${used}" && "${used}" -le "${GPU_WAIT_MAX_USED_MB}" ]]; then
      log "GPU ready for ${label}: used=${used}MB"
      return 0
    fi
    log "Waiting GPU for ${label}: used=${used:-unknown}MB"
    sleep 60
    waited=$((waited + 60))
  done
}

kill_port_vllm() {
  local port="$1"
  local pids
  pids="$(pgrep -af "vllm serve .*--port ${port}" | awk '{print $1}' || true)"
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
  local max_wait="${4:-900}"
  local waited=0
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
  wait_gpu_free "vLLM ${served_name}"
  kill_port_vllm "${port}"
  log "Starting vLLM: ${served_name}, port=${port}, model=${model_path}"
  log "vLLM log: ${log_file}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${VLLM}" serve "${model_path}" \
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
  log "vLLM PID for ${served_name}: $(cat "${log_file}.pid")"
  wait_server "${port}" "${served_name}" "${log_file}"
}

validate_generation() {
  local csv_path="$1"
  local expected="$2"
  "${PY}" - "${csv_path}" "${expected}" <<'PY'
import csv, json, re, sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
if not path.exists():
    raise SystemExit(f"missing results csv: {path}")
with path.open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
empty = sum(1 for r in rows if not (r.get("prediction") or "").strip())
errors = sum(1 for r in rows if "[EVAL_ERROR]" in (r.get("prediction") or ""))
think = sum(1 for r in rows if "<think" in (r.get("prediction") or "").lower() or "thinking process" in (r.get("prediction") or "").lower())
if len(rows) != expected:
    raise SystemExit(f"row count mismatch: got={len(rows)} expected={expected} path={path}")
if empty or errors or think:
    raise SystemExit(f"bad generation: rows={len(rows)} empty={empty} errors={errors} think={think} path={path}")
print(json.dumps({"path": str(path), "rows": len(rows), "empty": empty, "eval_error": errors, "think_like": think}, ensure_ascii=False))
PY
}

prepare_generation_retry() {
  local csv_path="$1"
  "${PY}" - "${csv_path}" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = []
with path.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames or []
    for row in reader:
        prediction = (row.get("prediction") or "").strip()
        if not prediction:
            row["prediction"] = "[EVAL_ERROR] empty_prediction_retry"
        rows.append(row)

tmp_path = path.with_suffix(path.suffix + ".retry.tmp")
with tmp_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})
tmp_path.replace(path)
print(f"prepared retry csv: {path}")
PY
}

run_full_generation() {
  local name="$1"
  local split_dir="$2"
  local model_path="$3"
  local served_name="$4"
  local forget_path="$5"
  local retain_path="$6"
  local forget_limit="$7"
  local retain_limit="$8"
  local retrieval_csv="$9"
  local rag_graph="${10}"
  local memory_graph="${11}"
  local eval_dir="${12}"
  local expected_rows="${13}"

  if [[ -f "${RUN_ROOT}/done/${name}.done" ]]; then
    log "SKIP done: ${name}"
    return 0
  fi

  mkdir -p "${eval_dir}/logs"
  local resume_args=()
  local generate_log="${eval_dir}/logs/generate.log"
  if [[ -f "${eval_dir}/results.csv" ]]; then
    log "FOUND existing generation CSV for ${name}; validating before rerun"
    if validate_generation "${eval_dir}/results.csv" "${expected_rows}" 2>&1 | tee -a "${STATUS_LOG}"; then
      mark_done "${name}"
      log "DONE generation ${name}: ${eval_dir}/results.csv"
      return 0
    fi
    log "Existing CSV for ${name} needs retry; marking empty predictions as retryable"
    prepare_generation_retry "${eval_dir}/results.csv" | tee -a "${STATUS_LOG}"
    generate_log="${eval_dir}/logs/generate_retry_$(date '+%Y%m%d_%H%M%S').log"
  else
    rm -f "${eval_dir}/results.json"
    resume_args+=(--no-resume)
  fi

  log "START generation ${name}: expected_rows=${expected_rows}"
  start_vllm "${model_path}" "${served_name}" "${MODEL_PORT}" "${eval_dir}/logs/vllm_generate.log" "0.86"

  VLLM_API_BASE="http://127.0.0.1:${MODEL_PORT}/v1" \
  VLLM_MODEL_NAME="${served_name}" \
  QWEN_INFER_BACKEND="vllm_api" \
  QWEN_EVAL_NUM_WORKERS="${QWEN_EVAL_NUM_WORKERS:-24}" \
  QWEN_TOP_K=1 \
  "${PY}" -m phase2.pipeline.step6_full_pipeline_eval \
    --base-model-path "${model_path}" \
    --lora-path "" \
    --rag-graph-path "${rag_graph}" \
    --memory-graph-path "${memory_graph}" \
    --retrieval-results-csv "${retrieval_csv}" \
    --forget-path "${forget_path}" \
    --retain-path "${retain_path}" \
    --forget-limit "${forget_limit}" \
    --retain-limit "${retain_limit}" \
    --output-csv "${eval_dir}/results.csv" \
    --output-json "${eval_dir}/results.json" \
    --backend vllm_api \
    --max-new-tokens 128 \
    --temperature 0.0 \
    --rag-fallback-top-k 3 \
    --memory-max-hits 3 \
    "${resume_args[@]}" \
    > "${generate_log}" 2>&1

  kill_port_vllm "${MODEL_PORT}"
  validate_generation "${eval_dir}/results.csv" "${expected_rows}" | tee -a "${STATUS_LOG}"
  mark_done "${name}"
  log "DONE generation ${name}: ${eval_dir}/results.csv"
}

run_dpo() {
  local name="$1"
  local split_dir="$2"
  local localization_pkl="$3"
  local forget_qa="$4"
  local retain_qa="$5"
  local retain_sample_size="$6"
  local output_dir="$7"

  if [[ -f "${RUN_ROOT}/done/${name}.done" ]]; then
    log "SKIP done: ${name}"
    return 0
  fi

  mkdir -p "${output_dir}/logs" "${output_dir}/reference_margin_cache"
  log "START DPO ${name}: output=${output_dir}"
  wait_gpu_free "DPO ${name}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PY}" -m phase2.pipeline.masked_dpo_optimization \
    --base-model-path "${BASE_MODEL}" \
    --output-dir "${output_dir}" \
    --localization-pkl "${localization_pkl}" \
    --forget-qa-path "${forget_qa}" \
    --retain-qa-path "${retain_qa}" \
    --new-lora-r 16 \
    --new-lora-alpha 32 \
    --new-lora-dropout 0.05 \
    --new-lora-target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --beta 0.2 \
    --learning-rate 8e-5 \
    --num-train-epochs 5 \
    --per-device-train-batch-size 2 \
    --gradient-accumulation-steps 4 \
    --max-length 512 \
    --warmup-steps 10 \
    --max-grad-norm 0.1 \
    --logging-steps 10 \
    --retain-sample-size "${retain_sample_size}" \
    --forget-sample-size 0 \
    --retain-loss-weight 1.0 \
    --forget-loss-weight 1.0 \
    --forget-answer-suppression-weight 0.8 \
    --forget-key-token-suppression-weight 0.25 \
    --reference-margin-cache-dir "${output_dir}/reference_margin_cache" \
    --save-steps 300 \
    > "${output_dir}/logs/train.log" 2>&1

  if [[ ! -f "${output_dir}/adapter_model.safetensors" ]]; then
    log "ERROR: missing final DPO adapter: ${output_dir}/adapter_model.safetensors"
    exit 1
  fi
  local ckpt_count
  ckpt_count="$(find "${output_dir}" -maxdepth 1 -type d -name 'checkpoint-step*' | wc -l)"
  log "DONE DPO ${name}: checkpoints=${ckpt_count}, output=${output_dir}"
  mark_done "${name}"
}

FOR05_DIR="${AGENT_DIR}/final_performance/literaryQA/for05_re95"
FOR10_DIR="${AGENT_DIR}/final_performance/literaryQA/for10_re90"

run_full_generation \
  "for05_ga_ce75_full_generation" \
  "${FOR05_DIR}" \
  "${FOR05_DIR}/merged_model/ga/literaryQA_for05_re95_ga_pure_literarynative_top50rr09_ce75_20260701" \
  "for05_ga_ce75" \
  "${DATA_ROOT}/forget05.json" \
  "${DATA_ROOT}/retain95.json" \
  992 \
  18069 \
  "${FOR05_DIR}/retrieval_cache/final_graph_full_forget992_retain18069_top50_20260629/graph_path_eval_results.csv" \
  "${FOR05_DIR}/rag_memory_unlearning_graph/rag_unlearning_graph.pkl" \
  "${FOR05_DIR}/rag_memory_unlearning_graph/memory_unlearning_graph.pkl" \
  "${FOR05_DIR}/full_pipeline_eval/ga_ce75_full_forget992_retain18069_graphhybrid_genonly_20260701" \
  19061

run_full_generation \
  "for05_rmu_ce65_full_generation" \
  "${FOR05_DIR}" \
  "${FOR05_DIR}/merged_model/rmu/literaryQA_for05_re95_rmu_nokl_ce65_20260701" \
  "for05_rmu_ce65" \
  "${DATA_ROOT}/forget05.json" \
  "${DATA_ROOT}/retain95.json" \
  992 \
  18069 \
  "${FOR05_DIR}/retrieval_cache/final_graph_full_forget992_retain18069_top50_20260629/graph_path_eval_results.csv" \
  "${FOR05_DIR}/rag_memory_unlearning_graph/rag_unlearning_graph.pkl" \
  "${FOR05_DIR}/rag_memory_unlearning_graph/memory_unlearning_graph.pkl" \
  "${FOR05_DIR}/full_pipeline_eval/rmu_ce65_full_forget992_retain18069_graphhybrid_genonly_20260701" \
  19061

run_full_generation \
  "for10_ga_ce12_full_generation" \
  "${FOR10_DIR}" \
  "${FOR10_DIR}/merged_model/ga/literaryQA_for10_re90_ga_ce12_20260701" \
  "for10_ga_ce12" \
  "${DATA_ROOT}/forget10.json" \
  "${DATA_ROOT}/retain90.json" \
  1970 \
  17091 \
  "${FOR10_DIR}/retrieval_cache/final_graph_full_forget1970_retain17091_top50_20260629/graph_path_eval_results.csv" \
  "${FOR10_DIR}/rag_memory_unlearning_graph/rag_unlearning_graph.pkl" \
  "${FOR10_DIR}/rag_memory_unlearning_graph/memory_unlearning_graph.pkl" \
  "${FOR10_DIR}/full_pipeline_eval/ga_ce12_full_forget1970_retain17091_graphhybrid_genonly_20260701" \
  19061

run_full_generation \
  "for10_rmu_ce13_full_generation" \
  "${FOR10_DIR}" \
  "${FOR10_DIR}/merged_model/rmu/literaryQA_for10_re90_rmu_nokl_ce13_20260701" \
  "for10_rmu_ce13" \
  "${DATA_ROOT}/forget10.json" \
  "${DATA_ROOT}/retain90.json" \
  1970 \
  17091 \
  "${FOR10_DIR}/retrieval_cache/final_graph_full_forget1970_retain17091_top50_20260629/graph_path_eval_results.csv" \
  "${FOR10_DIR}/rag_memory_unlearning_graph/rag_unlearning_graph.pkl" \
  "${FOR10_DIR}/rag_memory_unlearning_graph/memory_unlearning_graph.pkl" \
  "${FOR10_DIR}/full_pipeline_eval/rmu_ce13_full_forget1970_retain17091_graphhybrid_genonly_20260701" \
  19061

run_dpo \
  "for05_dpo_5epoch_save300" \
  "${FOR05_DIR}" \
  "${FOR05_DIR}/localization/literary_neative_model_cached_combined_top50_rr09_20260701/localized_neurons.pkl" \
  "${FOR05_DIR}/phase2_graph_qa_final_rag_memory_graph_20260629/graph_forget_qa.jsonl" \
  "${FOR05_DIR}/phase2_graph_qa_final_rag_memory_graph_20260629/graph_retain_qa.jsonl" \
  2217 \
  "${FOR05_DIR}/lora/dpo_enhanced_strong_literarynative_graphqa2217_eqretain_b02_lr8e5_as080_key025_e5_save300_20260701"

run_dpo \
  "for10_dpo_5epoch_save300" \
  "${FOR10_DIR}" \
  "${FOR10_DIR}/localization/literary_neative_model_cached_combined_top50_rr09_20260701/localized_neurons.pkl" \
  "${FOR10_DIR}/phase2_graph_qa_trial_01_strongqa_memoryfix/graph_forget_qa.jsonl" \
  "${FOR10_DIR}/phase2_graph_qa_trial_01_strongqa_memoryfix/graph_retain_qa.jsonl" \
  2230 \
  "${FOR10_DIR}/lora/dpo_enhanced_strong_literarynative_graphqa2230_eqretain_b02_lr8e5_as080_key025_e5_save300_20260701"

log "ALL TASKS COMPLETE"
