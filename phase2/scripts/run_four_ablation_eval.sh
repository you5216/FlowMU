#!/usr/bin/env bash
set -euo pipefail

AGENT_DIR="/home/likunhao/likunhao52shared/agent_unlearning_zhx/agent"
ENV50_PY="/home/likunhao/likunhao52shared/agent_unlearning_zhx/environment/env50/bin/python"
NATIVE_CONDA_SH="/home/likunhao/miniconda3/etc/profile.d/conda.sh"
NATIVE_ENV="/home/likunhao/miniconda3/envs/nativeqwen310"

BASE_MODEL="/home/likunhao/likunhao52shared/llms/Qwen3.5-9B"
MERGED_UNLEARNED_MODEL="${AGENT_DIR}/phase2/merged_models/dual_track_forget10_ckpt2000_full_v1"
JUDGE_MODEL="/home/likunhao/likunhao52shared/llms/Qwen2.5-VL-7B-Instruct"

FORGET_PATH="/home/likunhao/likunhao52shared/datasets/TOFU/forget10.json"
RETAIN_PATH="/home/likunhao/likunhao52shared/datasets/TOFU/retain90.json"

RAG_GRAPH="${AGENT_DIR}/phase1/dual_graph_unlearning_gat/path_supervision_runs/final_retain70_memory_path/rag_unlearning_graph.pkl"
MEMORY_GRAPH="${AGENT_DIR}/phase1/dual_graph_unlearning_gat/path_supervision_runs/final_retain70_memory_path/memory_unlearning_graph.pkl"

RUN_ROOT="${AGENT_DIR}/phase2/eval_runs/ablation_four_runs"
LOG_DIR="${AGENT_DIR}/logs/ablation_four_runs"
mkdir -p "${RUN_ROOT}" "${LOG_DIR}"

MODEL_PORT=8000
JUDGE_PORT=8001
GPU_ID=0

cd "${AGENT_DIR}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

kill_vllm() {
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 8
}

wait_openai_server() {
  local port="$1"
  local name="$2"
  local log_file="$3"
  local max_wait="${4:-900}"
  local waited=0
  log "Waiting for ${name} on port ${port} ..."
  while true; do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      log "${name} is ready on port ${port}."
      return 0
    fi
    if ! pgrep -f "vllm serve" >/dev/null 2>&1; then
      log "ERROR: ${name} vLLM process exited before becoming ready. Log: ${log_file}"
      tail -n 120 "${log_file}" || true
      exit 1
    fi
    if (( waited >= max_wait )); then
      log "ERROR: Timeout waiting for ${name}. Log: ${log_file}"
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
  local gpu_memory="${5:-0.90}"

  kill_vllm
  log "Starting vLLM: served_name=${served_name}, port=${port}, model=${model_path}"
  nohup bash -lc "
    source '${NATIVE_CONDA_SH}'
    conda activate '${NATIVE_ENV}'
    CUDA_VISIBLE_DEVICES='${GPU_ID}' vllm serve '${model_path}' \
      --host 0.0.0.0 \
      --port '${port}' \
      --served-model-name '${served_name}' \
      --max-model-len 2048 \
      --gpu-memory-utilization '${gpu_memory}' \
      --enforce-eager \
      --trust-remote-code
  " > "${log_file}" 2>&1 &
  wait_openai_server "${port}" "${served_name}" "${log_file}"
}

ensure_merged_processor_configs() {
  cp "${BASE_MODEL}/preprocessor_config.json" "${MERGED_UNLEARNED_MODEL}/preprocessor_config.json"
  cp "${BASE_MODEL}/video_preprocessor_config.json" "${MERGED_UNLEARNED_MODEL}/video_preprocessor_config.json"
}

clean_outputs() {
  local out_dir="$1"
  mkdir -p "${out_dir}"
  rm -f "${out_dir}/results.csv" "${out_dir}/results.json"
  rm -rf "${out_dir}/judge_qwen25"
}

validate_results_csv() {
  local csv_path="$1"
  local expected_lines=4001
  if [[ ! -f "${csv_path}" ]]; then
    log "ERROR: missing results csv: ${csv_path}"
    exit 1
  fi
  local header
  header="$(head -n 1 "${csv_path}")"
  if [[ "${header}" != split,* ]]; then
    log "ERROR: invalid CSV header in ${csv_path}: ${header}"
    exit 1
  fi
  local lines
  lines="$(wc -l < "${csv_path}")"
  if [[ "${lines}" -ne "${expected_lines}" ]]; then
    log "ERROR: expected ${expected_lines} lines but got ${lines}: ${csv_path}"
    exit 1
  fi
  log "Validated CSV: ${csv_path} (${lines} lines)"
}

run_param_eval() {
  local exp_name="$1"
  local served_name="$2"
  local lora_path="$3"
  local out_dir="${RUN_ROOT}/${exp_name}"
  local log_file="${LOG_DIR}/${exp_name}_generate.log"
  clean_outputs "${out_dir}"

  log "Generating pure-param answers: ${exp_name}"
  VLLM_API_BASE="http://127.0.0.1:${MODEL_PORT}/v1" \
  VLLM_MODEL_NAME="${served_name}" \
  QWEN_INFER_BACKEND="vllm_api" \
  "${ENV50_PY}" -m phase2.pipeline.parametric_eval \
    --backend vllm_api \
    --lora-path "${lora_path}" \
    --forget-path "${FORGET_PATH}" \
    --retain-path "${RETAIN_PATH}" \
    --forget-limit 400 \
    --retain-limit 3600 \
    --max-new-tokens 128 \
    --temperature 0.0 \
    --output-csv "${out_dir}/results.csv" \
    --output-json "${out_dir}/results.json" \
    --no-resume > "${log_file}" 2>&1

  validate_results_csv "${out_dir}/results.csv"
}

run_graph_eval() {
  local exp_name="$1"
  local served_name="$2"
  local lora_path="$3"
  local no_rag_flag="$4"
  local no_memory_flag="$5"
  local out_dir="${RUN_ROOT}/${exp_name}"
  local log_file="${LOG_DIR}/${exp_name}_generate.log"
  clean_outputs "${out_dir}"

  log "Generating graph-context answers: ${exp_name}"
  local extra_flags=()
  if [[ "${no_rag_flag}" == "1" ]]; then
    extra_flags+=(--no-rag)
  fi
  if [[ "${no_memory_flag}" == "1" ]]; then
    extra_flags+=(--no-memory)
  fi

  VLLM_API_BASE="http://127.0.0.1:${MODEL_PORT}/v1" \
  VLLM_MODEL_NAME="${served_name}" \
  QWEN_INFER_BACKEND="vllm_api" \
  "${ENV50_PY}" -m phase2.pipeline.step6_full_pipeline_eval \
    --base-model-path "${BASE_MODEL}" \
	    --lora-path "${lora_path}" \
	    --rag-graph-path "${RAG_GRAPH}" \
	    --memory-graph-path "${MEMORY_GRAPH}" \
    --forget-path "${FORGET_PATH}" \
    --retain-path "${RETAIN_PATH}" \
    --forget-limit 400 \
    --retain-limit 3600 \
    --output-csv "${out_dir}/results.csv" \
    --output-json "${out_dir}/results.json" \
    --backend vllm_api \
    --embedding-device cpu \
    --embedding-batch-size 2 \
    --embedding-max-length 256 \
	    --max-new-tokens 128 \
	    --temperature 0.0 \
	    --rag-fallback-top-k 3 \
	    --memory-max-depth 5 \
    --memory-max-seeds 3 \
    --memory-max-hits 3 \
    --memory-edge-threshold 0.0 \
    --memory-entity-sim-floor 0.35 \
    --no-resume \
    "${extra_flags[@]}" > "${log_file}" 2>&1

  validate_results_csv "${out_dir}/results.csv"
}

run_judge() {
  local exp_name="$1"
  local out_dir="${RUN_ROOT}/${exp_name}"
  local log_file="${LOG_DIR}/${exp_name}_judge.log"

  start_vllm "${JUDGE_MODEL}" "judge" "${JUDGE_PORT}" "${LOG_DIR}/${exp_name}_judge_vllm.log" "0.90"
  log "Judging: ${exp_name}"
  "${ENV50_PY}" evaluation/llm_judge.py \
    --files "${out_dir}/results.csv" \
    --labels "${exp_name}" \
    --api-url "http://127.0.0.1:${JUDGE_PORT}/v1" \
    --model judge \
    --output-dir "${out_dir}/judge_qwen25" > "${log_file}" 2>&1
  tail -n 20 "${log_file}" || true
  kill_vllm
}

run_one_with_base_model() {
  local exp_name="$1"
  local mode="$2"
  local no_rag="$3"
  local no_memory="$4"

  start_vllm "${BASE_MODEL}" "qwen35_base" "${MODEL_PORT}" "${LOG_DIR}/${exp_name}_vllm.log" "0.90"
  if [[ "${mode}" == "param" ]]; then
    run_param_eval "${exp_name}" "qwen35_base" ""
  else
    run_graph_eval "${exp_name}" "qwen35_base" "" "${no_rag}" "${no_memory}"
  fi
  kill_vllm
  run_judge "${exp_name}"
}

run_one_with_unlearned_model() {
  local exp_name="$1"
  ensure_merged_processor_configs
  start_vllm "${MERGED_UNLEARNED_MODEL}" "dual_graph_unlearned" "${MODEL_PORT}" "${LOG_DIR}/${exp_name}_vllm.log" "0.90"
  run_param_eval "${exp_name}" "dual_graph_unlearned" "${MERGED_UNLEARNED_MODEL}"
  kill_vllm
  run_judge "${exp_name}"
}

log "Starting four ablation runs. Outputs: ${RUN_ROOT}"

run_one_with_base_model "01_param_original_qwen35" "param" "1" "1"

run_one_with_base_model "02_rag_only_original_qwen35" "graph" "0" "1"

run_one_with_base_model "03_rag_memory_original_qwen35" "graph" "0" "0"

run_one_with_unlearned_model "04_param_unlearned_merged"

log "All four ablation runs finished. Summary files:"
find "${RUN_ROOT}" -path "*/judge_qwen25/judge_summary_custom.json" -print
