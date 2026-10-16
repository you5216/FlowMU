#!/usr/bin/env bash
set -euo pipefail

ROOT="/disk8T/lkh/python_projects/agent_unlearning_zhx/agent"
BASELINE_AGENT="/disk8T/lkh/python_projects/baseline_xzy(2)/agent"
PY="/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/env52/bin/python"
VLLM="/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/52vllm/bin/vllm"
RUN_DIR="${ROOT}/final_performance/tofu/small_exp/robustness/rewrite_induce_attack_plot/real_tofu_for05"
FINAL_DIR="${ROOT}/final_performance/final/tofu_for05_rewrite_induce_attack"
LOG_DIR="${RUN_DIR}/logs"
MANIFEST_DIR="${RUN_DIR}/attack_manifests"
GEN_DIR="${RUN_DIR}/generation"
JUDGE_DIR="${RUN_DIR}/judge_qwen25vl"
SUMMARY_DIR="${RUN_DIR}/summary"

GPU_ID="${GPU_ID:-0}"
MODEL_PORT="${MODEL_PORT:-8002}"
JUDGE_PORT="${JUDGE_PORT:-18081}"
MODEL_WORKERS="${MODEL_WORKERS:-8}"
JUDGE_WORKERS="${JUDGE_WORKERS:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
SMOKE_LIMIT="${SMOKE_LIMIT:-0}"
QUERY_EMBED_DEVICE="${QUERY_EMBED_DEVICE:-cuda}"
QUERY_EMBED_BATCH_SIZE="${QUERY_EMBED_BATCH_SIZE:-16}"

FORGET_PATH="/disk8T/lkh/python_projects/datasets/TOFU/re_split/forget05.json"
EMPTY_RETAIN="${RUN_DIR}/empty_retain.json"
ENCODER_MODEL="/disk8T/lkh/python_projects/llms/Qwen3.5-9B"
JUDGE_MODEL="/disk8T/lkh/python_projects/llms/Qwen2.5-VL-7B-Instruct"

VANILLA_MODEL="${ROOT}/merge_model/native_llm"
PURE_MODEL="/disk8T/lkh/python_projects/baseline_xzy/experiments/merged/pure_for05"
SBU_MODEL="/disk8T/lkh/python_projects/baseline_xzy/experiments/merged/sbu_for05"
FLOWMU_MODEL="${ROOT}/final_performance/tofu/for05_re95/phase2_merged/dpo_fullretain_lambda03top100_beta02_lr2e5_e02_20260622"
SFNL_MODEL="${ROOT}/merge_model/native_llm"

VANILLA_ASSETS="/disk8T/lkh/python_projects/baseline_xzy(2)/experiments/naive_faiss/vanilla_for05"
PURE_ASSETS="/disk8T/lkh/python_projects/baseline_xzy(2)/experiments/naive_faiss/pure_for05"
SBU_ASSETS="/disk8T/lkh/python_projects/baseline_xzy(2)/experiments/sbu_phase_a/sbu_for05"
SFNL_ASSETS="/disk8T/lkh/python_projects/baseline_xzy(2)/experiments/naive_faiss/sfnl_for05"
FLOWMU_RAG_GRAPH="${ROOT}/final_performance/tofu/for05_re95/phase1_gat_retrieval_oracle_lambda03_scorepos/rag_unlearning_graph.pkl"
FLOWMU_MEMORY_GRAPH="${ROOT}/final_performance/tofu/for05_re95/phase1_gat_retrieval_oracle_lambda03_scorepos/memory_unlearning_graph.pkl"

mkdir -p "${RUN_DIR}" "${FINAL_DIR}" "${LOG_DIR}" "${MANIFEST_DIR}" "${GEN_DIR}" "${JUDGE_DIR}" "${SUMMARY_DIR}"
printf '[]\n' > "${EMPTY_RETAIN}"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_DIR}/run.log"
}

kill_port() {
  local port="$1"
  local pids pid
  pids="$(fuser -n tcp "${port}" 2>/dev/null | tr ' ' '\n' | sed '/^$/d' | sort -u || true)"
  for pid in ${pids}; do
    log "stopping pid=${pid} on port=${port}"
    pkill -TERM -P "${pid}" 2>/dev/null || true
    kill -TERM "${pid}" 2>/dev/null || true
  done
  sleep 3
  pids="$(fuser -n tcp "${port}" 2>/dev/null | tr ' ' '\n' | sed '/^$/d' | sort -u || true)"
  for pid in ${pids}; do
    log "force stopping pid=${pid} on port=${port}"
    pkill -KILL -P "${pid}" 2>/dev/null || true
    kill -KILL "${pid}" 2>/dev/null || true
  done
}

wait_openai() {
  local port="$1" log_file="$2" max_wait="${3:-1200}" waited=0
  while true; do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      log "vLLM ready port=${port} waited=${waited}s"
      return 0
    fi
    if ! pgrep -f "[v]llm serve" >/dev/null 2>&1; then
      log "vLLM exited while waiting on port=${port}; tail log:"
      tail -n 80 "${log_file}" || true
      return 1
    fi
    if (( waited >= max_wait )); then
      log "timeout waiting vLLM port=${port}; tail log:"
      tail -n 80 "${log_file}" || true
      return 1
    fi
    if (( waited > 0 && waited % 30 == 0 )); then
      log "waiting vLLM port=${port} ... ${waited}s"
    fi
    sleep 5
    waited=$((waited + 5))
  done
}

start_model_vllm() {
  local model_path="$1" served_name="$2" port="$3" log_file="$4"
  kill_port "${port}"
  log "start model vLLM served=${served_name} model=${model_path} port=${port}"
  nohup bash -lc "
    CUDA_VISIBLE_DEVICES='${GPU_ID}' VLLM_USE_V1=0 '${VLLM}' serve '${model_path}' \
      --host 0.0.0.0 --port '${port}' \
      --served-model-name '${served_name}' \
      --max-model-len 2048 \
      --max-num-seqs '${VLLM_MAX_NUM_SEQS}' \
      --gpu-memory-utilization 0.86 \
      --enforce-eager --trust-remote-code
  " > "${log_file}" 2>&1 &
  wait_openai "${port}" "${log_file}"
}

start_judge_vllm() {
  local port="$1" log_file="$2"
  kill_port "${port}"
  log "start judge vLLM port=${port}"
  nohup bash -lc "
    CUDA_VISIBLE_DEVICES='${GPU_ID}' VLLM_USE_V1=0 '${VLLM}' serve '${JUDGE_MODEL}' \
      --host 0.0.0.0 --port '${port}' \
      --served-model-name judge \
      --max-model-len 2048 \
      --max-num-seqs '${VLLM_MAX_NUM_SEQS}' \
      --gpu-memory-utilization 0.86 \
      --enforce-eager --trust-remote-code \
      --limit-mm-per-prompt '{\"image\":0,\"video\":0}' \
      --mm-processor-cache-gb 0
  " > "${log_file}" 2>&1 &
  wait_openai "${port}" "${log_file}"
}

generate_manifests() {
  local extra=()
  if [[ "${SMOKE_LIMIT}" != "0" ]]; then
    extra=(--limit "${SMOKE_LIMIT}")
  fi
  log "generate rewrite/induce manifests"
  cd "${ROOT}"
  "${PY}" evaluation/robustness/build_attack_variants.py \
    --input "${FORGET_PATH}" \
    --output "${MANIFEST_DIR}/tofu_for05_rewrite.jsonl" \
    --mode rewrite \
    "${extra[@]}"
  "${PY}" evaluation/robustness/build_attack_variants.py \
    --input "${FORGET_PATH}" \
    --output "${MANIFEST_DIR}/tofu_for05_induce.jsonl" \
    --mode induce \
    "${extra[@]}"
}

build_query_cache() {
  log "build GPU query embedding cache device=${QUERY_EMBED_DEVICE}"
  cd "${ROOT}"
  "${PY}" evaluation/robustness/build_query_embedding_cache.py \
    --inputs "${MANIFEST_DIR}/tofu_for05_rewrite.jsonl" "${MANIFEST_DIR}/tofu_for05_induce.jsonl" \
    --output "${RUN_DIR}/query_embedding_cache_qwen35.npz" \
    --model-path "${ENCODER_MODEL}" \
    --device "${QUERY_EMBED_DEVICE}" \
    --batch-size "${QUERY_EMBED_BATCH_SIZE}" \
    --max-length 256 \
    --checkpoint-every 200 \
    > "${LOG_DIR}/query_embedding_cache.log" 2>&1
}

run_naive_method() {
  local method="$1" model_path="$2" assets_dir="$3" attack="$4" manifest="$5"
  local served="robust_${method,,}"
  local out_dir="${GEN_DIR}/${method}_${attack}"
  mkdir -p "${out_dir}"
  if [[ -f "${out_dir}/results.csv" ]]; then
    local have
    have="$("${PY}" - <<PY
import csv
print(sum(1 for _ in csv.DictReader(open('${out_dir}/results.csv', encoding='utf-8'))))
PY
)"
    local need
    need="$(wc -l < "${manifest}")"
    if [[ "${have}" -ge "${need}" ]]; then
      if grep -q '\[EVAL_ERROR\]' "${out_dir}/results.csv"; then
        log "regenerate ${method}/${attack}: existing results contain EVAL_ERROR"
        rm -f "${out_dir}/results.csv" "${out_dir}/results.json"
      else
        log "skip generation ${method}/${attack}: results complete ${have}/${need}"
        return 0
      fi
    fi
  fi
  start_model_vllm "${model_path}" "${served}" "${MODEL_PORT}" "${LOG_DIR}/${method}_${attack}_vllm.log"
  cd "${BASELINE_AGENT}"
  VLLM_API_BASE="http://127.0.0.1:${MODEL_PORT}/v1" \
  VLLM_MODEL_NAME="${served}" \
  QWEN_INFER_BACKEND="vllm_api" \
  VLLM_API_TIMEOUT="3600" \
  "${PY}" -m phase2.pipeline.step6_naive_faiss_eval \
    --assets-dir "${assets_dir}" \
    --base-model-path "${model_path}" \
    --lora-path "${model_path}" \
    --encoder-model-path "${ENCODER_MODEL}" \
    --forget-path "${manifest}" \
    --retain-path "${EMPTY_RETAIN}" \
    --forget-limit 0 \
    --retain-limit 0 \
    --output-csv "${out_dir}/results.csv" \
    --output-json "${out_dir}/results.json" \
    --backend vllm_api \
    --num-workers "${MODEL_WORKERS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --rag-top-k 8 \
    --memory-top-k 4 \
    --embedding-device cpu \
    --embedding-batch-size 8 \
    --query-embedding-cache "${RUN_DIR}/query_embedding_cache_qwen35.npz" \
    --require-query-cache \
    --progress-log-every 100 \
    --checkpoint-every 100 \
    > "${LOG_DIR}/${method}_${attack}_step6.log" 2>&1
  kill_port "${MODEL_PORT}"
}

run_sfnl_method() {
  local attack="$1" manifest="$2"
  local method="SF-NL"
  local served="robust_sfnl"
  local out_dir="${GEN_DIR}/${method}_${attack}"
  mkdir -p "${out_dir}"
  if [[ -f "${out_dir}/results.csv" ]]; then
    local have
    have="$("${PY}" - <<PY
import csv
print(sum(1 for _ in csv.DictReader(open('${out_dir}/results.csv', encoding='utf-8'))))
PY
)"
    local need
    need="$(wc -l < "${manifest}")"
    if [[ "${have}" -ge "${need}" ]]; then
      if grep -q '\[EVAL_ERROR\]' "${out_dir}/results.csv"; then
        log "regenerate ${method}/${attack}: existing results contain EVAL_ERROR"
        rm -f "${out_dir}/results.csv" "${out_dir}/results.json"
      else
        log "skip generation ${method}/${attack}: results complete ${have}/${need}"
        return 0
      fi
    fi
  fi
  start_model_vllm "${SFNL_MODEL}" "${served}" "${MODEL_PORT}" "${LOG_DIR}/SF-NL_${attack}_vllm.log"
  cd "${BASELINE_AGENT}"
  VLLM_API_BASE="http://127.0.0.1:${MODEL_PORT}/v1" \
  VLLM_MODEL_NAME="${served}" \
  QWEN_INFER_BACKEND="vllm_api" \
  VLLM_API_TIMEOUT="3600" \
  "${PY}" -m phase2.pipeline.step6_naive_faiss_eval \
    --assets-dir "${SFNL_ASSETS}" \
    --base-model-path "${SFNL_MODEL}" \
    --lora-path "${SFNL_MODEL}" \
    --encoder-model-path "${ENCODER_MODEL}" \
    --forget-path "${manifest}" \
    --retain-path "${EMPTY_RETAIN}" \
    --forget-limit 0 \
    --retain-limit 0 \
    --output-csv "${out_dir}/results.csv" \
    --output-json "${out_dir}/results.json" \
    --backend vllm_api \
    --num-workers "${MODEL_WORKERS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --rag-top-k 8 \
    --memory-top-k 4 \
    --embedding-device cpu \
    --embedding-batch-size 8 \
    --query-embedding-cache "${RUN_DIR}/query_embedding_cache_qwen35.npz" \
    --require-query-cache \
    --sfnl-prompt-mode paper_v2 \
    --progress-log-every 100 \
    --checkpoint-every 100 \
    > "${LOG_DIR}/SF-NL_${attack}_step6.log" 2>&1
  kill_port "${MODEL_PORT}"
}

run_sbu_method() {
  local attack="$1" manifest="$2"
  local method="SBU"
  local served="robust_sbu"
  local out_dir="${GEN_DIR}/${method}_${attack}"
  mkdir -p "${out_dir}"
  if [[ -f "${out_dir}/results.csv" ]]; then
    local have need
    have="$("${PY}" - <<PY
import csv
print(sum(1 for _ in csv.DictReader(open('${out_dir}/results.csv', encoding='utf-8'))))
PY
)"
    need="$(wc -l < "${manifest}")"
    if [[ "${have}" -ge "${need}" ]]; then
      if grep -q '\[EVAL_ERROR\]' "${out_dir}/results.csv"; then
        log "regenerate ${method}/${attack}: existing results contain EVAL_ERROR"
        rm -f "${out_dir}/results.csv" "${out_dir}/results.json"
      else
        log "skip generation ${method}/${attack}: results complete ${have}/${need}"
        return 0
      fi
    fi
  fi
  start_model_vllm "${SBU_MODEL}" "${served}" "${MODEL_PORT}" "${LOG_DIR}/${method}_${attack}_vllm.log"
  cd "${BASELINE_AGENT}"
  MY_SBU_ROOT="/disk8T/lkh/python_projects/my_sbu_plansv2.0" \
  VLLM_API_BASE="http://127.0.0.1:${MODEL_PORT}/v1" \
  VLLM_MODEL_NAME="${served}" \
  QWEN_INFER_BACKEND="vllm_api" \
  VLLM_API_TIMEOUT="3600" \
  "${PY}" -m phase2.pipeline.step6_sbu_eval \
    --base-model-path "${SBU_MODEL}" \
    --lora-path "${SBU_MODEL}" \
    --forget-path "${manifest}" \
    --retain-path "${EMPTY_RETAIN}" \
    --forget-limit 0 \
    --retain-limit 0 \
    --phase-a-assets-dir "${SBU_ASSETS}" \
    --output-csv "${out_dir}/results.csv" \
    --output-json "${out_dir}/results.json" \
    --backend vllm_api \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature 0.0 \
    --num-workers "${MODEL_WORKERS}" \
    --memory-top-k 5 \
    --checkpoint-every 100 \
    > "${LOG_DIR}/${method}_${attack}_step6.log" 2>&1
  kill_port "${MODEL_PORT}"
}

run_flowmu_method() {
  local attack="$1" manifest="$2"
  local method="FlowMU"
  local served="robust_flowmu"
  local out_dir="${GEN_DIR}/${method}_${attack}"
  mkdir -p "${out_dir}"
  if [[ -f "${out_dir}/results.csv" ]]; then
    local have need
    have="$("${PY}" - <<PY
import csv
print(sum(1 for _ in csv.DictReader(open('${out_dir}/results.csv', encoding='utf-8'))))
PY
)"
    need="$(wc -l < "${manifest}")"
    if [[ "${have}" -ge "${need}" ]]; then
      log "skip generation ${method}/${attack}: results complete ${have}/${need}"
      return 0
    fi
  fi
  start_model_vllm "${FLOWMU_MODEL}" "${served}" "${MODEL_PORT}" "${LOG_DIR}/${method}_${attack}_vllm.log"
  cd "${ROOT}"
  VLLM_API_BASE="http://127.0.0.1:${MODEL_PORT}/v1" \
  VLLM_MODEL_NAME="${served}" \
  QWEN_INFER_BACKEND="vllm_api" \
  VLLM_API_TIMEOUT="3600" \
  "${PY}" -m phase2.pipeline.step6_full_pipeline_eval \
    --base-model-path "${ROOT}/merge_model/native_llm" \
    --lora-path "${FLOWMU_MODEL}" \
    --encoder-model-path "${ENCODER_MODEL}" \
    --rag-graph-path "${FLOWMU_RAG_GRAPH}" \
    --memory-graph-path "${FLOWMU_MEMORY_GRAPH}" \
    --forget-path "${manifest}" \
    --retain-path "${EMPTY_RETAIN}" \
    --forget-limit 0 \
    --retain-limit 0 \
    --output-csv "${out_dir}/results.csv" \
    --output-json "${out_dir}/results.json" \
    --backend vllm_api \
    --embedding-device cpu \
    --embedding-batch-size 8 \
    --embedding-max-length 256 \
    --query-embedding-cache "${RUN_DIR}/query_embedding_cache_qwen35.npz" \
    --require-query-cache \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature 0.0 \
    --num-workers "${MODEL_WORKERS}" \
    --rag-fallback-top-k 3 \
    --memory-max-depth 5 \
    --memory-max-seeds 3 \
    --memory-max-hits 3 \
    --checkpoint-every 100 \
    > "${LOG_DIR}/${method}_${attack}_step6.log" 2>&1
  kill_port "${MODEL_PORT}"
}

run_generation() {
  local rewrite="${MANIFEST_DIR}/tofu_for05_rewrite.jsonl"
  local induce="${MANIFEST_DIR}/tofu_for05_induce.jsonl"
  run_naive_method "Vanilla" "${VANILLA_MODEL}" "${VANILLA_ASSETS}" "Rewrite" "${rewrite}"
  run_naive_method "Vanilla" "${VANILLA_MODEL}" "${VANILLA_ASSETS}" "Induce" "${induce}"
  run_naive_method "PURE" "${PURE_MODEL}" "${PURE_ASSETS}" "Rewrite" "${rewrite}"
  run_naive_method "PURE" "${PURE_MODEL}" "${PURE_ASSETS}" "Induce" "${induce}"
  run_sbu_method "Rewrite" "${rewrite}"
  run_sbu_method "Induce" "${induce}"
  run_sfnl_method "Rewrite" "${rewrite}"
  run_sfnl_method "Induce" "${induce}"
  run_flowmu_method "Rewrite" "${rewrite}"
  run_flowmu_method "Induce" "${induce}"
}

run_judge() {
  local labels=(
    "Vanilla_Rewrite" "Vanilla_Induce"
    "PURE_Rewrite" "PURE_Induce"
    "SBU_Rewrite" "SBU_Induce"
    "SF-NL_Rewrite" "SF-NL_Induce"
    "FlowMU_Rewrite" "FlowMU_Induce"
  )
  local files=(
    "${GEN_DIR}/Vanilla_Rewrite/results.csv" "${GEN_DIR}/Vanilla_Induce/results.csv"
    "${GEN_DIR}/PURE_Rewrite/results.csv" "${GEN_DIR}/PURE_Induce/results.csv"
    "${GEN_DIR}/SBU_Rewrite/results.csv" "${GEN_DIR}/SBU_Induce/results.csv"
    "${GEN_DIR}/SF-NL_Rewrite/results.csv" "${GEN_DIR}/SF-NL_Induce/results.csv"
    "${GEN_DIR}/FlowMU_Rewrite/results.csv" "${GEN_DIR}/FlowMU_Induce/results.csv"
  )
  local need_judge=0
  local pending_labels=()
  local pending_files=()
  for i in "${!labels[@]}"; do
    local label="${labels[$i]}"
    local raw_file="${files[$i]}"
    local judge_file="${JUDGE_DIR}/judge_${label}.csv"
    if [[ ! -f "${judge_file}" ]]; then
      need_judge=1
      pending_labels+=("${label}")
      pending_files+=("${raw_file}")
      continue
    fi
    local raw_rows judged_rows
    raw_rows="$("${PY}" - <<PY
import csv
print(sum(1 for _ in csv.DictReader(open('${raw_file}', encoding='utf-8'))))
PY
)"
    judged_rows="$("${PY}" - <<PY
import csv
print(sum(1 for _ in csv.DictReader(open('${judge_file}', encoding='utf-8'))))
PY
)"
    if [[ "${judged_rows}" -lt "${raw_rows}" ]]; then
      log "judge incomplete ${label}: ${judged_rows}/${raw_rows}"
      need_judge=1
      pending_labels+=("${label}")
      pending_files+=("${raw_file}")
    elif [[ "${judge_file}" -ot "${raw_file}" ]]; then
      log "judge stale ${label}: ${judge_file} older than ${raw_file}"
      rm -f "${judge_file}"
      need_judge=1
      pending_labels+=("${label}")
      pending_files+=("${raw_file}")
    fi
  done
  if [[ "${need_judge}" == "0" ]]; then
    log "skip judge: all judge csv files exist"
    return 0
  fi
  log "judge pending labels: ${pending_labels[*]}"
  start_judge_vllm "${JUDGE_PORT}" "${LOG_DIR}/judge_vllm.log"
  cd "${ROOT}"
  "${PY}" evaluation/llm_judge.py \
    --files "${pending_files[@]}" \
    --labels "${pending_labels[@]}" \
    --api-url "http://127.0.0.1:${JUDGE_PORT}/v1" \
    --model judge \
    --output-dir "${JUDGE_DIR}" \
    --num-workers "${JUDGE_WORKERS}" \
    > "${LOG_DIR}/judge.log" 2>&1
  kill_port "${JUDGE_PORT}"
}

summarize_and_plot() {
  cd "${ROOT}"
  "${PY}" evaluation/robustness/summarize_rewrite_induce_attack.py \
    --rewrite-manifest "${MANIFEST_DIR}/tofu_for05_rewrite.jsonl" \
    --induce-manifest "${MANIFEST_DIR}/tofu_for05_induce.jsonl" \
    --judge "Vanilla:Rewrite:${JUDGE_DIR}/judge_Vanilla_Rewrite.csv" \
    --judge "Vanilla:Induce:${JUDGE_DIR}/judge_Vanilla_Induce.csv" \
    --judge "PURE:Rewrite:${JUDGE_DIR}/judge_PURE_Rewrite.csv" \
    --judge "PURE:Induce:${JUDGE_DIR}/judge_PURE_Induce.csv" \
    --judge "SBU:Rewrite:${JUDGE_DIR}/judge_SBU_Rewrite.csv" \
    --judge "SBU:Induce:${JUDGE_DIR}/judge_SBU_Induce.csv" \
    --judge "SF-NL:Rewrite:${JUDGE_DIR}/judge_SF-NL_Rewrite.csv" \
    --judge "SF-NL:Induce:${JUDGE_DIR}/judge_SF-NL_Induce.csv" \
    --judge "FlowMU:Rewrite:${JUDGE_DIR}/judge_FlowMU_Rewrite.csv" \
    --judge "FlowMU:Induce:${JUDGE_DIR}/judge_FlowMU_Induce.csv" \
    --clean-acc "Vanilla=0.900" \
    --clean-acc "PURE=0.530" \
    --clean-acc "SBU=0.635" \
    --clean-acc "SF-NL=0.780" \
    --clean-acc "FlowMU=0.285" \
    --output-csv "${SUMMARY_DIR}/tofu_for05_rewrite_induce_attack_real_summary.csv" \
    --output-json "${SUMMARY_DIR}/tofu_for05_rewrite_induce_attack_real_summary.json"

  cp "${SUMMARY_DIR}/tofu_for05_rewrite_induce_attack_real_summary.csv" "${FINAL_DIR}/tofu_for05_rewrite_induce_attack_real_summary.csv"
  cp "${SUMMARY_DIR}/tofu_for05_rewrite_induce_attack_real_summary.json" "${FINAL_DIR}/tofu_for05_rewrite_induce_attack_real_summary.json"

  "${PY}" final_performance/tofu/small_exp/robustness/rewrite_induce_attack_plot/plot_tofu_for05_rewrite_induce_attack.py \
    --input-csv "${FINAL_DIR}/tofu_for05_rewrite_induce_attack_real_summary.csv" \
    --output-dir "${FINAL_DIR}" \
    --stem "tofu_for05_rewrite_induce_attack"
}

log "=== TOFU-for05 rewrite/induce robustness start SMOKE_LIMIT=${SMOKE_LIMIT} ==="
generate_manifests
build_query_cache
run_generation
run_judge
summarize_and_plot
kill_port "${MODEL_PORT}"
kill_port "${JUDGE_PORT}"
log "=== done; final dir: ${FINAL_DIR} ==="
