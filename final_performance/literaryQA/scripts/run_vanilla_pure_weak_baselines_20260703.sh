#!/usr/bin/env bash
set -euo pipefail

AGENT_ROOT="${AGENT_ROOT:-/disk8T/lkh/python_projects/agent_unlearning_zhx/agent}"
BASELINE_ROOT="${BASELINE_ROOT:-/disk8T/lkh/python_projects/baseline_xzy(2)}"
PY="${PY:-/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/env52/bin/python}"
VLLM_BIN_DIR="${VLLM_BIN_DIR:-/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/52vllm/bin}"
LQA_ROOT="${LQA_ROOT:-/disk8T/lkh/python_projects/datasets/LiteraryQA/literaryqa_annotations}"

NATIVE_MODEL="${NATIVE_MODEL:-${AGENT_ROOT}/final_performance/literaryQA/base/lora/merged/literary_neative_model}"
PURE_FOR01="${PURE_FOR01:-${AGENT_ROOT}/final_performance/literaryQA/for01_re99/merged_model/ga_pure_weak/literaryQA_for01_re99_ga_ce20_weakpure_20260703}"
PURE_FOR05="${PURE_FOR05:-${AGENT_ROOT}/final_performance/literaryQA/for05_re95/merged_model/ga_pure_weak/literaryQA_for05_re95_ga_ce25_weakpure_20260703}"
PURE_FOR10="${PURE_FOR10:-${AGENT_ROOT}/final_performance/literaryQA/for10_re90/merged_model/ga_pure_weak/literaryQA_for10_re90_ga_ce10_weakpure_20260703}"

RUN_TAG="${RUN_TAG:-vanilla_pure_weak_20260703}"
WEAK_MERGED_ROOT="${WEAK_MERGED_ROOT:-${BASELINE_ROOT}/experiments/literaryqa/merged_weakpure_20260703}"
LOG_ROOT="${LOG_ROOT:-${AGENT_ROOT}/final_performance/literaryQA/logs/${RUN_TAG}}"
QUERY_CACHE_ROOT="${QUERY_CACHE_ROOT:-${BASELINE_ROOT}/experiments/literaryqa/query_cache}"
QUERY_CACHE_PATH="${QUERY_CACHE_PATH:-${QUERY_CACHE_ROOT}/literaryqa_eval_queries_qwen35.npz}"
PRECOMPUTE_QUERY_CACHE="${PRECOMPUTE_QUERY_CACHE:-1}"
FAKE_CONDA_SH="${LOG_ROOT}/fake_conda_for_52vllm.sh"

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "${path}" ]]; then
    echo "[ERROR] missing ${label}: ${path}" >&2
    exit 2
  fi
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] missing ${label}: ${path}" >&2
    exit 2
  fi
}

mkdir -p "${LOG_ROOT}" "${WEAK_MERGED_ROOT}" "${QUERY_CACHE_ROOT}"

require_dir "${BASELINE_ROOT}" "baseline repo"
require_file "${PY}" "env52 python"
require_file "${VLLM_BIN_DIR}/vllm" "52vllm executable"
require_dir "${LQA_ROOT}" "LiteraryQA annotations"
require_file "${NATIVE_MODEL}/config.json" "native model config"
require_file "${PURE_FOR01}/config.json" "for01 weak Pure model config"
require_file "${PURE_FOR05}/config.json" "for05 weak Pure model config"
require_file "${PURE_FOR10}/config.json" "for10 weak Pure model config"

if ! grep -q '^from config_52 import \\*' "${BASELINE_ROOT}/agent/phase1/config.py"; then
  echo "[ERROR] ${BASELINE_ROOT}/agent/phase1/config.py is not set to config_52." >&2
  echo "[ERROR] Switch it to: from config_52 import *" >&2
  exit 2
fi

ln -sfn "${PURE_FOR01}" "${WEAK_MERGED_ROOT}/pure_for01"
ln -sfn "${PURE_FOR05}" "${WEAK_MERGED_ROOT}/pure_for05"
ln -sfn "${PURE_FOR10}" "${WEAK_MERGED_ROOT}/pure_for10"

cat > "${FAKE_CONDA_SH}" <<EOF
export PATH="${VLLM_BIN_DIR}:\${PATH}"
conda() { return 0; }
EOF
chmod +x "${FAKE_CONDA_SH}"

export PATH="${VLLM_BIN_DIR}:${PATH}"
export NATIVE_CONDA_SH="${FAKE_CONDA_SH}"
export NATIVE_ENV="noop"
export ENV50_PY="${PY}"
export LQA_ROOT
export NATIVE_LLM="${NATIVE_MODEL}"
export MERGED_ROOT="${WEAK_MERGED_ROOT}"
export PROTOCOL="${PROTOCOL:-tofu}"
export RATIOS="${RATIOS:-01 05 10}"
export SMOKE="${SMOKE:-0}"
export RESUME="${RESUME:-0}"
export SKIP_DONE="${SKIP_DONE:-0}"
export SKIP_JUDGE="${SKIP_JUDGE:-0}"
export GPU_ID="${GPU_ID:-0}"
export MODEL_PORT="${MODEL_PORT:-8000}"
export JUDGE_PORT="${JUDGE_PORT:-8001}"
export JUDGE_MODEL="${JUDGE_MODEL:-/disk8T/lkh/python_projects/llms/Qwen2.5-VL-7B-Instruct}"
export NUM_WORKERS="${NUM_WORKERS:-16}"
export JUDGE_NUM_WORKERS="${JUDGE_NUM_WORKERS:-16}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
export NAIVE_FAISS_EVAL_EMBED_DEVICE="${NAIVE_FAISS_EVAL_EMBED_DEVICE:-cpu}"
export NAIVE_FAISS_QUERY_CACHE_ROOT="${QUERY_CACHE_ROOT}"
export NAIVE_FAISS_REQUIRE_QUERY_CACHE="${NAIVE_FAISS_REQUIRE_QUERY_CACHE:-1}"
export NAIVE_FAISS_QUERY_CACHE_SAVE_EVERY="${NAIVE_FAISS_QUERY_CACHE_SAVE_EVERY:-512}"
export NAIVE_FAISS_PROGRESS_EVERY="${NAIVE_FAISS_PROGRESS_EVERY:-100}"
export EMBED_DEVICE="${EMBED_DEVICE:-cpu}"
export EMBED_BATCH="${EMBED_BATCH:-8}"
export ENCODER="${ENCODER:-/disk8T/lkh/python_projects/llms/Qwen3.5-9B}"
export REBUILD="${REBUILD:-0}"

{
  echo "RUN_TAG=${RUN_TAG}"
  echo "BASELINE_ROOT=${BASELINE_ROOT}"
  echo "AGENT_ROOT=${AGENT_ROOT}"
  echo "NATIVE_LLM=${NATIVE_LLM}"
  echo "MERGED_ROOT=${MERGED_ROOT}"
  echo "LQA_ROOT=${LQA_ROOT}"
  echo "PROTOCOL=${PROTOCOL}"
  echo "RATIOS=${RATIOS}"
  echo "SKIP_JUDGE=${SKIP_JUDGE}"
  echo "GPU_ID=${GPU_ID}"
  echo "MODEL_PORT=${MODEL_PORT}"
  echo "JUDGE_PORT=${JUDGE_PORT}"
  echo "JUDGE_MODEL=${JUDGE_MODEL}"
  echo "PY=${PY}"
  echo "VLLM=$(command -v vllm)"
  echo "ENCODER=${ENCODER}"
  echo "QUERY_CACHE_PATH=${QUERY_CACHE_PATH}"
  echo "PRECOMPUTE_QUERY_CACHE=${PRECOMPUTE_QUERY_CACHE}"
  echo "NAIVE_FAISS_REQUIRE_QUERY_CACHE=${NAIVE_FAISS_REQUIRE_QUERY_CACHE}"
  echo "NAIVE_FAISS_QUERY_CACHE_SAVE_EVERY=${NAIVE_FAISS_QUERY_CACHE_SAVE_EVERY}"
} | tee "${LOG_ROOT}/env.txt"

cd "${BASELINE_ROOT}"

echo "[stage] ensure vanilla FAISS assets" | tee -a "${LOG_ROOT}/run.log"
bash scripts/build_literaryqa_faiss_assets.sh vanilla all 2>&1 | tee -a "${LOG_ROOT}/build_vanilla.log"

echo "[stage] ensure pure FAISS assets" | tee -a "${LOG_ROOT}/run.log"
bash scripts/build_literaryqa_faiss_assets.sh pure all 2>&1 | tee -a "${LOG_ROOT}/build_pure.log"

ratio_cfg() {
  case "$1" in
    01) CFG_FORGET_PATH="${LQA_ROOT}/memory/forget01.json"; CFG_RETAIN_PATH="${LQA_ROOT}/memory/retain99.json"; CFG_FL=204; CFG_RL=18857 ;;
    05) CFG_FORGET_PATH="${LQA_ROOT}/memory/forget05.json"; CFG_RETAIN_PATH="${LQA_ROOT}/memory/retain95.json"; CFG_FL=992; CFG_RL=18069 ;;
    10) CFG_FORGET_PATH="${LQA_ROOT}/memory/forget10.json"; CFG_RETAIN_PATH="${LQA_ROOT}/memory/retain90.json"; CFG_FL=1970; CFG_RL=17091 ;;
    *) echo "[ERROR] bad ratio: $1" >&2; exit 2 ;;
  esac
}

if [[ "${PRECOMPUTE_QUERY_CACHE}" == "1" ]]; then
  echo "[stage] precompute query embedding cache on GPU" | tee -a "${LOG_ROOT}/run.log"
  for ratio in ${RATIOS}; do
    ratio_cfg "${ratio}"
    assets_dir="${BASELINE_ROOT}/experiments/literaryqa/naive_faiss/vanilla_for${ratio}"
    require_file "${assets_dir}/metadata.json" "vanilla FAISS metadata for${ratio}"
    (
      cd "${BASELINE_ROOT}/agent"
      CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PY}" -m phase2.pipeline.step6_naive_faiss_eval \
        --assets-dir "${assets_dir}" \
        --base-model-path "${NATIVE_MODEL}" \
        --lora-path "${NATIVE_MODEL}" \
        --forget-path "${CFG_FORGET_PATH}" \
        --retain-path "${CFG_RETAIN_PATH}" \
        --forget-limit "${CFG_FL}" \
        --retain-limit "${CFG_RL}" \
        --output-csv "${LOG_ROOT}/query_cache_dummy_for${ratio}.csv" \
        --output-json "${LOG_ROOT}/query_cache_dummy_for${ratio}.json" \
        --backend vllm_api \
        --embedding-device cuda \
        --query-embedding-cache "${QUERY_CACHE_PATH}" \
        --precompute-query-cache-only \
        > "${LOG_ROOT}/query_cache_for${ratio}.log" 2>&1
    )
  done
fi

echo "[stage] run vanilla matrix" | tee -a "${LOG_ROOT}/run.log"
METHODS=vanilla bash scripts/run_literaryqa_baseline_matrix.sh 2>&1 | tee -a "${LOG_ROOT}/vanilla_matrix.log"

echo "[stage] run pure matrix" | tee -a "${LOG_ROOT}/run.log"
METHODS=pure bash scripts/run_literaryqa_baseline_matrix.sh 2>&1 | tee -a "${LOG_ROOT}/pure_matrix.log"

echo "[stage] finished ${RUN_TAG}" | tee -a "${LOG_ROOT}/run.log"
