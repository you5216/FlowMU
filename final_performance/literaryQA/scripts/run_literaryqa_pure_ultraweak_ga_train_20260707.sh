#!/usr/bin/env bash
set -euo pipefail

AGENT_ROOT="${AGENT_ROOT:-/disk8T/lkh/python_projects/agent_unlearning_zhx/agent}"
PY="${PY:-/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/env52/bin/python}"
GPU_ID="${GPU_ID:-0}"
RATIOS="${RATIOS:-10 05 01}"
RUN_TAG="${RUN_TAG:-pure_ultraweak_ga_20260707}"

export HOME="${HOME:-/disk8T/lkh/python_projects/agent_unlearning_zhx/runtime_home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/disk8T/lkh/python_projects/agent_unlearning_zhx/runtime_cache}"
export HF_HOME="${HF_HOME:-/disk8T/lkh/python_projects/agent_unlearning_zhx/runtime_hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-/disk8T/lkh/python_projects/agent_unlearning_zhx/runtime_cache/torch}"
export TMPDIR="${TMPDIR:-/disk8T/lkh/python_projects/tmp_vllm}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

BASE_MODEL="${BASE_MODEL:-final_performance/literaryQA/base/lora/merged/literary_neative_model}"
ORIGINAL_LORA="${ORIGINAL_LORA:-/disk8T/lkh/python_projects/tofu_native_lora/checkpoint-4500}"
GRAPH_PATH="${GRAPH_PATH:-/disk8T/lkh/python_projects/agent_unlearning_zhx/agent/phase1/artifacts/unified_graph.pkl}"
CORE_PATH="${CORE_PATH:-/disk8T/lkh/python_projects/agent_unlearning_zhx/agent/phase1/artifacts/D_forget_core.jsonl}"
CONTEXT_PATH="${CONTEXT_PATH:-/disk8T/lkh/python_projects/agent_unlearning_zhx/agent/phase1/artifacts/D_forget_context.jsonl}"
LOG_ROOT="${LOG_ROOT:-${AGENT_ROOT}/final_performance/literaryQA/logs/${RUN_TAG}}"

mkdir -p "${LOG_ROOT}"

ratio_config() {
  case "$1" in
    01)
      CELL="for01_re99"
      FORGET_QA="final_performance/literaryQA/for01_re99/phase2_graph_qa_rag_memory_final/graph_forget_qa.jsonl"
      RETAIN_EMPTY="final_performance/literaryQA/for01_re99/empty_retain_for_forget_only.json"
      LOC_PKL="final_performance/literaryQA/for01_re99/localization/literary_neative_model_cached_top50_rr06_20260630/localized_neurons.pkl"
      OUT_DIR="final_performance/literaryQA/for01_re99/lora/ga_pure_ultraweak_literarynative_top50rr06_lr3e4_gn1_ce3_15_20260707"
      MAX_STEPS="${MAX_STEPS_FOR01:-300}"
      ;;
    05)
      CELL="for05_re95"
      FORGET_QA="final_performance/literaryQA/for05_re95/phase2_graph_qa_final_rag_memory_graph_20260629/graph_forget_qa.jsonl"
      RETAIN_EMPTY="final_performance/literaryQA/for05_re95/empty_retain_for_forget_only.json"
      LOC_PKL="final_performance/literaryQA/for05_re95/localization/literary_neative_model_cached_combined_top50_rr09_20260701/localized_neurons.pkl"
      OUT_DIR="final_performance/literaryQA/for05_re95/lora/ga_pure_ultraweak_literarynative_top50rr09_lr3e4_gn1_ce3_15_20260707"
      MAX_STEPS="${MAX_STEPS_FOR05:-300}"
      ;;
    10)
      CELL="for10_re90"
      FORGET_QA="final_performance/literaryQA/for10_re90/phase2_graph_qa_trial_01_strongqa_memoryfix/graph_forget_qa.jsonl"
      RETAIN_EMPTY="final_performance/literaryQA/for10_re90/empty_retain_for_forget_only.json"
      LOC_PKL="final_performance/literaryQA/for10_re90/localization/literary_neative_model_cached_combined_top50_rr09_20260701/localized_neurons.pkl"
      OUT_DIR="final_performance/literaryQA/for10_re90/lora/ga_pure_ultraweak_literarynative_top50rr09_lr3e4_gn1_ce3_15_20260707"
      MAX_STEPS="${MAX_STEPS_FOR10:-260}"
      ;;
    *)
      echo "[ERROR] bad ratio: $1" >&2
      exit 2
      ;;
  esac
}

cd "${AGENT_ROOT}"

{
  echo "RUN_TAG=${RUN_TAG}"
  echo "AGENT_ROOT=${AGENT_ROOT}"
  echo "PY=${PY}"
  echo "GPU_ID=${GPU_ID}"
  echo "RATIOS=${RATIOS}"
  echo "BASE_MODEL=${BASE_MODEL}"
  echo "ORIGINAL_LORA=${ORIGINAL_LORA}"
  echo "CE_THRESHOLDS=3,5,7.5,10,12.5,15"
  date
} | tee "${LOG_ROOT}/env.txt"

for ratio in ${RATIOS}; do
  ratio_config "${ratio}"
  mkdir -p "${OUT_DIR}/logs"
  echo "[stage] train ratio=${ratio} cell=${CELL} out=${OUT_DIR}" | tee -a "${LOG_ROOT}/run.log"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PY}" -m phase2.pipeline.main \
    --stage unlearning \
    --base-model-path "${BASE_MODEL}" \
    --original-lora-path "${ORIGINAL_LORA}" \
    --forget-target-path "${FORGET_QA}" \
    --retain-path "${RETAIN_EMPTY}" \
    --unlearning-output-dir "${OUT_DIR}" \
    --init-new-lora \
    --new-lora-r 16 \
    --new-lora-alpha 32 \
    --new-lora-dropout 0.05 \
    --new-lora-target-modules gate_proj,up_proj,down_proj \
    --graph-path "${GRAPH_PATH}" \
    --core-dataset-path "${CORE_PATH}" \
    --context-dataset-path "${CONTEXT_PATH}" \
    --reuse-localization-pkl "${LOC_PKL}" \
    --per-device-train-batch-size 2 \
    --gradient-accumulation-steps 4 \
    --learning-rate 3e-4 \
    --num-train-epochs 20 \
    --max-train-steps "${MAX_STEPS}" \
    --logging-steps 5 \
    --max-length 512 \
    --warmup-steps 5 \
    --max-grad-norm 1.0 \
    --forget-loss-strategy gradient_ascent \
    --forget-objective-weight 1.0 \
    --retain-kl-weight 0.0 \
    --disable-dual-kl-objective \
    --forget-ce-checkpoint-thresholds 3,5,7.5,10,12.5,15 \
    2>&1 | tee "${OUT_DIR}/logs/train.log"
  echo "[done] train ratio=${ratio}" | tee -a "${LOG_ROOT}/run.log"
done

echo "[stage] finished ${RUN_TAG}" | tee -a "${LOG_ROOT}/run.log"
