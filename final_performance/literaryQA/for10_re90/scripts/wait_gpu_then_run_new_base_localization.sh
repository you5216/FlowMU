#!/usr/bin/env bash
set -euo pipefail


AGENT_DIR="${AGENT_DIR:-/disk8T/lkh/python_projects/agent_unlearning_zhx/agent}"
PYTHON_BIN="${PYTHON_BIN:-/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/env52/bin/python}"
GPU_ID="${GPU_ID:-0}"
MAX_USED_MIB="${MAX_USED_MIB:-2500}"
POLL_SECONDS="${POLL_SECONDS:-60}"

RUN_DIR="${RUN_DIR:-${AGENT_DIR}/final_performance/literaryQA/for10_re90/localization/literary_neative_model_rag_memory_final_combined_top300_rr08_20260701}"
MONITOR_LOG="${MONITOR_LOG:-${RUN_DIR}/wait_gpu_then_run.log}"
LOCALIZATION_LOG="${LOCALIZATION_LOG:-${RUN_DIR}/run.log}"

BASE_MODEL="${BASE_MODEL:-${AGENT_DIR}/final_performance/literaryQA/base/lora/merged/literary_neative_model}"
RAG_GRAPH="${RAG_GRAPH:-${AGENT_DIR}/final_performance/literaryQA/for10_re90/rag_memory_unlearning_graph/rag_unlearning_graph.pkl}"
MEMORY_GRAPH="${MEMORY_GRAPH:-${AGENT_DIR}/final_performance/literaryQA/for10_re90/rag_memory_unlearning_graph/memory_unlearning_graph.pkl}"
MEMORY_REPORT="${MEMORY_REPORT:-${AGENT_DIR}/final_performance/literaryQA/for10_re90/rag_memory_unlearning_graph/train_artifacts/memory_unlearning_report.json}"
QA_FORGET="${QA_FORGET:-${AGENT_DIR}/final_performance/literaryQA/for10_re90/phase2_graph_qa_trial_01_strongqa_memoryfix/graph_forget_qa.jsonl}"
QA_RETAIN="${QA_RETAIN:-${AGENT_DIR}/final_performance/literaryQA/for10_re90/phase2_graph_qa_trial_01_strongqa_memoryfix/graph_retain_qa.jsonl}"

mkdir -p "${RUN_DIR}"

log() {
  local ts
  ts="$(date '+%F %T')"
  echo "[${ts}] $*" | tee -a "${MONITOR_LOG}"
}

require_file() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    log "ERROR: missing required path: ${path}"
    exit 2
  fi
}

require_file "${PYTHON_BIN}"
require_file "${BASE_MODEL}"
require_file "${RAG_GRAPH}"
require_file "${MEMORY_GRAPH}"
require_file "${MEMORY_REPORT}"
require_file "${QA_FORGET}"
require_file "${QA_RETAIN}"

log "Waiting for GPU ${GPU_ID}: memory.used <= ${MAX_USED_MIB} MiB; poll=${POLL_SECONDS}s"
log "Run dir: ${RUN_DIR}"

while true; do
  used_mib="$(
    nvidia-smi --id="${GPU_ID}" --query-gpu=memory.used --format=csv,noheader,nounits \
      | head -n 1 \
      | tr -d ' '
  )"
  util="$(
    nvidia-smi --id="${GPU_ID}" --query-gpu=utilization.gpu --format=csv,noheader,nounits \
      | head -n 1 \
      | tr -d ' '
  )"
  log "GPU ${GPU_ID}: memory.used=${used_mib} MiB, util=${util}%"

  if [[ "${used_mib}" =~ ^[0-9]+$ ]] && (( used_mib <= MAX_USED_MIB )); then
    break
  fi
  sleep "${POLL_SECONDS}"
done

log "GPU ${GPU_ID} is available. Starting for10_re90 new-base raw localization."

cd "${AGENT_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

"${PYTHON_BIN}" phase2/diagnostics/native_llm_graph_neuron_localizer.py \
  --base-model-path "${BASE_MODEL}" \
  --rag-graph-path "${RAG_GRAPH}" \
  --memory-graph-path "${MEMORY_GRAPH}" \
  --memory-report-path "${MEMORY_REPORT}" \
  --qa-forget-path "${QA_FORGET}" \
  --qa-retain-path "${QA_RETAIN}" \
  --qa-max-forget 0 \
  --qa-max-retain 2230 \
  --output-dir "${RUN_DIR}" \
  --device cuda \
  --seed 42 \
  --rag-forget-threshold 0.5 \
  --rag-retain-threshold 0.15 \
  --max-retain 0 \
  --layer-range 0-31 \
  --neurons-per-layer 300 \
  --batch-size 8 \
  --activation-position all \
  --anchor-text "Graph concept probe." \
  --qa-batch-size 1 \
  --qa-max-length 512 \
  --graph-view-weight 1.0 \
  --qa-view-weight 1.0 \
  --ranking-mode positive_diff \
  --positive-diff-retain-ratio-max 0.8 \
  --new-lora-r 16 \
  --new-lora-alpha 32 \
  --new-lora-dropout 0.05 \
  --new-lora-target-modules gate_proj,up_proj,down_proj \
  2>&1 | tee "${LOCALIZATION_LOG}"

log "Localization finished. Output: ${RUN_DIR}"
