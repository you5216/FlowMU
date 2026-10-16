#!/usr/bin/env bash
set -euo pipefail

AGENT_DIR="/home/likunhao/likunhao52shared/agent_unlearning_zhx/agent"
PY="/home/likunhao/likunhao52shared/agent_unlearning_zhx/environment/env50/bin/python"
VLLM_BIN="/home/likunhao/miniconda3/envs/nativeqwen310/bin/vllm"

BASE_MODEL="${AGENT_DIR}/merge_model/native_llm"
ORIGINAL_LORA="/home/likunhao/likunhao52shared/tofu_native_lora/checkpoint-2000"
LOCALIZATION_PKL="${AGENT_DIR}/phase2/diagnostics/native_llm_graph_unlearning_graph_full0_31_top300/localized_neurons.pkl"
JUDGE_MODEL="/home/likunhao/likunhao52shared/llms/Qwen2.5-VL-7B-Instruct"

SOURCE_DATA_DIR="${AGENT_DIR}/phase2/unlearning_runs/graph_subgraph_qa_fullqwen_randvec_anssup_kl/data"
RUN_NAME="${RUN_NAME:?RUN_NAME is required}"
RUN_ROOT="${AGENT_DIR}/phase2/unlearning_runs/${RUN_NAME}"
DATA_DIR="${RUN_ROOT}/data"
LORA_DIR="${RUN_ROOT}/lora"
MERGED_DIR="${AGENT_DIR}/phase2/merged_models/${RUN_NAME}"
EVAL_DIR="${AGENT_DIR}/phase2/eval_runs/${RUN_NAME}"
LOG_DIR="${RUN_ROOT}/logs"
MASTER_LOG="${LOG_DIR}/master.log"

FORGET_PATH="${DATA_DIR}/graph_forget_qa.jsonl"
RETAIN_PATH="${DATA_DIR}/graph_retain_qa.jsonl"

GPU_ID="${GPU_ID:-0}"
MODEL_PORT="${MODEL_PORT:-8000}"
JUDGE_PORT="${JUDGE_PORT:-8001}"
EVAL_WORKERS="${EVAL_WORKERS:-16}"

LEARNING_RATE="${LEARNING_RATE:-0.0002}"
RANDOM_WEIGHT="${RANDOM_WEIGHT:-1.0}"
RANDOM_SCALE="${RANDOM_SCALE:-0.01}"
NLL_WEIGHT="${NLL_WEIGHT:-1.5}"
NLL_MARGIN="${NLL_MARGIN:-0.75}"
RETAIN_KL_WEIGHT="${RETAIN_KL_WEIGHT:-1.1}"
EARLY_STOP_MIN_STEPS="${EARLY_STOP_MIN_STEPS:-200}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-220}"
BEST_SCORE_RETAIN_ALPHA="${BEST_SCORE_RETAIN_ALPHA:-4.5}"
BEST_SCORE_MIA_ALPHA="${BEST_SCORE_MIA_ALPHA:-0.0}"
DISABLE_EARLY_STOP="${DISABLE_EARLY_STOP:-0}"

mkdir -p "${DATA_DIR}" "${LORA_DIR}" "${MERGED_DIR}" "${EVAL_DIR}" "${LOG_DIR}"
cp "${SOURCE_DATA_DIR}/graph_forget_qa.jsonl" "${FORGET_PATH}"
cp "${SOURCE_DATA_DIR}/graph_retain_qa.jsonl" "${RETAIN_PATH}"
cd "${AGENT_DIR}"

log() {
  local line
  line="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "${line}" | tee -a "${MASTER_LOG}"
}

kill_vllm() {
  local pids=()
  mapfile -t pids < <(pgrep -f "[v]llm serve" || true)
  if ((${#pids[@]} > 0)); then
    log "Stopping existing vLLM processes: ${pids[*]}"
    kill "${pids[@]}" 2>/dev/null || true
    sleep 8
  fi
}

wait_openai_server() {
  local port="$1"
  local label="$2"
  local log_file="$3"
  local waited=0
  local max_wait=900
  log "Waiting for ${label} on port ${port}"
  while true; do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      log "${label} is ready"
      return 0
    fi
    if ! pgrep -f "vllm serve" >/dev/null 2>&1; then
      log "ERROR: ${label} exited before ready"
      tail -n 120 "${log_file}" || true
      exit 1
    fi
    if (( waited >= max_wait )); then
      log "ERROR: timeout waiting for ${label}"
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
  kill_vllm
  log "Starting vLLM served=${served_name} port=${port} model=${model_path}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" nohup "${VLLM_BIN}" serve "${model_path}" \
    --host 0.0.0.0 \
    --port "${port}" \
    --served-model-name "${served_name}" \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --trust-remote-code > "${log_file}" 2>&1 &
  wait_openai_server "${port}" "${served_name}" "${log_file}"
}

log "Starting run ${RUN_NAME}"
log "Hyperparams lr=${LEARNING_RATE} random_weight=${RANDOM_WEIGHT} random_scale=${RANDOM_SCALE} nll_weight=${NLL_WEIGHT} nll_margin=${NLL_MARGIN} retain_kl=${RETAIN_KL_WEIGHT} best_retain_alpha=${BEST_SCORE_RETAIN_ALPHA} best_mia_alpha=${BEST_SCORE_MIA_ALPHA} disable_early_stop=${DISABLE_EARLY_STOP}"

kill_vllm
rm -rf "${LORA_DIR}" "${MERGED_DIR}" "${EVAL_DIR}"
mkdir -p "${LORA_DIR}" "${MERGED_DIR}" "${EVAL_DIR}"

EXTRA_TRAIN_ARGS=()
if [[ "${DISABLE_EARLY_STOP}" == "1" ]]; then
  EXTRA_TRAIN_ARGS+=(--disable-early-stop)
fi

log "Training LoRA"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PY}" -m phase2.pipeline.main \
  --stage unlearning \
  --base-model-path "${BASE_MODEL}" \
  --original-lora-path "${ORIGINAL_LORA}" \
  --forget-target-path "${FORGET_PATH}" \
  --retain-path "${RETAIN_PATH}" \
  --unlearning-output-dir "${LORA_DIR}" \
  --init-new-lora \
  --new-lora-r 16 \
  --new-lora-alpha 32 \
  --new-lora-dropout 0.05 \
  --new-lora-target-modules gate_proj,up_proj,down_proj \
  --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 1 \
  --seed 42 \
  --learning-rate "${LEARNING_RATE}" \
  --num-train-epochs 3.0 \
  --logging-steps 20 \
  --max-length 512 \
  --warmup-steps 10 \
  --max-grad-norm 0.3 \
  --reuse-localization-pkl "${LOCALIZATION_PKL}" \
  --localization-key-annotation-path "" \
  --forget-loss-strategy parameter_random_vector \
  --forget-objective-weight 1.0 \
  --parameter-random-vector-weight "${RANDOM_WEIGHT}" \
  --parameter-random-vector-scale "${RANDOM_SCALE}" \
  --forget-loss-top-layers 32 \
  --graph-vector-suppress-weight 0.0 \
  --forget-nll-calibration-weight "${NLL_WEIGHT}" \
  --forget-nll-calibration-target-mode retain_ce \
  --forget-nll-calibration-margin "${NLL_MARGIN}" \
  --retain-kl-weight "${RETAIN_KL_WEIGHT}" \
  --disable-dual-kl-objective \
  --retain-kl-target 0.08 \
  --retain-topk 32 \
  --early-stop-min-steps "${EARLY_STOP_MIN_STEPS}" \
  --early-stop-patience "${EARLY_STOP_PATIENCE}" \
  --best-score-retain-alpha "${BEST_SCORE_RETAIN_ALPHA}" \
  --best-score-mia-alpha "${BEST_SCORE_MIA_ALPHA}" \
  --random-vector-seed 42 \
  "${EXTRA_TRAIN_ARGS[@]}" > "${LOG_DIR}/train.log" 2>&1
log "Training finished"

log "Merging LoRA"
"${PY}" -m phase2.pipeline.merge_lora_for_vllm \
  --base-model-path "${BASE_MODEL}" \
  --lora-path "${LORA_DIR}" \
  --output-dir "${MERGED_DIR}" \
  --torch-dtype float16 \
  --device-map cpu \
  --max-shard-size 4GB \
  --label "${RUN_NAME}" \
  --force > "${LOG_DIR}/merge.log" 2>&1
cp "${BASE_MODEL}/preprocessor_config.json" "${MERGED_DIR}/preprocessor_config.json" 2>/dev/null || true
cp "${BASE_MODEL}/video_preprocessor_config.json" "${MERGED_DIR}/video_preprocessor_config.json" 2>/dev/null || true
log "Merge finished"

start_vllm "${MERGED_DIR}" "${RUN_NAME}" "${MODEL_PORT}" "${LOG_DIR}/vllm_eval.log"
log "Generating evaluation outputs"
VLLM_API_BASE="http://127.0.0.1:${MODEL_PORT}/v1" \
VLLM_MODEL_NAME="${RUN_NAME}" \
QWEN_INFER_BACKEND="vllm_api" \
"${PY}" -m phase2.pipeline.parametric_eval \
  --backend vllm_api \
  --lora-path "${MERGED_DIR}" \
  --forget-path "${FORGET_PATH}" \
  --retain-path "${RETAIN_PATH}" \
  --forget-limit 400 \
  --retain-limit 3600 \
  --max-new-tokens 128 \
  --temperature 0.0 \
  --num-workers "${EVAL_WORKERS}" \
  --output-csv "${EVAL_DIR}/results.csv" \
  --output-json "${EVAL_DIR}/results.json" \
  --no-resume > "${LOG_DIR}/eval.log" 2>&1
kill_vllm
log "Generation finished"

start_vllm "${JUDGE_MODEL}" "judge" "${JUDGE_PORT}" "${LOG_DIR}/vllm_judge.log"
log "Judging outputs"
"${PY}" evaluation/llm_judge.py \
  --files "${EVAL_DIR}/results.csv" \
  --labels "${RUN_NAME}" \
  --api-url "http://127.0.0.1:${JUDGE_PORT}/v1" \
  --model judge \
  --output-dir "${EVAL_DIR}/judge_qwen25" > "${LOG_DIR}/judge.log" 2>&1
kill_vllm
log "Judge finished"

log "Final judge summary:"
cat "${EVAL_DIR}/judge_qwen25/judge_summary_custom.json" | tee -a "${MASTER_LOG}"
log "Run complete"
