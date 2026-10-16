#!/usr/bin/env bash
set -euo pipefail
trap 'status=$?; log "ERROR failed at line $LINENO status=$status"; exit $status' ERR

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
AGENT_DIR="${AGENT_DIR:-$(cd "$SCRIPT_DIR/../../../../../.." && pwd -P)}"
ENV52_PY="${ENV52_PY:-/home/likunhao/likunhao52shared/agent_unlearning_zhx/environment/env50/bin/python}"
VLLM_BIN="${VLLM_BIN:-/home/likunhao/miniconda3/envs/nativeqwen310/bin/vllm}"
DATASET_ROOT="${DATASET_ROOT:-/home/likunhao/likunhao52shared/datasets}"

BASE_MODEL="${BASE_MODEL:-$AGENT_DIR/final_performance/literaryQA/base/lora/merged/literary_neative_model}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR/runs_$RUN_TAG}"
TMP_MODEL_ROOT="${TMP_MODEL_ROOT:-$SCRIPT_DIR/tmp_models_$RUN_TAG}"
LOG_ROOT="${LOG_ROOT:-$SCRIPT_DIR/logs_$RUN_TAG}"

SPLITS="${SPLITS:-for05_re95}"
METHODS="${METHODS:-FlowMU Random}"
TOPKS="${TOPKS:-2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 12288}"
SEED="${SEED:-0}"
SCORE_KEY="${SCORE_KEY:-combined_raw_diff}"
MASK_MODE="${MASK_MODE:-drop}"
FORGET_LIMIT="${FORGET_LIMIT:-200}"
RETAIN_LIMIT="${RETAIN_LIMIT:-0}"
ONLY_FORGET="${ONLY_FORGET:-1}"
ONLY_RETAIN="${ONLY_RETAIN:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.86}"
ANSWER_PORT="${ANSWER_PORT:-8002}"
GPU_ID="${GPU_ID:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VLLM_API_TIMEOUT="${VLLM_API_TIMEOUT:-180}"
KEEP_MASKED_MODELS="${KEEP_MASKED_MODELS:-0}"
NO_RESUME="${NO_RESUME:-0}"
NEURON_DIM="${NEURON_DIM:-12288}"

case "$MASK_MODE" in
  drop|keep) ;;
  *)
    echo "ERROR MASK_MODE must be 'drop' or 'keep', got: $MASK_MODE" >&2
    exit 2
    ;;
esac

mkdir -p "$RUN_ROOT" "$TMP_MODEL_ROOT" "$LOG_ROOT"
cd "$AGENT_DIR"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG_ROOT/runner.log"
}

require_path() {
  local path="$1"
  [[ -e "$path" ]] || { log "ERROR missing required path: $path"; exit 2; }
}

stop_port() {
  local port="$1"
  local pids
  set +e
  pids=$(ss -ltnp "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | sort -u | tr '\n' ' ')
  set -e
  if [[ -n "${pids// }" ]]; then
    log "stopping port=$port pids=$pids"
    set +e
    kill $pids 2>/dev/null
    set -e
    sleep 8
  fi
}

wait_server() {
  local port="$1"
  local pid_file="$2"
  local log_file="$3"
  for i in $(seq 1 240); do
    if curl -fsS "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
      log "server ready port=$port waited=$((i * 5))s"
      return 0
    fi
    if [[ -f "$pid_file" ]] && ! ps -p "$(cat "$pid_file")" >/dev/null 2>&1; then
      log "ERROR server on port=$port exited"
      tail -160 "$log_file" | tee -a "$LOG_ROOT/runner.log" || true
      return 1
    fi
    if (( i % 12 == 0 )); then
      log "waiting server port=$port waited=$((i * 5))s"
      tail -8 "$log_file" | tee -a "$LOG_ROOT/runner.log" || true
    fi
    sleep 5
  done
  log "ERROR server on port=$port not ready"
  tail -160 "$log_file" | tee -a "$LOG_ROOT/runner.log" || true
  return 1
}

split_config() {
  local split="$1"
  case "$split" in
    for05_re95)
      CANDIDATE_STATS="$AGENT_DIR/final_performance/literaryQA/for05_re95/localization/literary_neative_model_rag_memory_final_combined_top300_rr08_20260701/candidate_neuron_stats.npz"
      RAG_GRAPH="$AGENT_DIR/final_performance/literaryQA/for05_re95/rag_memory_unlearning_graph/rag_unlearning_graph.pkl"
      MEMORY_GRAPH="$AGENT_DIR/final_performance/literaryQA/for05_re95/rag_memory_unlearning_graph/memory_unlearning_graph.pkl"
      RETRIEVAL_CSV="$AGENT_DIR/final_performance/literaryQA/for05_re95/retrieval_cache/final_graph_full_forget992_retain18069_top50_20260629/graph_path_eval_results.csv"
      FORGET_PATH="${FORGET_PATH:-$DATASET_ROOT/LiteraryQA/literaryqa_annotations/rag/forget05.json}"
      RETAIN_PATH="${RETAIN_PATH:-$DATASET_ROOT/LiteraryQA/literaryqa_annotations/rag/retain95.json}"
      ;;
    *)
      log "ERROR unsupported split for this driver: $split"
      log "Add its candidate stats, graphs, retrieval cache, forget and retain paths in split_config()."
      exit 2
      ;;
  esac
}

start_answer_vllm() {
  local model_path="$1"
  local served_name="$2"
  local log_dir="$3"
  mkdir -p "$log_dir"
  stop_port "$ANSWER_PORT"
  log "starting answer vLLM model=$model_path served=$served_name gpu=$GPU_ID port=$ANSWER_PORT gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
  setsid bash -lc "exec env CUDA_VISIBLE_DEVICES='$GPU_ID' '$VLLM_BIN' serve '$model_path' --host 0.0.0.0 --port '$ANSWER_PORT' --served-model-name '$served_name' --max-model-len '$MAX_MODEL_LEN' --max-num-seqs '$MAX_NUM_SEQS' --gpu-memory-utilization '$GPU_MEMORY_UTILIZATION' --enforce-eager --trust-remote-code --limit-mm-per-prompt '{\"image\":0,\"video\":0}' --mm-processor-cache-gb 0" > "$log_dir/answer_vllm.log" 2>&1 < /dev/null &
  echo $! > "$log_dir/answer_vllm.pid"
  wait_server "$ANSWER_PORT" "$log_dir/answer_vllm.pid" "$log_dir/answer_vllm.log"
}

build_model_if_needed() {
  local split="$1"
  local method="$2"
  local topk="$3"
  local model_dir="$TMP_MODEL_ROOT/${split}_${MASK_MODE}_${method}_topk${topk}_seed${SEED}"
  if [[ "$MASK_MODE" == "keep" ]] && (( topk >= NEURON_DIM )); then
    echo "$BASE_MODEL"
    return 0
  fi
  "$ENV52_PY" "$SCRIPT_DIR/build_topk_masked_model.py" \
    --base-model "$BASE_MODEL" \
    --candidate-stats "$CANDIDATE_STATS" \
    --output-model-dir "$model_dir" \
    --method "$method" \
    --topk "$topk" \
    --mask-mode "$MASK_MODE" \
    --seed "$SEED" \
    --score-key "$SCORE_KEY" \
    --overwrite \
    2>&1 | tee "$LOG_ROOT/build_${split}_${MASK_MODE}_${method}_topk${topk}_seed${SEED}.log"
  echo "$model_dir"
}

run_one() {
  local split="$1"
  local method="$2"
  local topk="$3"
  split_config "$split"
  for path in "$ENV52_PY" "$VLLM_BIN" "$BASE_MODEL" "$CANDIDATE_STATS" "$RAG_GRAPH" "$MEMORY_GRAPH" "$RETRIEVAL_CSV" "$FORGET_PATH" "$RETAIN_PATH"; do
    require_path "$path"
  done

  local out_dir="$RUN_ROOT/$split/${method}_${MASK_MODE}_topk${topk}_seed${SEED}"
  local log_dir="$LOG_ROOT/$split/${method}_${MASK_MODE}_topk${topk}_seed${SEED}"
  mkdir -p "$out_dir" "$log_dir"
  local forget_eval_path="$FORGET_PATH"
  local forget_eval_limit="$FORGET_LIMIT"
  local retain_eval_path="$RETAIN_PATH"
  local retain_eval_limit="$RETAIN_LIMIT"
  if [[ "$ONLY_RETAIN" == "1" ]]; then
    forget_eval_path="$RUN_ROOT/empty_forget.json"
    printf '[]\n' > "$forget_eval_path"
    forget_eval_limit=0
  fi
  if [[ "$ONLY_FORGET" == "1" ]]; then
    retain_eval_path="$RUN_ROOT/empty_retain.json"
    printf '[]\n' > "$retain_eval_path"
    retain_eval_limit=0
  fi

  if [[ -s "$out_dir/results.csv" && "$NO_RESUME" != "1" ]]; then
    log "skip existing results: $out_dir/results.csv"
    return 0
  fi
  if [[ "$NO_RESUME" == "1" ]]; then
    rm -f "$out_dir/results.csv" "$out_dir/results.json"
  fi

  log "START split=$split method=$method mask_mode=$MASK_MODE topk=$topk seed=$SEED forget_limit=$forget_eval_limit retain_limit=$retain_eval_limit"
  local model_path
  model_path="$(build_model_if_needed "$split" "$method" "$topk" | tail -n 1)"
  local served_name="topk_${split}_${MASK_MODE}_${method}_${topk}_${SEED}"
  start_answer_vllm "$model_path" "$served_name" "$log_dir"

  VLLM_API_BASE="http://127.0.0.1:$ANSWER_PORT/v1" \
  VLLM_MODEL_NAME="$served_name" \
  QWEN_INFER_BACKEND="vllm_api" \
  VLLM_API_TIMEOUT="$VLLM_API_TIMEOUT" \
  "$ENV52_PY" -m phase2.pipeline.step6_full_pipeline_eval \
    --base-model-path "$BASE_MODEL" \
    --lora-path "$model_path" \
    --rag-graph-path "$RAG_GRAPH" \
    --memory-graph-path "$MEMORY_GRAPH" \
    --retrieval-results-csv "$RETRIEVAL_CSV" \
    --forget-path "$forget_eval_path" \
    --retain-path "$retain_eval_path" \
    --forget-limit "$forget_eval_limit" \
    --retain-limit "$retain_eval_limit" \
    --output-csv "$out_dir/results.csv" \
    --output-json "$out_dir/results.json" \
    --backend vllm_api \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature 0.0 \
    --num-workers "$NUM_WORKERS" \
    --rag-fallback-top-k 3 \
    --memory-max-hits 3 \
    --checkpoint-every 20 \
    --no-resume \
    2>&1 | tee "$log_dir/generation.log"

  stop_port "$ANSWER_PORT"
  if [[ "$KEEP_MASKED_MODELS" != "1" && "$model_path" == "$TMP_MODEL_ROOT"* ]]; then
    log "remove temporary masked model: $model_path"
    rm -rf "$model_path"
  fi
  log "DONE split=$split method=$method mask_mode=$MASK_MODE topk=$topk"
}

summarize() {
  "$ENV52_PY" "$SCRIPT_DIR/summarize_topk_neuron_ablation.py" \
    --run-root "$RUN_ROOT" \
    --output-csv "$SCRIPT_DIR/real_topk_neuron_ablation_${RUN_TAG}.csv"
}

status() {
  log "status RUN_ROOT=$RUN_ROOT"
  find "$RUN_ROOT" -path '*/results.csv' -print | sort || true
  if compgen -G "$SCRIPT_DIR/real_topk_neuron_ablation_${RUN_TAG}.csv" >/dev/null; then
    cat "$SCRIPT_DIR/real_topk_neuron_ablation_${RUN_TAG}.csv"
  fi
}

run_grid() {
  for split in $SPLITS; do
    for topk in $TOPKS; do
      for method in $METHODS; do
        if (( topk >= NEURON_DIM )) && [[ "$method" == "Random" ]]; then
          log "skip duplicate all-neuron point: split=$split method=$method mask_mode=$MASK_MODE topk=$topk"
          continue
        fi
        run_one "$split" "$method" "$topk"
        summarize || true
      done
    done
  done
  summarize
}

usage() {
  cat <<EOF
Usage: $0 <command>

Commands:
  run-one <split> <method> <topk>   Run one generation job.
  run-grid                          Run SPLITS x TOPKS x METHODS.
  summarize                         Summarize existing results under RUN_ROOT.
  status                            Print generated CSVs and summary if present.
  stop                              Stop answer vLLM on ANSWER_PORT.

Default smoke environment:
  SPLITS=$SPLITS
  METHODS=$METHODS
  TOPKS=$TOPKS
  MASK_MODE=$MASK_MODE
  FORGET_LIMIT=$FORGET_LIMIT RETAIN_LIMIT=$RETAIN_LIMIT
  ONLY_FORGET=$ONLY_FORGET
  ONLY_RETAIN=$ONLY_RETAIN
  RUN_ROOT=$RUN_ROOT
EOF
}

main() {
  local command="${1:-}"
  case "$command" in
    run-one)
      [[ $# -eq 4 ]] || { usage; exit 2; }
      run_one "$2" "$3" "$4"
      summarize
      ;;
    run-grid)
      run_grid
      ;;
    summarize)
      summarize
      ;;
    status)
      status
      ;;
    stop)
      stop_port "$ANSWER_PORT"
      ;;
    *)
      usage
      exit 2
      ;;
  esac
}

main "$@"
