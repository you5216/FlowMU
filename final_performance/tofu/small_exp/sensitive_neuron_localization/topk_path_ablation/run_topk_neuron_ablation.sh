#!/usr/bin/env bash
set -euo pipefail
trap 'status=$?; log "ERROR failed at line $LINENO status=$status"; exit $status' ERR

AGENT_DIR="${AGENT_DIR:-/home/likunhao/likunhao52shared/agent_unlearning_zhx/agent}"
ENV52_PY="${ENV52_PY:-/home/likunhao/likunhao52shared/agent_unlearning_zhx/environment/env50/bin/python}"
VLLM_BIN="${VLLM_BIN:-/home/likunhao/miniconda3/envs/nativeqwen310/bin/vllm}"
DATASET_ROOT="${DATASET_ROOT:-/home/likunhao/likunhao52shared/datasets/TOFU/re_split}"

SCRIPT_DIR="$AGENT_DIR/final_performance/tofu/small_exp/sensitive_neuron_localization/topk_path_ablation"
BASE_MODEL="${BASE_MODEL:-$AGENT_DIR/merge_model/native_llm}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR/runs_$RUN_TAG}"
TMP_MODEL_ROOT="${TMP_MODEL_ROOT:-$SCRIPT_DIR/tmp_models_$RUN_TAG}"
LOG_ROOT="${LOG_ROOT:-$SCRIPT_DIR/logs_$RUN_TAG}"

SPLITS="${SPLITS:-for05_re95}"
METHODS="${METHODS:-FlowMU Random}"
TOPKS="${TOPKS:-4 16 64 256 1024 4096 8192 12288}"
SEED="${SEED:-0}"
SCORE_KEY="${SCORE_KEY:-combined_raw_diff}"
MASK_MODE="${MASK_MODE:-drop}"
FORGET_LIMIT="${FORGET_LIMIT:-200}"
RETAIN_LIMIT="${RETAIN_LIMIT:-0}"
ONLY_FORGET="${ONLY_FORGET:-1}"
ONLY_RETAIN="${ONLY_RETAIN:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
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
      CANDIDATE_STATS="$AGENT_DIR/final_performance/tofu/for05_re95/phase2_localization_dualview_lambda03_top300_qaw2_20260622/candidate_neuron_stats.npz"
      RAG_GRAPH="$AGENT_DIR/phase1/dual_graph_unlearning_gat/path_supervision_runs/final_retain70_memory_path/rag_unlearning_graph.pkl"
      MEMORY_GRAPH="$AGENT_DIR/phase1/dual_graph_unlearning_gat/path_supervision_runs/final_retain70_memory_path/memory_unlearning_graph.pkl"
      RETRIEVAL_CSV="$AGENT_DIR/final_performance/tofu/for05_re95/retrieval_eval_graph_hybrid_top50_resplit_gr06_20260623/graph_path_eval_results.csv"
      FORGET_PATH="$DATASET_ROOT/forget05.json"
      RETAIN_PATH="$DATASET_ROOT/retain95.json"
      ;;
    for10_re90)
      CANDIDATE_STATS="$AGENT_DIR/final_performance/tofu/for10_re90/localization/tofu_native_gatselect_rag0925_memory0925_raw_top300_rr08_20260702/candidate_neuron_stats.npz"
      RAG_GRAPH="$AGENT_DIR/final_performance/tofu/for10_re90/phase1_gat_retrain_gatselect_rag0925_memory0925_20260702/rag_unlearning_graph.pkl"
      MEMORY_GRAPH="$AGENT_DIR/final_performance/tofu/for10_re90/phase1_gat_retrain_gatselect_rag0925_memory0925_20260702/memory_unlearning_graph.pkl"
      RETRIEVAL_CSV="$AGENT_DIR/final_performance/tofu/for10_re90/retrieval_cache/final_gatselect_rag0925_memory0925_forget10_retain90_top50_20260702/graph_path_eval_results.csv"
      FORGET_PATH="$DATASET_ROOT/forget10.json"
      RETAIN_PATH="$DATASET_ROOT/retain90.json"
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
  log "starting answer vLLM model=$model_path served=$served_name gpu=$GPU_ID port=$ANSWER_PORT"
  setsid bash -lc "exec env CUDA_VISIBLE_DEVICES='$GPU_ID' VLLM_USE_V1=0 '$VLLM_BIN' serve '$model_path' --host 0.0.0.0 --port '$ANSWER_PORT' --served-model-name '$served_name' --max-model-len 2048 --max-num-seqs 8 --gpu-memory-utilization 0.86 --enforce-eager --trust-remote-code" > "$log_dir/answer_vllm.log" 2>&1 < /dev/null &
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
  local retain_eval_path="$RETAIN_PATH"
  local retain_eval_limit="$RETAIN_LIMIT"
  local forget_eval_path="$FORGET_PATH"
  local forget_eval_limit="$FORGET_LIMIT"
  if [[ "$ONLY_FORGET" == "1" ]]; then
    retain_eval_path="$RUN_ROOT/empty_retain.json"
    printf '[]\n' > "$retain_eval_path"
    retain_eval_limit=0
  fi
  if [[ "$ONLY_RETAIN" == "1" ]]; then
    forget_eval_path="$RUN_ROOT/empty_forget.json"
    printf '[]\n' > "$forget_eval_path"
    forget_eval_limit=0
  fi

  if [[ -s "$out_dir/results.csv" && "$NO_RESUME" != "1" ]]; then
    log "skip existing results: $out_dir/results.csv"
    return 0
  fi
  if [[ "$NO_RESUME" == "1" ]]; then
    rm -f "$out_dir/results.csv" "$out_dir/results.json"
  fi

  log "START split=$split method=$method mask_mode=$MASK_MODE topk=$topk seed=$SEED forget_limit=$FORGET_LIMIT retain_limit=$RETAIN_LIMIT"
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
