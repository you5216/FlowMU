#!/usr/bin/env bash
set -euo pipefail
trap 'status=$?; log "ERROR failed at line $LINENO status=$status"; exit $status' ERR

AGENT_DIR="/disk8T/lkh/python_projects/agent_unlearning_zhx/agent"
ENV52_PY="/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/env52/bin/python"
VLLM_BIN="/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/52vllm/bin/vllm"

ROOT="$AGENT_DIR/final_performance/tofu/for10_re90"
BASE_MODEL="$AGENT_DIR/merge_model/native_llm"
JUDGE_MODEL="/disk8T/lkh/python_projects/llms/Qwen2.5-VL-7B-Instruct"

MAIN_RAG_GRAPH="$ROOT/phase1_gat_retrain_gatselect_rag0925_memory0925_20260702/rag_unlearning_graph.pkl"
MAIN_MEMORY_GRAPH="$ROOT/phase1_gat_retrain_gatselect_rag0925_memory0925_20260702/memory_unlearning_graph.pkl"
SENSOR_RAG_GRAPH="$ROOT/phase1_similarity_sensor_rag560_memory1255_20260704/rag_unlearning_graph.pkl"
SENSOR_MEMORY_GRAPH="$ROOT/phase1_similarity_sensor_rag560_memory1255_20260704/memory_unlearning_graph.pkl"
RETRIEVAL_CSV="$ROOT/retrieval_cache/final_gatselect_rag0925_memory0925_forget10_retain90_top50_20260702/graph_path_eval_results.csv"

FORGET_PATH="/disk8T/lkh/python_projects/datasets/TOFU/re_split/forget10.json"
RETAIN_PATH="/disk8T/lkh/python_projects/datasets/TOFU/re_split/retain90.json"

INJECT_MODEL="${INJECT_MODEL:-$ROOT/merged_model/ablation/tofu_for10_re90_ga_inject_rawforget_top20rr09_ce60_20260704}"
SENSOR_MODEL="${SENSOR_MODEL:-$ROOT/merged_model/ablation/tofu_for10_re90_ga_similarity_sensor_top20rr09_ce60_20260704}"

RUN_TAG="${RUN_TAG:-20260704}"
LOG_ROOT="$ROOT/logs/flowmu_inject_sensor_ablation_for10_${RUN_TAG}"
JUDGE_OUT="$ROOT/full_pipeline_eval/judge_flowmu_inject_sensor_ablation_for10_${RUN_TAG}"

ANSWER_PORT="${ANSWER_PORT:-8002}"
JUDGE_PORT="${JUDGE_PORT:-8011}"
GPU_ID="${GPU_ID:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
JUDGE_WORKERS="${JUDGE_WORKERS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100}"
VLLM_API_TIMEOUT="${VLLM_API_TIMEOUT:-180}"
NO_RESUME="${NO_RESUME:-0}"

INJECT_RUN_NAME="${INJECT_RUN_NAME:-flowmu_inject_rawforget_ga_ce60_full4000_ablation_${RUN_TAG}}"
SENSOR_RUN_NAME="${SENSOR_RUN_NAME:-flowmu_sensor_similarity_ga_ce60_full4000_ablation_${RUN_TAG}}"
INJECT_JUDGE_LABEL="${INJECT_JUDGE_LABEL:-FlowMU_Inject_for10_GA_CE60}"
SENSOR_JUDGE_LABEL="${SENSOR_JUDGE_LABEL:-FlowMU_Sensor_for10_GA_CE60}"
INJECT_METRIC_LABEL="${INJECT_METRIC_LABEL:-FlowMU-Inject}"
SENSOR_METRIC_LABEL="${SENSOR_METRIC_LABEL:-FlowMU-Sensor}"
INJECT_OUT="$ROOT/full_pipeline_eval/$INJECT_RUN_NAME"
SENSOR_OUT="$ROOT/full_pipeline_eval/$SENSOR_RUN_NAME"

mkdir -p "$LOG_ROOT"
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
      tail -180 "$log_file" | tee -a "$LOG_ROOT/runner.log" || true
      return 1
    fi
    if (( i % 12 == 0 )); then
      log "waiting server port=$port waited=$((i * 5))s"
      tail -8 "$log_file" | tee -a "$LOG_ROOT/runner.log" || true
    fi
    sleep 5
  done
  log "ERROR server on port=$port not ready"
  tail -180 "$log_file" | tee -a "$LOG_ROOT/runner.log" || true
  return 1
}

semantic_check() {
  local csv_path="$1"
  local out_json="$2"
  "$ENV52_PY" - "$csv_path" "$out_json" <<'PY'
import csv
import json
import re
import statistics
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
out_json = Path(sys.argv[2])
rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
words = []
empty = 0
errors = 0
short = 0
repeat_heavy = 0
split_counts = {}

for row in rows:
    split = row.get("split", "")
    split_counts[split] = split_counts.get(split, 0) + 1
    pred = (row.get("prediction") or "").strip()
    if not pred:
        empty += 1
    if "[EVAL_ERROR]" in pred:
        errors += 1
    toks = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", pred)
    words.append(len(toks))
    if len(toks) < 3:
        short += 1
    if re.search(r"(.)\1{20,}", pred):
        repeat_heavy += 1

summary = {
    "rows": len(rows),
    "split_counts": split_counts,
    "empty": empty,
    "eval_error": errors,
    "short_lt3_words": short,
    "repeat_heavy_gt20": repeat_heavy,
    "avg_words": round(statistics.mean(words), 4) if words else 0.0,
    "semantic_ok": (
        len(rows) == 4000
        and split_counts.get("forget", 0) == 400
        and split_counts.get("retain", 0) == 3600
        and empty == 0
        and errors == 0
        and repeat_heavy <= 100
    ),
}
out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
if not summary["semantic_ok"]:
    sys.exit(3)
PY
}

require_common() {
  for path in \
    "$ENV52_PY" \
    "$VLLM_BIN" \
    "$BASE_MODEL" \
    "$JUDGE_MODEL" \
    "$RETRIEVAL_CSV" \
    "$FORGET_PATH" \
    "$RETAIN_PATH"; do
    require_path "$path"
  done
}

start_answer_vllm() {
  local model_path="$1"
  local served_name="$2"
  local log_dir="$3"
  require_path "$model_path"
  mkdir -p "$log_dir"
  stop_port "$JUDGE_PORT"
  stop_port "$ANSWER_PORT"
  log "starting answer vLLM model=$model_path served=$served_name gpu=$GPU_ID port=$ANSWER_PORT"
  nohup bash -lc "exec env CUDA_VISIBLE_DEVICES='$GPU_ID' VLLM_USE_V1=0 '$VLLM_BIN' serve '$model_path' --host 0.0.0.0 --port '$ANSWER_PORT' --served-model-name '$served_name' --max-model-len 2048 --max-num-seqs 8 --gpu-memory-utilization 0.86 --enforce-eager --trust-remote-code" > "$log_dir/answer_vllm.log" 2>&1 &
  echo $! > "$log_dir/answer_vllm.pid"
  wait_server "$ANSWER_PORT" "$log_dir/answer_vllm.pid" "$log_dir/answer_vllm.log"
}

start_judge_vllm() {
  local log_dir="$LOG_ROOT/judge_vllm"
  mkdir -p "$log_dir"
  stop_port "$ANSWER_PORT"
  stop_port "$JUDGE_PORT"
  log "starting judge vLLM gpu=$GPU_ID port=$JUDGE_PORT"
  nohup bash -lc "exec env CUDA_VISIBLE_DEVICES='$GPU_ID' '$VLLM_BIN' serve '$JUDGE_MODEL' --host 0.0.0.0 --port '$JUDGE_PORT' --served-model-name judge --max-model-len 2048 --max-num-seqs 2 --gpu-memory-utilization 0.90 --enforce-eager --trust-remote-code --generation-config vllm" > "$log_dir/judge_vllm.log" 2>&1 &
  echo $! > "$log_dir/judge_vllm.pid"
  wait_server "$JUDGE_PORT" "$log_dir/judge_vllm.pid" "$log_dir/judge_vllm.log"
}

run_generation() {
  local label="$1"
  local run_name="$2"
  local served_name="$3"
  local model_path="$4"
  local rag_graph="$5"
  local memory_graph="$6"
  shift 6

  local out_dir="$ROOT/full_pipeline_eval/$run_name"
  local log_dir="$LOG_ROOT/$run_name"
  mkdir -p "$out_dir" "$log_dir"
  require_path "$model_path"
  require_path "$rag_graph"
  require_path "$memory_graph"

  local resume_args=()
  if [[ "$NO_RESUME" == "1" ]]; then
    resume_args+=(--no-resume)
    rm -f "$out_dir/results.csv" "$out_dir/results.json"
  fi

  start_answer_vllm "$model_path" "$served_name" "$log_dir"
  log "START generation label=$label out=$out_dir"
  VLLM_API_BASE="http://127.0.0.1:$ANSWER_PORT/v1" \
  VLLM_MODEL_NAME="$served_name" \
  QWEN_INFER_BACKEND="vllm_api" \
  VLLM_API_TIMEOUT="$VLLM_API_TIMEOUT" \
  "$ENV52_PY" -m phase2.pipeline.step6_full_pipeline_eval \
    --base-model-path "$BASE_MODEL" \
    --lora-path "$model_path" \
    --rag-graph-path "$rag_graph" \
    --memory-graph-path "$memory_graph" \
    --retrieval-results-csv "$RETRIEVAL_CSV" \
    --forget-path "$FORGET_PATH" \
    --retain-path "$RETAIN_PATH" \
    --forget-limit 0 \
    --retain-limit 0 \
    --output-csv "$out_dir/results.csv" \
    --output-json "$out_dir/results.json" \
    --backend vllm_api \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature 0.0 \
    --num-workers "$NUM_WORKERS" \
    --rag-fallback-top-k 3 \
    --memory-max-hits 3 \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    "${resume_args[@]}" \
    "$@" \
    2>&1 | tee "$log_dir/generation.log"

  semantic_check "$out_dir/results.csv" "$out_dir/semantic_sanity.json" 2>&1 | tee "$log_dir/semantic_sanity.log"
  stop_port "$ANSWER_PORT"
  log "DONE generation label=$label"
}

generate_inject() {
  run_generation \
    "FlowMU-Inject raw-forget localization" \
    "$INJECT_RUN_NAME" \
    "flowmu_inject" \
    "$INJECT_MODEL" \
    "$MAIN_RAG_GRAPH" \
    "$MAIN_MEMORY_GRAPH"
}

generate_sensor() {
  run_generation \
    "FlowMU-Sensor similarity-only subgraph" \
    "$SENSOR_RUN_NAME" \
    "flowmu_sensor" \
    "$SENSOR_MODEL" \
    "$SENSOR_RAG_GRAPH" \
    "$SENSOR_MEMORY_GRAPH"
}

run_judge() {
  require_path "$INJECT_OUT/results.csv"
  require_path "$SENSOR_OUT/results.csv"
  local judge_log="$LOG_ROOT/judge"
  mkdir -p "$JUDGE_OUT/judge_qwen25" "$JUDGE_OUT/final_metrics" "$judge_log"
  start_judge_vllm
  log "START judge Inject/Sensor ablations"
  "$ENV52_PY" evaluation/llm_judge.py \
    --files "$INJECT_OUT/results.csv" "$SENSOR_OUT/results.csv" \
    --labels "$INJECT_JUDGE_LABEL" "$SENSOR_JUDGE_LABEL" \
    --api-url "http://127.0.0.1:$JUDGE_PORT/v1" \
    --model judge \
    --output-dir "$JUDGE_OUT/judge_qwen25" \
    --num-workers "$JUDGE_WORKERS" \
    2>&1 | tee "$judge_log/judge.log"

  "$ENV52_PY" evaluation/final_metrics.py \
    --files \
      "$JUDGE_OUT/judge_qwen25/judge_$INJECT_JUDGE_LABEL.csv" \
      "$JUDGE_OUT/judge_qwen25/judge_$SENSOR_JUDGE_LABEL.csv" \
    --labels "$INJECT_METRIC_LABEL" "$SENSOR_METRIC_LABEL" \
    --output-dir "$JUDGE_OUT/final_metrics" \
    2>&1 | tee "$judge_log/final_metrics.log"
  log "DONE judge; judge vLLM remains on port=$JUDGE_PORT pid=$(cat "$LOG_ROOT/judge_vllm/judge_vllm.pid")"
}

status() {
  log "status"
  for csv in "$INJECT_OUT/results.csv" "$SENSOR_OUT/results.csv"; do
    if [[ -s "$csv" ]]; then
      "$ENV52_PY" - "$csv" <<'PY'
import csv
import sys
from pathlib import Path
p = Path(sys.argv[1])
rows = list(csv.DictReader(p.open(encoding="utf-8")))
split_counts = {}
errors = 0
for row in rows:
    split_counts[row.get("split", "")] = split_counts.get(row.get("split", ""), 0) + 1
    errors += int("[EVAL_ERROR]" in (row.get("prediction") or ""))
print(f"{p}: rows={len(rows)} split_counts={split_counts} eval_errors={errors}")
PY
    else
      echo "$csv: missing"
    fi
  done
  if [[ -s "$JUDGE_OUT/final_metrics/final_metrics_summary.csv" ]]; then
    echo "final metrics: $JUDGE_OUT/final_metrics/final_metrics_summary.csv"
    cat "$JUDGE_OUT/final_metrics/final_metrics_summary.csv"
  fi
}

usage() {
  cat <<EOF
Usage: $0 <command>

Commands:
  generate-inject   Generate FlowMU-Inject full pipeline only.
  generate-sensor   Generate FlowMU-Sensor full pipeline only.
  generate          Generate both ablations; no judge.
  judge             Judge both generated CSVs and compute final metrics.
  all               Generate both ablations, then judge.
  status            Print generation/judge status.
  stop              Stop answer/judge vLLM ports used by this script.

Environment:
  GPU_ID=$GPU_ID ANSWER_PORT=$ANSWER_PORT JUDGE_PORT=$JUDGE_PORT NUM_WORKERS=$NUM_WORKERS
  NO_RESUME=1 forces regeneration; default resumes from partial CSVs.
EOF
}

main() {
  local command="${1:-}"
  case "$command" in
    generate-inject)
      require_common
      require_path "$MAIN_RAG_GRAPH"
      require_path "$MAIN_MEMORY_GRAPH"
      generate_inject
      ;;
    generate-sensor)
      require_common
      require_path "$SENSOR_RAG_GRAPH"
      require_path "$SENSOR_MEMORY_GRAPH"
      generate_sensor
      ;;
    generate)
      require_common
      require_path "$MAIN_RAG_GRAPH"
      require_path "$MAIN_MEMORY_GRAPH"
      require_path "$SENSOR_RAG_GRAPH"
      require_path "$SENSOR_MEMORY_GRAPH"
      generate_inject
      generate_sensor
      ;;
    judge)
      require_common
      run_judge
      ;;
    all)
      require_common
      require_path "$MAIN_RAG_GRAPH"
      require_path "$MAIN_MEMORY_GRAPH"
      require_path "$SENSOR_RAG_GRAPH"
      require_path "$SENSOR_MEMORY_GRAPH"
      generate_inject
      generate_sensor
      run_judge
      ;;
    status)
      status
      ;;
    stop)
      stop_port "$ANSWER_PORT"
      stop_port "$JUDGE_PORT"
      ;;
    *)
      usage
      exit 2
      ;;
  esac
}

main "$@"
