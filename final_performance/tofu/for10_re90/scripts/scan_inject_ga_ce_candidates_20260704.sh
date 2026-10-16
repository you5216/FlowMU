#!/usr/bin/env bash
set -euo pipefail
trap 'status=$?; log "ERROR failed at line $LINENO status=$status"; exit $status' ERR

AGENT_DIR="/disk8T/lkh/python_projects/agent_unlearning_zhx/agent"
ENV52_PY="/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/env52/bin/python"
VLLM_BIN="/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/52vllm/bin/vllm"

ROOT="$AGENT_DIR/final_performance/tofu/for10_re90"
BASE_MODEL="$AGENT_DIR/merge_model/native_llm"
JUDGE_MODEL="/disk8T/lkh/python_projects/llms/Qwen2.5-VL-7B-Instruct"

LORA_ROOT="$ROOT/lora/ga_inject_rawforget_native_top20rr09_lr3e4_gn1_ce5_120_20260704/ce_threshold_checkpoints"
MERGE_ROOT="$ROOT/merged_model/ablation/inject_scan_20260704"
EVAL_ROOT="$ROOT/full_pipeline_eval/inject_ga_ce_quick_scan_20260704"
LOG_ROOT="$ROOT/logs/inject_ga_ce_quick_scan_20260704"

RAG_GRAPH="$ROOT/phase1_gat_retrain_gatselect_rag0925_memory0925_20260702/rag_unlearning_graph.pkl"
MEMORY_GRAPH="$ROOT/phase1_gat_retrain_gatselect_rag0925_memory0925_20260702/memory_unlearning_graph.pkl"
RETRIEVAL_CSV="$ROOT/retrieval_cache/final_gatselect_rag0925_memory0925_forget10_retain90_top50_20260702/graph_path_eval_results.csv"

FORGET_PATH="/disk8T/lkh/python_projects/datasets/TOFU/re_split/forget10.json"
RETAIN_PATH="/disk8T/lkh/python_projects/datasets/TOFU/re_split/retain90.json"

CANDIDATES="${CANDIDATES:-20 25 30 35}"
FORGET_LIMIT="${FORGET_LIMIT:-100}"
RETAIN_LIMIT="${RETAIN_LIMIT:-300}"
ANSWER_PORT="${ANSWER_PORT:-8002}"
JUDGE_PORT="${JUDGE_PORT:-8011}"
GPU_ID="${GPU_ID:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
JUDGE_WORKERS="${JUDGE_WORKERS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

mkdir -p "$MERGE_ROOT" "$EVAL_ROOT" "$LOG_ROOT"
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
    kill $pids 2>/dev/null || true
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

semantic_check() {
  local csv_path="$1"
  local out_json="$2"
  "$ENV52_PY" - "$csv_path" "$out_json" "$FORGET_LIMIT" "$RETAIN_LIMIT" <<'PY'
import csv
import json
import re
import statistics
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
out_json = Path(sys.argv[2])
expected_forget = int(sys.argv[3])
expected_retain = int(sys.argv[4])

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
        len(rows) == expected_forget + expected_retain
        and split_counts.get("forget", 0) == expected_forget
        and split_counts.get("retain", 0) == expected_retain
        and empty == 0
        and errors == 0
        and repeat_heavy == 0
        and short <= max(5, int(0.03 * len(rows)))
        and (statistics.mean(words) if words else 0.0) >= 10.0
    ),
}
out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
}

find_checkpoint() {
  local ce="$1"
  find "$LORA_ROOT" -maxdepth 1 -type d -name "ce${ce}_step*" | sort -V | tail -n 1
}

merge_candidate() {
  local ce="$1"
  local checkpoint="$2"
  local model_dir="$3"
  local log_dir="$LOG_ROOT/merge_ce${ce}"
  mkdir -p "$log_dir"

  if [[ -f "$model_dir/config.json" ]]; then
    log "CE${ce} merged model exists: $model_dir"
  else
    log "START merge CE${ce} checkpoint=$checkpoint -> $model_dir"
    "$ENV52_PY" -m phase2.pipeline.merge_lora_for_vllm \
      --base-model-path "$BASE_MODEL" \
      --lora-path "$checkpoint" \
      --output-dir "$model_dir" \
      --torch-dtype float16 \
      --device-map cpu \
      --label "tofu_for10_re90_ga_inject_rawforget_top20rr09_ce${ce}_quick_scan" \
      2>&1 | tee "$log_dir/merge.log"
    log "DONE merge CE${ce}"
  fi
}

start_answer_vllm() {
  local ce="$1"
  local model_dir="$2"
  local log_dir="$LOG_ROOT/answer_ce${ce}"
  mkdir -p "$log_dir"
  stop_port "$JUDGE_PORT"
  stop_port "$ANSWER_PORT"
  log "starting answer vLLM CE${ce} model=$model_dir gpu=$GPU_ID port=$ANSWER_PORT"
  nohup bash -lc "exec env CUDA_VISIBLE_DEVICES='$GPU_ID' VLLM_USE_V1=0 '$VLLM_BIN' serve '$model_dir' --host 0.0.0.0 --port '$ANSWER_PORT' --served-model-name 'inject_ce${ce}' --max-model-len 2048 --max-num-seqs 8 --gpu-memory-utilization 0.86 --enforce-eager --trust-remote-code" > "$log_dir/answer_vllm.log" 2>&1 &
  echo $! > "$log_dir/answer_vllm.pid"
  wait_server "$ANSWER_PORT" "$log_dir/answer_vllm.pid" "$log_dir/answer_vllm.log"
}

run_generation() {
  local ce="$1"
  local model_dir="$2"
  local run_name="flowmu_inject_ga_ce${ce}_quick_f${FORGET_LIMIT}_r${RETAIN_LIMIT}_20260704"
  local out_dir="$EVAL_ROOT/$run_name"
  local log_dir="$LOG_ROOT/$run_name"
  mkdir -p "$out_dir" "$log_dir"

  start_answer_vllm "$ce" "$model_dir"
  log "START quick generation CE${ce} out=$out_dir"
  VLLM_API_BASE="http://127.0.0.1:$ANSWER_PORT/v1" \
  VLLM_MODEL_NAME="inject_ce${ce}" \
  QWEN_INFER_BACKEND="vllm_api" \
  "$ENV52_PY" -m phase2.pipeline.step6_full_pipeline_eval \
    --base-model-path "$BASE_MODEL" \
    --lora-path "$model_dir" \
    --rag-graph-path "$RAG_GRAPH" \
    --memory-graph-path "$MEMORY_GRAPH" \
    --retrieval-results-csv "$RETRIEVAL_CSV" \
    --forget-path "$FORGET_PATH" \
    --retain-path "$RETAIN_PATH" \
    --forget-limit "$FORGET_LIMIT" \
    --retain-limit "$RETAIN_LIMIT" \
    --output-csv "$out_dir/results.csv" \
    --output-json "$out_dir/results.json" \
    --backend vllm_api \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature 0.0 \
    --num-workers "$NUM_WORKERS" \
    --rag-fallback-top-k 3 \
    --memory-max-hits 3 \
    --checkpoint-every 100 \
    --no-resume \
    2>&1 | tee "$log_dir/generation.log"

  semantic_check "$out_dir/results.csv" "$out_dir/semantic_sanity.json" 2>&1 | tee "$log_dir/semantic_sanity.log"
  stop_port "$ANSWER_PORT"
  log "DONE quick generation CE${ce}"
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

run_judge() {
  local files=()
  local labels=()
  for ce in $CANDIDATES; do
    local csv="$EVAL_ROOT/flowmu_inject_ga_ce${ce}_quick_f${FORGET_LIMIT}_r${RETAIN_LIMIT}_20260704/results.csv"
    if [[ -f "$csv" ]]; then
      files+=("$csv")
      labels+=("Inject_CE${ce}_quick")
    fi
  done
  if [[ "${#files[@]}" -eq 0 ]]; then
    log "ERROR no quick CSV files to judge"
    exit 3
  fi

  local judge_out="$EVAL_ROOT/judge_qwen25"
  local metrics_out="$EVAL_ROOT/final_metrics"
  local judge_log="$LOG_ROOT/judge"
  mkdir -p "$judge_out" "$metrics_out" "$judge_log"

  start_judge_vllm
  log "START judge quick candidates files=${#files[@]}"
  "$ENV52_PY" evaluation/llm_judge.py \
    --files "${files[@]}" \
    --labels "${labels[@]}" \
    --api-url "http://127.0.0.1:$JUDGE_PORT/v1" \
    --model judge \
    --output-dir "$judge_out" \
    --num-workers "$JUDGE_WORKERS" \
    2>&1 | tee "$judge_log/judge.log"

  local judged_files=()
  for label in "${labels[@]}"; do
    judged_files+=("$judge_out/judge_${label}.csv")
  done

  "$ENV52_PY" evaluation/final_metrics.py \
    --files "${judged_files[@]}" \
    --labels "${labels[@]}" \
    --output-dir "$metrics_out" \
    2>&1 | tee "$judge_log/final_metrics.log"
  log "DONE judge quick candidates; judge vLLM remains on port=$JUDGE_PORT pid=$(cat "$LOG_ROOT/judge_vllm/judge_vllm.pid")"
}

main() {
  for path in "$ENV52_PY" "$VLLM_BIN" "$BASE_MODEL" "$JUDGE_MODEL" "$RAG_GRAPH" "$MEMORY_GRAPH" "$RETRIEVAL_CSV" "$FORGET_PATH" "$RETAIN_PATH" "$LORA_ROOT"; do
    require_path "$path"
  done

  log "START Inject GA CE quick scan candidates=[$CANDIDATES] forget_limit=$FORGET_LIMIT retain_limit=$RETAIN_LIMIT"
  for ce in $CANDIDATES; do
    local checkpoint
    checkpoint="$(find_checkpoint "$ce")"
    [[ -n "$checkpoint" ]] || { log "ERROR missing CE${ce} checkpoint under $LORA_ROOT"; exit 2; }
    local model_dir
    model_dir="$MERGE_ROOT/tofu_for10_re90_ga_inject_rawforget_top20rr09_ce${ce}_20260704"
    merge_candidate "$ce" "$checkpoint" "$model_dir"
    run_generation "$ce" "$model_dir"
  done
  run_judge
  log "DONE Inject GA CE quick scan"
}

main "$@"
