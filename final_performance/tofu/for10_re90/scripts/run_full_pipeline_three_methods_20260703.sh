#!/usr/bin/env bash
set -euo pipefail

AGENT_DIR="/disk8T/lkh/python_projects/agent_unlearning_zhx/agent"
ENV52_PY="/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/env52/bin/python"
VLLM_BIN="/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/52vllm/bin/vllm"
BASE_MODEL="$AGENT_DIR/merge_model/native_llm"
JUDGE_MODEL="/disk8T/lkh/python_projects/llms/Qwen2.5-VL-7B-Instruct"
ROOT="$AGENT_DIR/final_performance/tofu/for10_re90"
LOG_ROOT="$ROOT/logs/full_pipeline_three_methods_20260703"
RAG_GRAPH="$ROOT/phase1_gat_retrain_gatselect_rag0925_memory0925_20260702/rag_unlearning_graph.pkl"
MEMORY_GRAPH="$ROOT/phase1_gat_retrain_gatselect_rag0925_memory0925_20260702/memory_unlearning_graph.pkl"
RETRIEVAL_CSV="$ROOT/retrieval_cache/final_gatselect_rag0925_memory0925_forget10_retain90_top50_20260702/graph_path_eval_results.csv"
FORGET_PATH="/disk8T/lkh/python_projects/datasets/TOFU/re_split/forget10.json"
RETAIN_PATH="/disk8T/lkh/python_projects/datasets/TOFU/re_split/retain90.json"
ANSWER_PORT=8002
JUDGE_PORT=8003

mkdir -p "$LOG_ROOT"
cd "$AGENT_DIR"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG_ROOT/runner.log"
}

stop_port() {
  local port="$1"
  local pids
  pids=$(ss -ltnp "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | sort -u | tr '\n' ' ' || true)
  if [[ -n "${pids// }" ]]; then
    log "stopping port $port pids: $pids"
    kill $pids 2>/dev/null || true
    sleep 8
  fi
}

wait_server() {
  local port="$1"
  local pid_file="$2"
  local log_file="$3"
  for i in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
      log "server_ready port=$port waited=$((i*5))s"
      return 0
    fi
    if [[ -f "$pid_file" ]] && ! ps -p "$(cat "$pid_file")" >/dev/null 2>&1; then
      log "ERROR server on port $port exited"
      tail -180 "$log_file" | tee -a "$LOG_ROOT/runner.log" || true
      return 1
    fi
    if (( i % 12 == 0 )); then
      log "waiting server port=$port $((i*5))s"
      tail -8 "$log_file" | tee -a "$LOG_ROOT/runner.log" || true
    fi
    sleep 5
  done
  log "ERROR server on port $port not ready"
  tail -180 "$log_file" | tee -a "$LOG_ROOT/runner.log" || true
  return 1
}

semantic_check() {
  local csv_path="$1"
  local out_json="$2"
  "$ENV52_PY" - "$csv_path" "$out_json" <<'PY'
import csv, json, re, statistics, sys
from pathlib import Path
csv_path = Path(sys.argv[1])
out_json = Path(sys.argv[2])
rows = list(csv.DictReader(csv_path.open(encoding='utf-8')))
words = []
empty = 0
err = 0
short = 0
repeat_heavy = 0
split_counts = {}
for r in rows:
    split_counts[r.get('split','')] = split_counts.get(r.get('split',''), 0) + 1
    pred = (r.get('prediction') or '').strip()
    if not pred:
        empty += 1
    if '[EVAL_ERROR]' in pred:
        err += 1
    toks = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", pred)
    wc = len(toks)
    words.append(wc)
    if wc < 3:
        short += 1
    if re.search(r"(.)\1{20,}", pred):
        repeat_heavy += 1
summary = {
    "rows": len(rows),
    "split_counts": split_counts,
    "empty": empty,
    "eval_error": err,
    "short_lt3_words": short,
    "repeat_heavy_gt20": repeat_heavy,
    "avg_words": round(statistics.mean(words), 4) if words else 0.0,
    "semantic_ok": len(rows) == 4000 and split_counts.get('forget', 0) == 400 and split_counts.get('retain', 0) == 3600 and empty == 0 and err == 0 and repeat_heavy <= 100,
}
out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
if not summary["semantic_ok"]:
    sys.exit(3)
PY
}

run_generation() {
  local key="$1"
  local served_name="$2"
  local model_path="$3"
  local run_name="$4"
  local out_dir="$ROOT/full_pipeline_eval/$run_name"
  local log_dir="$LOG_ROOT/$run_name"
  mkdir -p "$out_dir" "$log_dir"
  log "START generation key=$key model=$model_path out=$out_dir"
  stop_port "$ANSWER_PORT"
  nohup bash -lc "exec env CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=0 '$VLLM_BIN' serve '$model_path' --host 0.0.0.0 --port $ANSWER_PORT --served-model-name '$served_name' --max-model-len 2048 --max-num-seqs 8 --gpu-memory-utilization 0.86 --enforce-eager --trust-remote-code" > "$log_dir/answer_vllm.log" 2>&1 &
  echo $! > "$log_dir/answer_vllm.pid"
  wait_server "$ANSWER_PORT" "$log_dir/answer_vllm.pid" "$log_dir/answer_vllm.log"

  VLLM_API_BASE="http://127.0.0.1:$ANSWER_PORT/v1" VLLM_MODEL_NAME="$served_name" QWEN_INFER_BACKEND="vllm_api" \
  "$ENV52_PY" -m phase2.pipeline.step6_full_pipeline_eval \
    --base-model-path "$BASE_MODEL" \
    --lora-path "$model_path" \
    --rag-graph-path "$RAG_GRAPH" \
    --memory-graph-path "$MEMORY_GRAPH" \
    --retrieval-results-csv "$RETRIEVAL_CSV" \
    --forget-path "$FORGET_PATH" \
    --retain-path "$RETAIN_PATH" \
    --forget-limit 0 \
    --retain-limit 0 \
    --output-csv "$out_dir/results.csv" \
    --output-json "$out_dir/results.json" \
    --backend vllm_api \
    --max-new-tokens 128 \
    --temperature 0.0 \
    --num-workers 8 \
    --rag-fallback-top-k 3 \
    --memory-max-hits 3 \
    2>&1 | tee "$log_dir/full_pipeline_generation.log"
  stop_port "$ANSWER_PORT"
  semantic_check "$out_dir/results.csv" "$out_dir/semantic_sanity.json" 2>&1 | tee "$log_dir/semantic_sanity.log"
  log "DONE generation key=$key"
}

run_judge_all() {
  local judge_out="$ROOT/full_pipeline_eval/judge_three_methods_20260703"
  local judge_log="$LOG_ROOT/judge_three_methods"
  mkdir -p "$judge_out" "$judge_log"
  local ga_csv="$ROOT/full_pipeline_eval/ga_ce60_full4000_graphcache_20260703/results.csv"
  local rmu_csv="$ROOT/full_pipeline_eval/rmu_ce109_full4000_graphcache_20260703/results.csv"
  local dpo_csv="$ROOT/full_pipeline_eval/dpo_step8200_full4000_graphcache_20260703/results.csv"
  log "START judge all three methods"
  stop_port "$JUDGE_PORT"
  nohup bash -lc "exec env CUDA_VISIBLE_DEVICES=0 '$VLLM_BIN' serve '$JUDGE_MODEL' --host 0.0.0.0 --port $JUDGE_PORT --served-model-name judge --max-model-len 2048 --max-num-seqs 2 --gpu-memory-utilization 0.90 --enforce-eager --trust-remote-code --generation-config vllm" > "$judge_log/judge_vllm.log" 2>&1 &
  echo $! > "$judge_log/judge_vllm.pid"
  wait_server "$JUDGE_PORT" "$judge_log/judge_vllm.pid" "$judge_log/judge_vllm.log"

  "$ENV52_PY" evaluation/llm_judge.py \
    --files "$ga_csv" "$rmu_csv" "$dpo_csv" \
    --labels "FlowMU_GA_for10_full" "FlowMU_RMU_for10_full" "FlowMU_DPO_for10_full" \
    --api-url "http://127.0.0.1:$JUDGE_PORT/v1" \
    --model judge \
    --output-dir "$judge_out/judge_qwen25" \
    --num-workers 1 \
    2>&1 | tee "$judge_log/judge.log"

  mapfile -t judge_csvs < <(find "$judge_out/judge_qwen25" -name 'judge_*.csv' | sort)
  "$ENV52_PY" evaluation/final_metrics.py \
    --files "${judge_csvs[@]}" \
    --labels "FlowMU_DPO_for10_full" "FlowMU_GA_for10_full" "FlowMU_RMU_for10_full" \
    --output-dir "$judge_out/final_metrics" \
    2>&1 | tee "$judge_log/final_metrics.log"
  log "DONE judge all three methods. Judge vLLM kept alive on port $JUDGE_PORT pid=$(cat "$judge_log/judge_vllm.pid")"
}

main() {
  log "RUNNER START"
  for p in "$RAG_GRAPH" "$MEMORY_GRAPH" "$RETRIEVAL_CSV" "$FORGET_PATH" "$RETAIN_PATH"; do
    [[ -e "$p" ]] || { log "ERROR missing required file: $p"; exit 2; }
  done
  run_generation "ga" "ga_ce60" "$ROOT/merged_model/ga/tofu_for10_re90_ga_native_top20rr09_ce60_20260702" "ga_ce60_full4000_graphcache_20260703"
  run_generation "rmu" "rmu_ce109" "$ROOT/merged_model/rmu/tofu_for10_re90_rmu_native_top20rr09_ce109_20260702" "rmu_ce109_full4000_graphcache_20260703"
  run_generation "dpo" "dpo_step8200" "$ROOT/merged_model/dpo/tofu_for10_re90_dpo_native_top20rr09_step8200_float16cpu_20260703" "dpo_step8200_full4000_graphcache_20260703"
  run_judge_all
  log "RUNNER DONE"
}

main "$@"
