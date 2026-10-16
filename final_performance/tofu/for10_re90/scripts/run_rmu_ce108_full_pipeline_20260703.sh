#!/usr/bin/env bash
set -euo pipefail

AGENT_DIR="/disk8T/lkh/python_projects/agent_unlearning_zhx/agent"
ENV52_PY="/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/env52/bin/python"
VLLM_BIN="/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/52vllm/bin/vllm"
BASE_MODEL="$AGENT_DIR/merge_model/native_llm"
JUDGE_MODEL="/disk8T/lkh/python_projects/llms/Qwen2.5-VL-7B-Instruct"
MERGED="$AGENT_DIR/final_performance/tofu/for10_re90/merged_model/rmu/tofu_for10_re90_rmu_native_top20rr09_ce108_20260702"
RAG_GRAPH="$AGENT_DIR/final_performance/tofu/for10_re90/phase1_gat_retrain_gatselect_rag0925_memory0925_20260702/rag_unlearning_graph.pkl"
MEM_GRAPH="$AGENT_DIR/final_performance/tofu/for10_re90/phase1_gat_retrain_gatselect_rag0925_memory0925_20260702/memory_unlearning_graph.pkl"
RETRIEVAL="$AGENT_DIR/final_performance/tofu/for10_re90/retrieval_cache/final_gatselect_rag0925_memory0925_forget10_retain90_top50_20260702/graph_path_eval_results.csv"
FORGET_PATH="/disk8T/lkh/python_projects/datasets/TOFU/re_split/forget10.json"
RETAIN_PATH="/disk8T/lkh/python_projects/datasets/TOFU/re_split/retain90.json"
RUN_NAME="rmu_ce108_full4000_graphcache_20260703"
OUT_DIR="$AGENT_DIR/final_performance/tofu/for10_re90/full_pipeline_eval/$RUN_NAME"
JUDGE_DIR="$AGENT_DIR/final_performance/tofu/for10_re90/full_pipeline_eval/judge_rmu_ce108_20260703"
LOG_DIR="$AGENT_DIR/final_performance/tofu/for10_re90/logs/$RUN_NAME"
mkdir -p "$OUT_DIR" "$JUDGE_DIR" "$LOG_DIR"
cd "$AGENT_DIR"

stop_port() {
  local port="$1"
  local pids
  pids=$(ss -ltnp "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | sort -u | tr '\n' ' ' || true)
  if [[ -n "${pids// }" ]]; then
    echo "stopping port $port pids: $pids"
    kill $pids 2>/dev/null || true
    sleep 8
  fi
  return 0
}

wait_server() {
  local port="$1"
  local pid_file="$2"
  local log_file="$3"
  for i in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then
      echo "server_ready port=$port waited=$((i*5))s"
      return 0
    fi
    if [[ -f "$pid_file" ]] && ! ps -p "$(cat "$pid_file")" >/dev/null 2>&1; then
      echo "ERROR server on port $port exited"
      tail -160 "$log_file" || true
      return 1
    fi
    if (( i % 6 == 0 )); then
      echo "waiting server port=$port $((i*5))s"
      tail -8 "$log_file" || true
    fi
    sleep 5
  done
  echo "ERROR server on port $port not ready"
  tail -160 "$log_file" || true
  return 1
}

semantic_sanity() {
  "$ENV52_PY" - <<PY
import csv,json,re
from pathlib import Path
p=Path('$OUT_DIR/results.csv')
rows=list(csv.DictReader(p.open(newline='',encoding='utf-8',errors='ignore')))
summary={'rows':len(rows),'split_counts':{},'empty':0,'eval_error':0,'short_lt3_words':0,'repeat_heavy_gt20':0,'avg_words':0.0}
words=[]
for r in rows:
    sp=(r.get('split') or '').strip()
    summary['split_counts'][sp]=summary['split_counts'].get(sp,0)+1
    pred=(r.get('prediction') or '').strip()
    if not pred: summary['empty']+=1
    if 'EVAL_ERROR' in pred: summary['eval_error']+=1
    w=pred.split(); words.append(len(w))
    if len(w)<3: summary['short_lt3_words']+=1
    toks=[t.lower() for t in re.findall(r'\\w+', pred)]
    max_run=1; cur=1
    for a,b in zip(toks,toks[1:]):
        cur=cur+1 if a==b else 1
        max_run=max(max_run,cur)
    if max_run>=20: summary['repeat_heavy_gt20']+=1
summary['avg_words']=round(sum(words)/len(words),4) if words else 0.0
summary['semantic_ok']=summary['rows']==4000 and summary['eval_error']==0 and summary['repeat_heavy_gt20']==0
Path('$OUT_DIR/semantic_sanity.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
PY
}

echo "=== cleanup existing servers ==="
stop_port 8002
stop_port 8003
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true

echo "=== start answer vllm ce108 ==="
nohup bash -lc "exec env CUDA_VISIBLE_DEVICES=0 '$VLLM_BIN' serve '$MERGED' --host 0.0.0.0 --port 8002 --served-model-name rmu_ce108 --max-model-len 1024 --max-num-seqs 4 --gpu-memory-utilization 0.96 --enforce-eager --trust-remote-code" > "$LOG_DIR/answer_vllm.log" 2>&1 &
echo $! > "$LOG_DIR/answer_vllm.pid"
wait_server 8002 "$LOG_DIR/answer_vllm.pid" "$LOG_DIR/answer_vllm.log"

echo "=== generate full pipeline ce108 ==="
VLLM_API_BASE="http://127.0.0.1:8002/v1" VLLM_MODEL_NAME="rmu_ce108" QWEN_INFER_BACKEND="vllm_api" \
"$ENV52_PY" -m phase2.pipeline.step6_full_pipeline_eval \
  --base-model-path "$BASE_MODEL" \
  --lora-path "$MERGED" \
  --rag-graph-path "$RAG_GRAPH" \
  --memory-graph-path "$MEM_GRAPH" \
  --retrieval-results-csv "$RETRIEVAL" \
  --forget-path "$FORGET_PATH" \
  --retain-path "$RETAIN_PATH" \
  --forget-limit 0 \
  --retain-limit 0 \
  --output-csv "$OUT_DIR/results.csv" \
  --output-json "$OUT_DIR/results.json" \
  --backend vllm_api \
  --max-new-tokens 128 \
  --temperature 0.0 \
  --num-workers 4 \
  --rag-fallback-top-k 3 \
  --memory-max-hits 3 \
  2>&1 | tee "$LOG_DIR/full_pipeline_generation.log"
semantic_sanity 2>&1 | tee "$LOG_DIR/semantic_sanity.log"
stop_port 8002

echo "=== start judge vllm ==="
stop_port 8003
nohup bash -lc "exec env CUDA_VISIBLE_DEVICES=0 '$VLLM_BIN' serve '$JUDGE_MODEL' --host 0.0.0.0 --port 8003 --served-model-name judge --max-model-len 2048 --max-num-seqs 2 --gpu-memory-utilization 0.90 --enforce-eager --trust-remote-code --generation-config vllm" > "$LOG_DIR/judge_vllm.log" 2>&1 &
echo $! > "$LOG_DIR/judge_vllm.pid"
wait_server 8003 "$LOG_DIR/judge_vllm.pid" "$LOG_DIR/judge_vllm.log"

echo "=== judge ce108 ==="
rm -rf "$JUDGE_DIR/judge_qwen25" "$JUDGE_DIR/final_metrics"
"$ENV52_PY" evaluation/llm_judge.py \
  --files "$OUT_DIR/results.csv" \
  --labels FlowMU_RMU_ce108_for10_full \
  --api-url http://127.0.0.1:8003/v1 \
  --model judge \
  --output-dir "$JUDGE_DIR/judge_qwen25" \
  --num-workers 1 \
  2>&1 | tee "$LOG_DIR/judge.log"

JUDGE_CSV="$(find "$JUDGE_DIR/judge_qwen25" -name 'judge_*.csv' | head -1)"
"$ENV52_PY" evaluation/final_metrics.py \
  --files "$JUDGE_CSV" \
  --labels FlowMU_RMU_ce108_for10_full \
  --output-dir "$JUDGE_DIR/final_metrics" \
  2>&1 | tee "$LOG_DIR/final_metrics.log"
cat "$JUDGE_DIR/final_metrics/final_metrics_summary.json"
echo "=== DONE rmu ce108 full pipeline ==="
