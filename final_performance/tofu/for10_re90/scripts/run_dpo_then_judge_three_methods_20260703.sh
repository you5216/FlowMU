#!/usr/bin/env bash
set -euo pipefail
AGENT_DIR="/disk8T/lkh/python_projects/agent_unlearning_zhx/agent"
ENV52_PY="/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/env52/bin/python"
VLLM_BIN="/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/52vllm/bin/vllm"
BASE_MODEL="$AGENT_DIR/merge_model/native_llm"
JUDGE_MODEL="/disk8T/lkh/python_projects/llms/Qwen2.5-VL-7B-Instruct"
ROOT="$AGENT_DIR/final_performance/tofu/for10_re90"
LOG_ROOT="$ROOT/logs/full_pipeline_three_methods_20260703_dpo_only"
RAG_GRAPH="$ROOT/phase1_gat_retrain_gatselect_rag0925_memory0925_20260702/rag_unlearning_graph.pkl"
MEMORY_GRAPH="$ROOT/phase1_gat_retrain_gatselect_rag0925_memory0925_20260702/memory_unlearning_graph.pkl"
RETRIEVAL_CSV="$ROOT/retrieval_cache/final_gatselect_rag0925_memory0925_forget10_retain90_top50_20260702/graph_path_eval_results.csv"
FORGET_PATH="/disk8T/lkh/python_projects/datasets/TOFU/re_split/forget10.json"
RETAIN_PATH="/disk8T/lkh/python_projects/datasets/TOFU/re_split/retain90.json"
DPO_MODEL="$ROOT/merged_model/dpo/tofu_for10_re90_dpo_native_top20rr09_step8200_float16cpu_20260703"
DPO_OUT="$ROOT/full_pipeline_eval/dpo_step8200_full4000_graphcache_20260703"
ANSWER_PORT=8002
JUDGE_PORT=8003
mkdir -p "$LOG_ROOT" "$DPO_OUT"
cd "$AGENT_DIR"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG_ROOT/continue.log"; }
stop_port(){ local port="$1"; local pids; pids=$(ss -ltnp "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | sort -u | tr '\n' ' ' || true); if [[ -n "${pids// }" ]]; then log "stopping port $port pids: $pids"; kill $pids 2>/dev/null || true; sleep 8; fi; }
wait_server(){ local port="$1" pid_file="$2" log_file="$3"; for i in $(seq 1 180); do if curl -fsS "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then log "server_ready port=$port waited=$((i*5))s"; return 0; fi; if [[ -f "$pid_file" ]] && ! ps -p "$(cat "$pid_file")" >/dev/null 2>&1; then log "ERROR server on port $port exited"; tail -180 "$log_file" | tee -a "$LOG_ROOT/continue.log" || true; return 1; fi; if (( i % 12 == 0 )); then log "waiting server port=$port $((i*5))s"; tail -8 "$log_file" | tee -a "$LOG_ROOT/continue.log" || true; fi; sleep 5; done; log "ERROR server on port $port not ready"; tail -180 "$log_file" | tee -a "$LOG_ROOT/continue.log" || true; return 1; }
semantic_record(){ local csv_path="$1" out_json="$2"; "$ENV52_PY" - "$csv_path" "$out_json" <<'PY'
import csv,json,re,statistics,sys
from pathlib import Path
csv_path=Path(sys.argv[1]); out=Path(sys.argv[2])
rows=list(csv.DictReader(csv_path.open(encoding='utf-8')))
words=[]; empty=err=short=repeat=0; split={}
for r in rows:
    split[r.get('split','')]=split.get(r.get('split',''),0)+1
    pred=(r.get('prediction') or '').strip()
    empty += int(not pred)
    err += int('[EVAL_ERROR]' in pred)
    toks=re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", pred)
    words.append(len(toks)); short += int(len(toks)<3)
    repeat += int(bool(re.search(r"(.)\1{20,}", pred)))
summary={"rows":len(rows),"split_counts":split,"empty":empty,"eval_error":err,"short_lt3_words":short,"repeat_heavy_gt20":repeat,"avg_words":round(statistics.mean(words),4) if words else 0.0,"semantic_ok":len(rows)==4000 and split.get('forget',0)==400 and split.get('retain',0)==3600 and err==0}
out.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
PY
}
run_dpo(){ local log_dir="$LOG_ROOT/dpo_step8200_full4000_graphcache_20260703"; mkdir -p "$log_dir"; log "START DPO generation"; stop_port "$ANSWER_PORT"; nohup bash -lc "exec env CUDA_VISIBLE_DEVICES=0 VLLM_USE_V1=0 '$VLLM_BIN' serve '$DPO_MODEL' --host 0.0.0.0 --port $ANSWER_PORT --served-model-name dpo_step8200 --max-model-len 2048 --max-num-seqs 8 --gpu-memory-utilization 0.86 --enforce-eager --trust-remote-code" > "$log_dir/answer_vllm.log" 2>&1 & echo $! > "$log_dir/answer_vllm.pid"; wait_server "$ANSWER_PORT" "$log_dir/answer_vllm.pid" "$log_dir/answer_vllm.log"; VLLM_API_BASE="http://127.0.0.1:$ANSWER_PORT/v1" VLLM_MODEL_NAME="dpo_step8200" QWEN_INFER_BACKEND="vllm_api" "$ENV52_PY" -m phase2.pipeline.step6_full_pipeline_eval --base-model-path "$BASE_MODEL" --lora-path "$DPO_MODEL" --rag-graph-path "$RAG_GRAPH" --memory-graph-path "$MEMORY_GRAPH" --retrieval-results-csv "$RETRIEVAL_CSV" --forget-path "$FORGET_PATH" --retain-path "$RETAIN_PATH" --forget-limit 0 --retain-limit 0 --output-csv "$DPO_OUT/results.csv" --output-json "$DPO_OUT/results.json" --backend vllm_api --max-new-tokens 128 --temperature 0.0 --num-workers 8 --rag-fallback-top-k 3 --memory-max-hits 3 2>&1 | tee "$log_dir/full_pipeline_generation.log"; stop_port "$ANSWER_PORT"; semantic_record "$DPO_OUT/results.csv" "$DPO_OUT/semantic_sanity.json" 2>&1 | tee "$log_dir/semantic_sanity.log"; log "DONE DPO generation"; }
run_judge(){ local judge_out="$ROOT/full_pipeline_eval/judge_three_methods_20260703" judge_log="$LOG_ROOT/judge_three_methods"; mkdir -p "$judge_out" "$judge_log"; local ga_csv="$ROOT/full_pipeline_eval/ga_ce60_full4000_graphcache_20260703/results.csv" rmu_csv="$ROOT/full_pipeline_eval/rmu_ce109_full4000_graphcache_20260703/results.csv" dpo_csv="$DPO_OUT/results.csv"; log "START judge all three"; stop_port "$JUDGE_PORT"; nohup bash -lc "exec env CUDA_VISIBLE_DEVICES=0 '$VLLM_BIN' serve '$JUDGE_MODEL' --host 0.0.0.0 --port $JUDGE_PORT --served-model-name judge --max-model-len 2048 --max-num-seqs 2 --gpu-memory-utilization 0.90 --enforce-eager --trust-remote-code --generation-config vllm" > "$judge_log/judge_vllm.log" 2>&1 & echo $! > "$judge_log/judge_vllm.pid"; wait_server "$JUDGE_PORT" "$judge_log/judge_vllm.pid" "$judge_log/judge_vllm.log"; "$ENV52_PY" evaluation/llm_judge.py --files "$ga_csv" "$rmu_csv" "$dpo_csv" --labels FlowMU_GA_for10_full FlowMU_RMU_for10_full FlowMU_DPO_for10_full --api-url "http://127.0.0.1:$JUDGE_PORT/v1" --model judge --output-dir "$judge_out/judge_qwen25" --num-workers 1 2>&1 | tee "$judge_log/judge.log"; "$ENV52_PY" evaluation/final_metrics.py --files "$judge_out/judge_qwen25/judge_FlowMU_GA_for10_full.csv" "$judge_out/judge_qwen25/judge_FlowMU_RMU_for10_full.csv" "$judge_out/judge_qwen25/judge_FlowMU_DPO_for10_full.csv" --labels FlowMU_GA_for10_full FlowMU_RMU_for10_full FlowMU_DPO_for10_full --output-dir "$judge_out/final_metrics" 2>&1 | tee "$judge_log/final_metrics.log"; log "DONE judge all three; judge vLLM kept alive pid=$(cat "$judge_log/judge_vllm.pid")"; }
log "CONTINUE START"
run_dpo
run_judge
log "CONTINUE DONE"
