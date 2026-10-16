#!/usr/bin/env bash
set -euo pipefail

AGENT_DIR="/disk8T/lkh/python_projects/agent_unlearning_zhx/agent"
ENV52_PY="/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/env52/bin/python"
VLLM_BIN="/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/52vllm/bin/vllm"

ROOT="$AGENT_DIR/final_performance/tofu/for10_re90"
BASE_MODEL="$AGENT_DIR/merge_model/native_llm"
NATIVE_LORA="/disk8T/lkh/python_projects/tofu_native_lora/checkpoint-4500"
GA_MODEL="$ROOT/merged_model/ga/tofu_for10_re90_ga_native_top20rr09_ce60_20260702"

RAG_GRAPH="$ROOT/phase1_gat_retrain_gatselect_rag0925_memory0925_20260702/rag_unlearning_graph.pkl"
MEMORY_GRAPH="$ROOT/phase1_gat_retrain_gatselect_rag0925_memory0925_20260702/memory_unlearning_graph.pkl"
RETRIEVAL_CSV="$ROOT/retrieval_cache/final_gatselect_rag0925_memory0925_forget10_retain90_top50_20260702/graph_path_eval_results.csv"

FORGET_PATH="/disk8T/lkh/python_projects/datasets/TOFU/re_split/forget10.json"
RETAIN_PATH="/disk8T/lkh/python_projects/datasets/TOFU/re_split/retain90.json"
PHASE2_QA_DIR="$ROOT/phase2_qa_gatselect_rag0925_memory0925_20260702"
EMPTY_RETAIN="$ROOT/empty_retain_for_forget_only.json"

SOURCE_MASK="$ROOT/localization/tofu_native_gatselect_rag0925_memory0925_cached_combined_top20_rr09_20260702/localized_neurons.pkl"
RANDOM_SENSOR_DIR="$ROOT/localization/tofu_native_random_same_density_top20_rr09_20260703"
RANDOM_SENSOR_MASK="$RANDOM_SENSOR_DIR/localized_neurons.pkl"
RANDOM_SENSOR_LORA="$ROOT/lora/ga_random_sensor_same_density_native_top20rr09_lr3e4_gn1_ce5_120_20260703"

LOG_ROOT="$ROOT/logs/ga_ablation_for10_20260703"
ANSWER_PORT="${ANSWER_PORT:-8002}"
GPU_ID="${GPU_ID:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"

mkdir -p "$LOG_ROOT"
cd "$AGENT_DIR"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG_ROOT/runner.log"
}

require_file() {
  local path="$1"
  [[ -e "$path" ]] || { log "ERROR missing required path: $path"; exit 2; }
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
rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
words = []
empty = 0
err = 0
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
        err += 1
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
    "eval_error": err,
    "short_lt3_words": short,
    "repeat_heavy_gt20": repeat_heavy,
    "avg_words": round(statistics.mean(words), 4) if words else 0.0,
    "semantic_ok": (
        len(rows) == 4000
        and split_counts.get("forget", 0) == 400
        and split_counts.get("retain", 0) == 3600
        and empty == 0
        and err == 0
        and repeat_heavy <= 100
    ),
}
out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
}

start_answer_vllm() {
  local log_dir="$LOG_ROOT/answer_ga_ce60"
  mkdir -p "$log_dir"
  stop_port "$ANSWER_PORT"
  log "starting GA CE60 answer vLLM on gpu=$GPU_ID port=$ANSWER_PORT"
  nohup bash -lc "exec env CUDA_VISIBLE_DEVICES='$GPU_ID' VLLM_USE_V1=0 '$VLLM_BIN' serve '$GA_MODEL' --host 0.0.0.0 --port '$ANSWER_PORT' --served-model-name ga_ce60 --max-model-len 2048 --max-num-seqs 8 --gpu-memory-utilization 0.86 --enforce-eager --trust-remote-code" > "$log_dir/answer_vllm.log" 2>&1 &
  echo $! > "$log_dir/answer_vllm.pid"
  wait_server "$ANSWER_PORT" "$log_dir/answer_vllm.pid" "$log_dir/answer_vllm.log"
}

run_one_context_ablation() {
  local label="$1"
  local run_name="$2"
  shift 2
  local out_dir="$ROOT/full_pipeline_eval/$run_name"
  local log_dir="$LOG_ROOT/$run_name"
  mkdir -p "$out_dir" "$log_dir"

  log "START context ablation label=$label output=$out_dir"
  VLLM_API_BASE="http://127.0.0.1:$ANSWER_PORT/v1" VLLM_MODEL_NAME="ga_ce60" QWEN_INFER_BACKEND="vllm_api" \
  "$ENV52_PY" -m phase2.pipeline.step6_full_pipeline_eval \
    --base-model-path "$BASE_MODEL" \
    --lora-path "$GA_MODEL" \
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
    --num-workers "$NUM_WORKERS" \
    --rag-fallback-top-k 3 \
    --memory-max-hits 3 \
    "$@" \
    2>&1 | tee "$log_dir/full_pipeline_generation.log"

  semantic_check "$out_dir/results.csv" "$out_dir/semantic_sanity.json" 2>&1 | tee "$log_dir/semantic_sanity.log"
  log "DONE context ablation label=$label"
}

run_context_ablations() {
  require_file "$GA_MODEL"
  require_file "$RAG_GRAPH"
  require_file "$MEMORY_GRAPH"
  require_file "$RETRIEVAL_CSV"
  require_file "$FORGET_PATH"
  require_file "$RETAIN_PATH"

  start_answer_vllm
  run_one_context_ablation "FlowMU-R_no_rag" "ablation_ga_ce60_no_rag_graphcache_20260703" --no-rag
  run_one_context_ablation "FlowMU-M_no_memory" "ablation_ga_ce60_no_memory_graphcache_20260703" --no-memory
  run_one_context_ablation "FlowMU-Inject_unfiltered" "ablation_ga_ce60_unfiltered_inject_graphcache_20260703" --disable-unlearning-filter

  if [[ "${KEEP_ANSWER_VLLM:-0}" != "1" ]]; then
    stop_port "$ANSWER_PORT"
  else
    log "KEEP_ANSWER_VLLM=1, answer vLLM remains on port $ANSWER_PORT"
  fi
}

create_random_sensor_mask() {
  require_file "$SOURCE_MASK"
  mkdir -p "$RANDOM_SENSOR_DIR"
  "$ENV52_PY" - "$SOURCE_MASK" "$RANDOM_SENSOR_DIR" <<'PY'
import json
import pickle
import sys
from pathlib import Path

import numpy as np

from phase2.diagnostics.derive_native_llm_localization_from_candidate_stats import build_lora_masks_from_shapes
from phase2.pipeline.stage2_unlearning import LocalizationArtifacts, save_json, save_pickle

source_pkl = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
seed = 20260703
rng = np.random.default_rng(seed)

with source_pkl.open("rb") as handle:
    source = pickle.load(handle)

shapes = {name: [int(x) for x in mask.shape] for name, mask in source.parameter_masks.items()}

def layer_dim(layer_idx: int) -> int:
    marker = f".layers.{layer_idx}.mlp.gate_proj."
    for name, shape in shapes.items():
        if marker in name and "lora_B" in name:
            return int(shape[0])
    raise ValueError(f"Cannot infer FFN dim for layer {layer_idx}")

selected_by_layer = {}
for layer_key, neurons in sorted(source.target_neurons.items(), key=lambda item: int(item[0].split("_")[1])):
    layer_idx = int(layer_key.split("_")[1])
    count = len(neurons)
    dim = layer_dim(layer_idx)
    selected_by_layer[layer_idx] = sorted(int(x) for x in rng.choice(dim, size=count, replace=False).tolist())

metadata = dict(source.metadata or {})
include_down_proj = bool(metadata.get("include_down_proj_mask", True))
include_down_proj_lora_b = bool(metadata.get("include_down_proj_lora_b", True))
parameter_masks = build_lora_masks_from_shapes(
    shapes,
    selected_by_layer,
    include_down_proj=include_down_proj,
    include_down_proj_lora_b=include_down_proj_lora_b,
)
total_selected_params = int(sum(np.asarray(mask).sum() for mask in parameter_masks.values()))
total_candidate_params = int(sum(np.asarray(mask).size for mask in parameter_masks.values()))

metadata.update(
    {
        "method_name": "random_same_density_sensor_ablation",
        "source_localization_pkl": str(source_pkl),
        "random_seed": seed,
        "same_layer_neuron_counts": True,
        "source_total_target_neurons": int(source.total_target_neurons),
        "total_selected_params": total_selected_params,
        "total_candidate_params": total_candidate_params,
        "parameter_mask_density": float(total_selected_params / max(1, total_candidate_params)),
        "ablation_role": "FlowMU-Sensor random same-density mask",
    }
)

target_neurons = {f"layer_{layer}": rows for layer, rows in selected_by_layer.items()}
layer_scores = {f"layer_{layer}": [1.0 for _ in rows] for layer, rows in selected_by_layer.items()}
localization = LocalizationArtifacts(
    strategy="random_same_density_sensor_ablation",
    top_fraction=float(getattr(source, "top_fraction", 1.0)),
    target_neurons=target_neurons,
    layer_scores=layer_scores,
    total_target_neurons=int(sum(len(rows) for rows in target_neurons.values())),
    parameter_masks=parameter_masks,
    metadata=metadata,
)

save_pickle(output_dir / "localized_neurons.pkl", localization)
save_json(output_dir / "localized_neurons.json", localization.to_serializable())
save_json(output_dir / "random_same_density_sensor_report.json", metadata)
print(json.dumps({"output_dir": str(output_dir), "total_target_neurons": localization.total_target_neurons, **metadata}, ensure_ascii=False, indent=2))
PY
}

train_random_sensor_ga() {
  require_file "$PHASE2_QA_DIR/graph_forget_qa.jsonl"
  require_file "$NATIVE_LORA"
  if [[ ! -e "$EMPTY_RETAIN" ]]; then
    printf '[]\n' > "$EMPTY_RETAIN"
  fi
  create_random_sensor_mask
  mkdir -p "$RANDOM_SENSOR_LORA/logs"
  log "START FlowMU-Sensor random-mask GA training output=$RANDOM_SENSOR_LORA"

  CUDA_VISIBLE_DEVICES="$GPU_ID" "$ENV52_PY" -m phase2.pipeline.main \
    --stage unlearning \
    --base-model-path "$BASE_MODEL" \
    --original-lora-path "$NATIVE_LORA" \
    --forget-target-path "$PHASE2_QA_DIR/graph_forget_qa.jsonl" \
    --retain-path "$EMPTY_RETAIN" \
    --unlearning-output-dir "$RANDOM_SENSOR_LORA" \
    --init-new-lora \
    --new-lora-r 16 \
    --new-lora-alpha 32 \
    --new-lora-dropout 0.05 \
    --new-lora-target-modules gate_proj,up_proj,down_proj \
    --per-device-train-batch-size 2 \
    --gradient-accumulation-steps 4 \
    --seed 42 \
    --learning-rate 3e-4 \
    --num-train-epochs 20 \
    --max-train-steps 1500 \
    --logging-steps 5 \
    --max-length 512 \
    --warmup-steps 5 \
    --max-grad-norm 1.0 \
    --reuse-localization-pkl "$RANDOM_SENSOR_MASK" \
    --forget-loss-strategy gradient_ascent \
    --forget-target-mode random_vector \
    --forget-objective-weight 1.0 \
    --forget-ce-weight 0.0 \
    --answer-suppression-weight 0.0 \
    --random-vector-forget-weight 1.0 \
    --forget-anchor-weight 1.0 \
    --forget-anchor-max-similarity 0.0 \
    --random-vector-target-similarity 0.25 \
    --forget-anchor-target-similarity -0.1 \
    --forget-weight-lr 0.05 \
    --forget-weight-min 0.0 \
    --forget-weight-max 12.0 \
    --random-vector-seed 42 \
    --parameter-random-vector-weight 1.0 \
    --parameter-random-vector-scale 0.01 \
    --forget-loss-top-layers 8 \
    --retain-ce-weight 0.0 \
    --retain-kl-weight 0.0 \
    --disable-dual-kl-objective \
    --disable-early-stop \
    --early-stop-min-steps 200 \
    --early-stop-patience 200 \
    --best-score-retain-alpha 2.0 \
    --best-score-mia-alpha 0.0 \
    --best-retain-kl-max -1 \
    --best-retain-kl-fallback-max -1 \
    --forget-ce-checkpoint-thresholds 5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120 \
    2>&1 | tee "$RANDOM_SENSOR_LORA/logs/train.log"

  log "DONE FlowMU-Sensor random-mask GA training"
}

usage() {
  cat <<EOF
Usage: $0 <command>

Commands:
  run-context          Generate FlowMU-R, FlowMU-M, and FlowMU-Inject with GA CE60 using the existing retrieval cache.
  make-random-mask     Create the same-density random localization mask for FlowMU-Sensor only.
  train-random-sensor  Create same-density random mask and train GA only for FlowMU-Sensor.
  all                  Run run-context, then train-random-sensor.

Environment:
  GPU_ID=0             GPU used by vLLM/training.
  ANSWER_PORT=8002     vLLM answer server port.
  NUM_WORKERS=8        Full-pipeline generation workers.
  KEEP_ANSWER_VLLM=1   Keep answer vLLM alive after run-context.
EOF
}

main() {
  local command="${1:-}"
  case "$command" in
    run-context)
      run_context_ablations
      ;;
    make-random-mask)
      create_random_sensor_mask
      ;;
    train-random-sensor)
      train_random_sensor_ga
      ;;
    all)
      run_context_ablations
      train_random_sensor_ga
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
