#!/usr/bin/env bash
set -euo pipefail

AGENT_DIR="/disk8T/lkh/python_projects/agent_unlearning_zhx/agent"
ENV52_PY="/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/env52/bin/python"

ROOT="$AGENT_DIR/final_performance/tofu/for10_re90"
BASE_MODEL="$AGENT_DIR/merge_model/native_llm"
NATIVE_LORA="/disk8T/lkh/python_projects/tofu_native_lora/checkpoint-4500"

MAIN_GRAPH_DIR="$ROOT/phase1_gat_retrain_gatselect_rag0925_memory0925_20260702"
MAIN_QA_DIR="$ROOT/phase2_qa_gatselect_rag0925_memory0925_20260702"
EMPTY_RETAIN="$ROOT/empty_retain_for_forget_only.json"
OFFICIAL_FORGET="/disk8T/lkh/python_projects/datasets/TOFU/re_split/forget10.json"

INJECT_QA_DIR="$ROOT/phase2_qa_inject_rawforget_20260704"
INJECT_RAW_LOC="$ROOT/localization/tofu_native_inject_rawforget_raw_top300_rr08_20260704"
INJECT_MASK_DIR="$ROOT/localization/tofu_native_inject_rawforget_cached_combined_top20_rr09_20260704"
INJECT_LORA="$ROOT/lora/ga_inject_rawforget_native_top20rr09_lr3e4_gn1_ce5_120_20260704"
INJECT_MERGED="$ROOT/merged_model/ablation/tofu_for10_re90_ga_inject_rawforget_top20rr09_ce60_20260704"

SENSOR_GRAPH_DIR="$ROOT/phase1_similarity_sensor_rag560_memory1255_20260704"
SENSOR_QA_DIR="$ROOT/phase2_qa_similarity_sensor_20260704"
SENSOR_RAW_LOC="$ROOT/localization/tofu_native_similarity_sensor_raw_top300_rr08_20260704"
SENSOR_MASK_DIR="$ROOT/localization/tofu_native_similarity_sensor_cached_combined_top20_rr09_20260704"
SENSOR_LORA="$ROOT/lora/ga_similarity_sensor_native_top20rr09_lr3e4_gn1_ce5_120_20260704"
SENSOR_MERGED="$ROOT/merged_model/ablation/tofu_for10_re90_ga_similarity_sensor_top20rr09_ce60_20260704"

LOG_ROOT="$ROOT/logs/flowmu_inject_sensor_pregen_20260704"
GPU_ID="${GPU_ID:-0}"

mkdir -p "$LOG_ROOT"
cd "$AGENT_DIR"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG_ROOT/runner.log"
}

require_file() {
  local path="$1"
  [[ -e "$path" ]] || { log "ERROR missing required path: $path"; exit 2; }
}

ensure_empty_retain() {
  if [[ ! -e "$EMPTY_RETAIN" ]]; then
    printf '[]\n' > "$EMPTY_RETAIN"
  fi
}

prepare_assets() {
  require_file "$OFFICIAL_FORGET"
  require_file "$MAIN_GRAPH_DIR/rag_unlearning_graph.pkl"
  require_file "$MAIN_GRAPH_DIR/memory_unlearning_graph.pkl"
  log "prepare raw official QA and similarity-only Sensor graphs"
  "$ENV52_PY" "$ROOT/scripts/prepare_inject_sensor_ablation_assets.py" \
    --root "$ROOT" \
    --official-forget "$OFFICIAL_FORGET" \
    --source-rag-graph "$MAIN_GRAPH_DIR/rag_unlearning_graph.pkl" \
    --source-memory-graph "$MAIN_GRAPH_DIR/memory_unlearning_graph.pkl" \
    --inject-qa-dir "$INJECT_QA_DIR" \
    --sensor-graph-dir "$SENSOR_GRAPH_DIR" \
    --rag-target-nodes 560 \
    --memory-target-chunks 1255 \
    --memory-target-edges 1135 \
    2>&1 | tee "$LOG_ROOT/prepare_assets.log"
}

export_sensor_qa() {
  require_file "$SENSOR_GRAPH_DIR/rag_unlearning_graph.pkl"
  require_file "$SENSOR_GRAPH_DIR/memory_unlearning_graph.pkl"
  log "export graph-derived QA for similarity-only Sensor"
  "$ENV52_PY" -m phase2.pipeline.export_graph_phase2_qa \
    --rag-graph "$SENSOR_GRAPH_DIR/rag_unlearning_graph.pkl" \
    --memory-graph "$SENSOR_GRAPH_DIR/memory_unlearning_graph.pkl" \
    --output-dir "$SENSOR_QA_DIR" \
    2>&1 | tee "$LOG_ROOT/export_sensor_qa.log"
}

derive_top20_mask() {
  local raw_dir="$1"
  local out_dir="$2"
  require_file "$raw_dir/candidate_neuron_stats.npz"
  require_file "$raw_dir/candidate_neuron_stats_manifest.json"
  log "derive top20 rr09 mask from $raw_dir"
  "$ENV52_PY" -m phase2.diagnostics.derive_native_llm_localization_from_candidate_stats \
    --candidate-stats-npz "$raw_dir/candidate_neuron_stats.npz" \
    --manifest-json "$raw_dir/candidate_neuron_stats_manifest.json" \
    --source-report-json "$raw_dir/native_llm_graph_neuron_localizer_report.json" \
    --output-dir "$out_dir" \
    --layer-range all \
    --neurons-per-layer 20 \
    --score-field combined_raw_diff \
    --retain-ratio-field combined_retain_to_forget_ratio \
    --forget-mean-field combined_forget_mean \
    --retain-ratio-max 0.9 \
    2>&1 | tee "$out_dir/derive_top20_rr09.log"
}

localize_inject() {
  require_file "$INJECT_QA_DIR/raw_official_forget_qa.jsonl"
  require_file "$MAIN_QA_DIR/graph_retain_qa.jsonl"
  require_file "$MAIN_GRAPH_DIR/rag_unlearning_graph.pkl"
  require_file "$MAIN_GRAPH_DIR/memory_unlearning_graph.pkl"
  mkdir -p "$INJECT_RAW_LOC" "$INJECT_MASK_DIR"
  log "localize FlowMU-Inject with raw official forget QA text view"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$ENV52_PY" -m phase2.diagnostics.native_llm_graph_neuron_localizer \
    --base-model-path "$BASE_MODEL" \
    --rag-graph-path "$MAIN_GRAPH_DIR/rag_unlearning_graph.pkl" \
    --memory-graph-path "$MAIN_GRAPH_DIR/memory_unlearning_graph.pkl" \
    --memory-report-path "$MAIN_GRAPH_DIR/memory_unlearning_report.json" \
    --output-dir "$INJECT_RAW_LOC" \
    --device cuda \
    --seed 42 \
    --layer-range 0-31 \
    --neurons-per-layer 300 \
    --batch-size 8 \
    --activation-position prefix \
    --qa-forget-path "$INJECT_QA_DIR/raw_official_forget_qa.jsonl" \
    --qa-retain-path "$MAIN_QA_DIR/graph_retain_qa.jsonl" \
    --qa-max-forget 0 \
    --qa-max-retain 400 \
    --qa-batch-size 1 \
    --qa-max-length 512 \
    --graph-view-weight 1.0 \
    --qa-view-weight 2.0 \
    --ranking-mode positive_diff \
    --new-lora-r 16 \
    --new-lora-alpha 32 \
    --new-lora-dropout 0.05 \
    --new-lora-target-modules gate_proj,up_proj,down_proj \
    2>&1 | tee "$LOG_ROOT/localize_inject_raw.log"
  derive_top20_mask "$INJECT_RAW_LOC" "$INJECT_MASK_DIR"
}

localize_sensor() {
  require_file "$SENSOR_QA_DIR/graph_forget_qa.jsonl"
  require_file "$SENSOR_QA_DIR/graph_retain_qa.jsonl"
  mkdir -p "$SENSOR_RAW_LOC" "$SENSOR_MASK_DIR"
  log "localize FlowMU-Sensor with similarity-only graph-derived QA"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$ENV52_PY" -m phase2.diagnostics.native_llm_graph_neuron_localizer \
    --base-model-path "$BASE_MODEL" \
    --rag-graph-path "$SENSOR_GRAPH_DIR/rag_unlearning_graph.pkl" \
    --memory-graph-path "$SENSOR_GRAPH_DIR/memory_unlearning_graph.pkl" \
    --memory-report-path "$SENSOR_GRAPH_DIR/memory_unlearning_report.json" \
    --output-dir "$SENSOR_RAW_LOC" \
    --device cuda \
    --seed 42 \
    --layer-range 0-31 \
    --neurons-per-layer 300 \
    --batch-size 8 \
    --activation-position prefix \
    --qa-forget-path "$SENSOR_QA_DIR/graph_forget_qa.jsonl" \
    --qa-retain-path "$SENSOR_QA_DIR/graph_retain_qa.jsonl" \
    --qa-max-forget 0 \
    --qa-max-retain 1815 \
    --qa-batch-size 1 \
    --qa-max-length 512 \
    --graph-view-weight 1.0 \
    --qa-view-weight 2.0 \
    --ranking-mode positive_diff \
    --new-lora-r 16 \
    --new-lora-alpha 32 \
    --new-lora-dropout 0.05 \
    --new-lora-target-modules gate_proj,up_proj,down_proj \
    2>&1 | tee "$LOG_ROOT/localize_sensor_raw.log"
  derive_top20_mask "$SENSOR_RAW_LOC" "$SENSOR_MASK_DIR"
}

train_ga() {
  local label="$1"
  local mask_pkl="$2"
  local forget_qa="$3"
  local out_dir="$4"
  require_file "$mask_pkl"
  require_file "$forget_qa"
  require_file "$NATIVE_LORA"
  ensure_empty_retain
  mkdir -p "$out_dir/logs"
  log "train GA for $label output=$out_dir"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$ENV52_PY" -m phase2.pipeline.main \
    --stage unlearning \
    --base-model-path "$BASE_MODEL" \
    --original-lora-path "$NATIVE_LORA" \
    --forget-target-path "$forget_qa" \
    --retain-path "$EMPTY_RETAIN" \
    --unlearning-output-dir "$out_dir" \
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
    --max-train-steps 700 \
    --logging-steps 5 \
    --max-length 512 \
    --warmup-steps 5 \
    --max-grad-norm 1.0 \
    --reuse-localization-pkl "$mask_pkl" \
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
    2>&1 | tee "$out_dir/logs/train.log"
}

merge_ce60() {
  local label="$1"
  local lora_dir="$2"
  local merged_dir="$3"
  local checkpoint
  checkpoint=$(find "$lora_dir/ce_threshold_checkpoints" -maxdepth 1 -type d -name 'ce60_step*' | sort -V | tail -n 1 || true)
  if [[ -z "$checkpoint" ]]; then
    log "ERROR no ce60 checkpoint for $label under $lora_dir/ce_threshold_checkpoints"
    exit 3
  fi
  log "merge $label ce60 checkpoint=$checkpoint output=$merged_dir"
  "$ENV52_PY" -m phase2.pipeline.merge_lora_for_vllm \
    --base-model-path "$BASE_MODEL" \
    --lora-path "$checkpoint" \
    --output-dir "$merged_dir" \
    --torch-dtype float16 \
    --device-map cpu \
    --label "$label" \
    --note "TOFU for10 FlowMU ablation pre-generation merge: $label" \
    --force \
    2>&1 | tee "$LOG_ROOT/merge_${label}.log"
}

run_inject() {
  localize_inject
  train_ga "inject_rawforget" "$INJECT_MASK_DIR/localized_neurons.pkl" "$MAIN_QA_DIR/graph_forget_qa.jsonl" "$INJECT_LORA"
  merge_ce60 "inject_rawforget" "$INJECT_LORA" "$INJECT_MERGED"
}

run_sensor() {
  export_sensor_qa
  localize_sensor
  train_ga "similarity_sensor" "$SENSOR_MASK_DIR/localized_neurons.pkl" "$SENSOR_QA_DIR/graph_forget_qa.jsonl" "$SENSOR_LORA"
  merge_ce60 "similarity_sensor" "$SENSOR_LORA" "$SENSOR_MERGED"
}

usage() {
  cat <<EOF
Usage: $0 <command>

Commands:
  prepare-assets    CPU only: raw forget QA + similarity-only graphs.
  export-sensor-qa  CPU only: export QA from similarity-only graphs.
  localize-inject   GPU: raw official forget QA localization + top20 rr09 mask.
  localize-sensor   GPU: similarity-only graph localization + top20 rr09 mask.
  train-inject      GPU: GA train Inject mask and merge CE60.
  train-sensor      GPU: GA train Sensor mask and merge CE60.
  run-inject        GPU: localize + train + merge Inject.
  run-sensor        GPU: export QA + localize + train + merge Sensor.
  all               prepare-assets + run-inject + run-sensor.
EOF
}

main() {
  local command="${1:-}"
  case "$command" in
    prepare-assets) prepare_assets ;;
    export-sensor-qa) export_sensor_qa ;;
    localize-inject) localize_inject ;;
    localize-sensor) localize_sensor ;;
    train-inject)
      train_ga "inject_rawforget" "$INJECT_MASK_DIR/localized_neurons.pkl" "$MAIN_QA_DIR/graph_forget_qa.jsonl" "$INJECT_LORA"
      merge_ce60 "inject_rawforget" "$INJECT_LORA" "$INJECT_MERGED"
      ;;
    train-sensor)
      train_ga "similarity_sensor" "$SENSOR_MASK_DIR/localized_neurons.pkl" "$SENSOR_QA_DIR/graph_forget_qa.jsonl" "$SENSOR_LORA"
      merge_ce60 "similarity_sensor" "$SENSOR_LORA" "$SENSOR_MERGED"
      ;;
    run-inject) run_inject ;;
    run-sensor) run_sensor ;;
    all)
      prepare_assets
      run_inject
      run_sensor
      ;;
    *)
      usage
      exit 2
      ;;
  esac
}

main "$@"
