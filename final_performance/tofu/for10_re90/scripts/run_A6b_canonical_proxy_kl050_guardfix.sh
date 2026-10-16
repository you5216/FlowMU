#!/usr/bin/env bash
set -euo pipefail

cd /disk8T/lkh/python_projects/agent_unlearning_zhx/agent

RUN=flowmu_ga_projected_canonproxy_A6b_guardfix_lr8e5_kl002_gn015_proxykl050_top50keepdownb_20260621
ROOT=final_performance/tofu/for10_re90
OUT_DIR="$ROOT/phase2_lora_trials/$RUN/lora"
LOG_DIR="$ROOT/phase2_lora_trials/$RUN/logs"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=0
export PHASE2_PROBE_PREFER_GPU=1

/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/env52/bin/python -m phase2.pipeline.main \
  --stage unlearning \
  --base-model-path /disk8T/lkh/python_projects/agent_unlearning_zhx/agent/merge_model/native_llm \
  --original-lora-path /disk8T/lkh/python_projects/agent_unlearning_zhx/agent/merge_model/native_llm \
  --forget-target-path /disk8T/lkh/python_projects/agent_unlearning_zhx/agent/final_performance/tofu/for10_re90/phase2_qa_trial16/graph_forget_qa.jsonl \
  --retain-path /disk8T/lkh/python_projects/agent_unlearning_zhx/agent/final_performance/tofu/for10_re90/phase2_qa_trial16/graph_retain_qa.jsonl \
  --unlearning-output-dir "$OUT_DIR" \
  --init-new-lora \
  --new-lora-r 16 \
  --new-lora-alpha 32 \
  --new-lora-dropout 0.05 \
  --new-lora-target-modules gate_proj,up_proj,down_proj \
  --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 1 \
  --seed 42 \
  --learning-rate 8e-5 \
  --num-train-epochs 2.0 \
  --logging-steps 10 \
  --max-length 512 \
  --warmup-steps 10 \
  --max-grad-norm 0.15 \
  --reuse-localization-pkl /disk8T/lkh/python_projects/agent_unlearning_zhx/agent/final_performance/tofu/for10_re90/phase2_localization_dualview_trial16_top50_qaw2_keepdownb_20260621/localized_neurons.pkl \
  --localization-key-annotation-path "" \
  --forget-loss-strategy gradient_ascent_projected \
  --forget-objective-weight 20.0 \
  --forget-target-mode random_vector \
  --forget-ce-weight 0.0 \
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
  --graph-vector-suppress-weight 0.0 \
  --retain-ce-weight 0.0 \
  --retain-kl-weight 0.02 \
  --disable-dual-kl-objective \
  --retain-kl-target 0.02 \
  --retain-topk 32 \
  --forget-nll-calibration-weight 0.0 \
  --early-stop-min-steps 200 \
  --early-stop-patience 200 \
  --best-score-retain-alpha 2.0 \
  --best-score-mia-alpha 0.0 \
  --best-retain-kl-max 0.5 \
  --best-retain-kl-fallback-max 0.6 \
  --enable-canonical-proxy \
  --canonical-proxy-forget-path /disk8T/lkh/python_projects/datasets/TOFU/forget10.json \
  --canonical-proxy-retain-path /disk8T/lkh/python_projects/datasets/TOFU/retain90.json \
  --canonical-proxy-forget-samples 400 \
  --canonical-proxy-retain-samples 400 \
  --canonical-proxy-interval 20 \
  --canonical-proxy-batch-size 2 \
  --canonical-proxy-retain-nll-delta-max 2.0 \
  --canonical-proxy-retain-kl-max 0.5 \
  --canonical-proxy-forget-nll-weight 1.0 \
  --canonical-proxy-forget-logprob-drop-weight 0.5 \
  --canonical-proxy-retain-nll-penalty 1.0 \
  --canonical-proxy-retain-kl-penalty 4.0 \
  > "$LOG_DIR/train.log" 2>&1
