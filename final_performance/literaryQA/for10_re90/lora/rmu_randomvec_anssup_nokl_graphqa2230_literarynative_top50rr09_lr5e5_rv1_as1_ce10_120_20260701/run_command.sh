#!/usr/bin/env bash
set -euo pipefail

cd /disk8T/lkh/python_projects/agent_unlearning_zhx/agent

export HOME=/disk8T/lkh/runtime_home
export XDG_CACHE_HOME=/disk8T/lkh/runtime_cache
export HF_HOME=/disk8T/lkh/hf_cache
export TRANSFORMERS_CACHE=/disk8T/lkh/hf_cache
export TORCH_HOME=/disk8T/lkh/torch_cache
export TMPDIR=/disk8T/lkh/tmp
export CUDA_VISIBLE_DEVICES=0

/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/env52/bin/python phase2/pipeline/main.py \
  --stage unlearning \
  --base-model-path final_performance/literaryQA/base/lora/merged/literary_neative_model \
  --original-lora-path final_performance/literaryQA/base/lora/merged/literary_neative_model \
  --forget-target-path final_performance/literaryQA/for10_re90/phase2_graph_qa_trial_01_strongqa_memoryfix/graph_forget_qa.jsonl \
  --retain-path final_performance/literaryQA/for10_re90/phase2_graph_qa_trial_01_strongqa_memoryfix/graph_retain_qa.jsonl \
  --unlearning-output-dir final_performance/literaryQA/for10_re90/lora/rmu_randomvec_anssup_nokl_graphqa2230_literarynative_top50rr09_lr5e5_rv1_as1_ce10_120_20260701 \
  --init-new-lora \
  --new-lora-r 16 \
  --new-lora-alpha 32 \
  --new-lora-dropout 0.05 \
  --new-lora-target-modules gate_proj,up_proj,down_proj \
  --reuse-localization-pkl final_performance/literaryQA/for10_re90/localization/literary_neative_model_cached_combined_top50_rr09_20260701/localized_neurons.pkl \
  --forget-loss-strategy parameter_random_vector \
  --forget-objective-weight 1.0 \
  --parameter-random-vector-weight 1.0 \
  --parameter-random-vector-scale 0.01 \
  --answer-suppression-path final_performance/literaryQA/for10_re90/phase2_graph_qa_trial_01_strongqa_memoryfix/graph_forget_qa.jsonl \
  --answer-suppression-weight 1.0 \
  --retain-kl-weight 0.0 \
  --disable-dual-kl-objective \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 4 \
  --learning-rate 5e-5 \
  --num-train-epochs 8 \
  --max-length 512 \
  --warmup-steps 5 \
  --max-grad-norm 0.3 \
  --logging-steps 5 \
  --disable-early-stop \
  --forget-ce-checkpoint-thresholds 10,20,30,40,50,60,70,80,90,100,110,120
