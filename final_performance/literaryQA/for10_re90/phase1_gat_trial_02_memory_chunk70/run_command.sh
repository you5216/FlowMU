#!/usr/bin/env bash
set -euo pipefail

cd /disk8T/lkh/python_projects/agent_unlearning_zhx/agent

/disk8T/lkh/python_projects/agent_unlearning_zhx/environment/env52/bin/python -m phase1.dual_graph_unlearning_gat.run_dual_graph_unlearning \
  --rag-graph final_performance/literaryQA/base/rag/rag_graph.pkl \
  --memory-graph final_performance/literaryQA/base/memory/generic_memory_graph.pkl \
  --forget-queries-path final_performance/literaryQA/base/author/forget10_author.json \
  --forget-qa-query-path /disk8T/lkh/python_projects/datasets/LiteraryQA/literaryqa_annotations/rag/forget10.json \
  --encoder-model-path /disk8T/lkh/python_projects/llms/Qwen3.5-9B \
  --output-dir final_performance/literaryQA/for10_re90/phase1_gat_trial_02_memory_chunk70 \
  --device cuda \
  --embedding-device cuda \
  --embedding-batch-size 1 \
  --embedding-max-length 128 \
  --embedding-pooling mean \
  --query-embedding-cache final_performance/literaryQA/base/query_embedding_cache/for10_re90_strongqa_memoryfix.npz \
  --skip-pretrain \
  --rag-pretrain-model-path final_performance/literaryQA/base/phase1_gat_pretrain/rag_pre_gat.pt \
  --memory-pretrain-model-path final_performance/literaryQA/base/phase1_gat_pretrain/memory_pre_gat.pt \
  --rag-retrain-epochs 18 \
  --memory-retrain-epochs 30 \
  --rag-sim-percentile 99 \
  --rag-qa-topk 3 \
  --rag-community-top-m 3 \
  --rag-drop-class-weight-scale 1.2 \
  --rag-lexical-loss-weight 1.0 \
  --rag-safe-loss-weight 0.05 \
  --memory-path-cutoff 5 \
  --memory-qa-topk 10 \
  --memory-hard-threshold 0.2 \
  --memory-relation-prior 0.85 \
  --memory-mentioned-prior 1.0 \
  --memory-next-turn-prior 0.7 \
  --memory-background-relation-scale 0.08 \
  --memory-community-floor 0.25 \
  --memory-path-decay 1.5 \
  --memory-path-loss-weight 0.65 \
  --memory-safe-loss-weight 0.18 \
  --memory-toxic-loss-weight 0.65 \
  --memory-sparse-loss-weight 0.015 \
  --memory-consistency-loss-weight 0.05 \
  --memory-next-turn-loss-weight 0.15 \
  --memory-relation-loss-weight 0.08 \
  --memory-ranking-loss-weight 0.25
