#!/usr/bin/env bash
set -euo pipefail

cd /home/likunhao/likunhao52shared/agent_unlearning_zhx/agent

: "${DEEPSEEK_API_KEY:?Set DEEPSEEK_API_KEY before running this script.}"

exec /home/likunhao/likunhao52shared/agent_unlearning_zhx/environment/env50/bin/python \
  tools/rewrite_literaryqa_deepseek.py
