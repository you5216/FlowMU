#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/likunhao/likunhao52shared/agent_unlearning_zhx/agent/sequential_incremental_unlearning_tofu_for10"
AGENT="/home/likunhao/likunhao52shared/agent_unlearning_zhx/agent"
PY="/home/likunhao/likunhao52shared/agent_unlearning_zhx/environment/env50/bin/python"

cd "${AGENT}"
mkdir -p "${ROOT}/logs" "${ROOT}/results"

echo "[CE10] start $(date '+%F %T')"
echo "[CE10] root=${ROOT}"

"${PY}" "${ROOT}/scripts/prepare_splits.py" \
  --forget10 "/home/likunhao/likunhao52shared/datasets/TOFU/re_split/forget10.json" \
  --retain90 "/home/likunhao/likunhao52shared/datasets/TOFU/re_split/retain90.json" \
  --output-dir "${ROOT}/splits" \
  --rounds 5 \
  --retain-replay-size 1500 \
  --seed 42 | tee "${ROOT}/logs/prepare_splits.log"

"${PY}" "${ROOT}/scripts/run_ce10_sequential.py" \
  --config "${ROOT}/config/default_paths.json" \
  --rounds 1 2 3 4 5 \
  --initial-max-steps 260 \
  --extra-steps 120 \
  --eval-forget-limit 400 \
  --eval-retain-limit 400

"${PY}" "${ROOT}/scripts/aggregate_results.py" \
  --manifest "${ROOT}/splits/round_manifest.json" \
  --output-dir "${ROOT}/results"

echo "[CE10] done $(date '+%F %T')"
