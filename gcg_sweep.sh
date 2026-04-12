#!/bin/bash
# Precompute GCG adversarial suffixes across all 9 shards (sequential, 1 GPU).
# Saves clean.pt and poisoned.pt per shard for direct use in training.

set -e
export PYTHONUNBUFFERED=1

for SHARD in $(seq 0 8); do
    echo "=========================================="
    echo "STARTING: gcg-shard-${SHARD} ($(date))"
    echo "=========================================="
    pixi run -e cuda python -m olmo_tap.experiments.robustness.precompute_gcg \
        --shard-id "$SHARD"
    echo "FINISHED: gcg-shard-${SHARD} ($(date))"
    echo ""
done
echo "ALL SHARDS COMPLETE ($(date))"
