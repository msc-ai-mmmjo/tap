#!/bin/bash
# Train all 9 robustness heads sequentially.
# Usage: bash olmo_tap/experiments/robustness/run_all.sh
for SHARD in $(seq 0 8); do
    echo "=== Training shard $SHARD / 8 (epochs=1)"
    pixi run -e cuda python -m olmo_tap.experiments.robustness.training --shard-id "$SHARD"
done
