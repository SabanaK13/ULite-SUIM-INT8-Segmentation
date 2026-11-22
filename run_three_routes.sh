#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
experiment_config="${1:-experiment.json}"
experiment_seed="${2:-42}"
python workflow.py doctor --config "$experiment_config"
# Run prepare once and inspect its preview BEFORE invoking this script.
for experiment_route in fp32 amp_fp16 qat; do
  python workflow.py train --config "$experiment_config" --route "$experiment_route" --seed "$experiment_seed" --device cuda
  python workflow.py quantize --config "$experiment_config" --route "$experiment_route" --seed "$experiment_seed"
done
python workflow.py compare --config "$experiment_config" --seed "$experiment_seed"
# Final TEST evaluation is deliberately a separate explicit command.
