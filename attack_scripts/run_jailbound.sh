#!/usr/bin/env bash
# JailBound: boundary probing -> joint image/text crossing -> target -> judge.
# The first stage is expensive: 128 probe forwards (64 safe/unsafe pairs), then
# 100 gradient iterations for each requested evaluation behavior.
#
# Usage:
#   LIMIT=1 bash attack_scripts/run_jailbound.sh
#   BOUNDARY=/path/jailbound_boundaries.pt LIMIT=3 bash ...
set -uo pipefail
export PYTHONNOUSERSITE=1
PYTHON=${PYTHON:-python}
cd "$(dirname "$0")/.."

ATTACK=JailBound
DATASET=${DATASET:-MM-SafetyBench}
MODEL=${MODEL:-llava_v1_5}
LIMIT=${LIMIT:-1}
CONFIG=${CONFIG:-configs/attacks/JailBound.yaml}
TAG=${TAG:-}
RUN_DIR=${RUN_DIR:-runs/${ATTACK}_${DATASET}${TAG:+_$TAG}}

config_path="$CONFIG"
temporary_config=""
if [ -n "${BOUNDARY:-}" ]; then
  temporary_config=$(mktemp /tmp/jailbound_config.XXXXXX.yaml)
  cp "$CONFIG" "$temporary_config"
  printf '\nboundary_artifact: %s\n' "$BOUNDARY" >> "$temporary_config"
  config_path="$temporary_config"
fi
trap 'if [ -n "$temporary_config" ]; then rm -f "$temporary_config"; fi' EXIT

"$PYTHON" run_generate_test_cases.py \
  --attack "$ATTACK" --dataset "$DATASET" --limit "$LIMIT" \
  --config "$config_path" --save_dir "$RUN_DIR" || exit 1
"$PYTHON" run_generate_completions.py \
  --run_dir "$RUN_DIR" --model "$MODEL" || exit 1
"$PYTHON" run_evaluate_completions.py \
  --run_dir "$RUN_DIR" --model "$MODEL"
