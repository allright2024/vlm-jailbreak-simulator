#!/usr/bin/env bash
# Run official-profile JailGuard over an existing unified stage-1 run.
#
# Example:
#   bash defense_scripts/setup_jailguard.sh
#   RUN_DIR=runs/full3_methods_20261017/FigStep LIMIT=1 \
#     bash defense_scripts/run_jailguard.sh
set -euo pipefail
export PYTHONNOUSERSITE=1
PYTHON=${PYTHON:-python}
cd "$(dirname "$0")/.."

: "${RUN_DIR:?Set RUN_DIR to a stage-1 directory containing test_cases.jsonl}"
MODEL=${MODEL:-llava_v1_5}
DEFENSE_CONFIG=${DEFENSE_CONFIG:-configs/defense_configs/JailGuard.yaml}
LIMIT=${LIMIT:-0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
EVALUATE=${EVALUATE:-0}

completion_path="$RUN_DIR/completions_${MODEL}__defense-JailGuard.jsonl"
limit_args=()
if [ "$LIMIT" -gt 0 ]; then
  limit_args=(--limit "$LIMIT")
fi

"$PYTHON" run_generate_completions.py \
  --run_dir "$RUN_DIR" \
  --model "$MODEL" \
  --defense JailGuard \
  --defense_config "$DEFENSE_CONFIG" \
  --save_path "$completion_path" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --verbose \
  "${limit_args[@]}"

if [ "$EVALUATE" = "1" ]; then
  "$PYTHON" run_evaluate_completions.py \
    --run_dir "$RUN_DIR" \
    --model "$MODEL" \
    --completions_path "$completion_path" \
    --save_path "$RUN_DIR/results_${MODEL}__defense-JailGuard.jsonl"
fi
