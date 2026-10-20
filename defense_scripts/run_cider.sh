#!/usr/bin/env bash
# Run official-profile CIDER over an existing unified stage-1 run.
#
# Example:
#   bash defense_scripts/setup_cider.sh
#   RUN_DIR=runs/full3_methods_20261017/FigStep LIMIT=1 \
#     bash defense_scripts/run_cider.sh
set -euo pipefail
export PYTHONNOUSERSITE=1
PYTHON=${PYTHON:-python}
cd "$(dirname "$0")/.."

: "${RUN_DIR:?Set RUN_DIR to a stage-1 directory containing test_cases.jsonl}"
MODEL=${MODEL:-llava_v1_5}
DEFENSE_CONFIG=${DEFENSE_CONFIG:-configs/defense_configs/CIDER.yaml}
LIMIT=${LIMIT:-0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
EVALUATE=${EVALUATE:-0}

if [ "$MODEL" != "llava_v1_5" ] && [ "$MODEL" != "LLaVA_v1_5" ]; then
  echo "This CIDER adapter reuses the registered LLaVA-1.5 encoder; MODEL must be llava_v1_5." >&2
  exit 2
fi

completion_path="$RUN_DIR/completions_${MODEL}__defense-CIDER.jsonl"
limit_args=()
if [ "$LIMIT" -gt 0 ]; then
  limit_args=(--limit "$LIMIT")
fi

"$PYTHON" run_generate_completions.py \
  --run_dir "$RUN_DIR" \
  --model "$MODEL" \
  --defense CIDER \
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
    --save_path "$RUN_DIR/results_${MODEL}__defense-CIDER.jsonl"
fi
