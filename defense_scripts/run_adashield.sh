#!/usr/bin/env bash
# Apply AdaShield to an existing unified stage-1 run.
#
# Static official profile (immediately runnable):
#   RUN_DIR=runs/FigStep_SafeBench bash defense_scripts/run_adashield.sh
#
# Adaptive inference (requires upstream-generated final_table.csv artifacts):
#   RUN_DIR=runs/FigStep_SafeBench \
#   DEFENSE_CONFIG=/path/to/AdaShield-A.yaml \
#   bash defense_scripts/run_adashield.sh
#
# Optional one-sample smoke / ASR evaluation:
#   RUN_DIR=runs/FigStep_SafeBench LIMIT=1 bash defense_scripts/run_adashield.sh
#   RUN_DIR=runs/FigStep_SafeBench EVALUATE=1 bash defense_scripts/run_adashield.sh
set -euo pipefail
export PYTHONNOUSERSITE=1
PYTHON=${PYTHON:-python}
cd "$(dirname "$0")/.."

: "${RUN_DIR:?Set RUN_DIR to a stage-1 directory containing test_cases.jsonl}"
MODEL=${MODEL:-llava_v1_5}
DEFENSE_CONFIG=${DEFENSE_CONFIG:-configs/defense_configs/AdaShield.yaml}
LIMIT=${LIMIT:-0}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
EVALUATE=${EVALUATE:-0}

completion_path="$RUN_DIR/completions_${MODEL}__defense-AdaShield.jsonl"
limit_args=()
if [ "$LIMIT" -gt 0 ]; then
  limit_args=(--limit "$LIMIT")
fi

"$PYTHON" run_generate_completions.py \
  --run_dir "$RUN_DIR" \
  --model "$MODEL" \
  --defense AdaShield \
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
    --save_path "$RUN_DIR/results_${MODEL}__defense-AdaShield.jsonl"
fi
