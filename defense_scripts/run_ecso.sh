#!/usr/bin/env bash
# Run the official MM-SafetyBench ECSO profile on an existing stage-1 run.
set -euo pipefail
export PYTHONNOUSERSITE=1
PYTHON=${PYTHON:-python}
cd "$(dirname "$0")/.."

: "${RUN_DIR:?Set RUN_DIR to a stage-1 directory containing test_cases.jsonl}"
MODEL=${MODEL:-llava_v1_5}
DEFENSE_CONFIG=${DEFENSE_CONFIG:-configs/defense_configs/ECSO.yaml}
LIMIT=${LIMIT:-0}
EVALUATE=${EVALUATE:-0}
JUDGE_BATCH_SIZE=${JUDGE_BATCH_SIZE:-4}

completion_path="$RUN_DIR/completions_${MODEL}__defense-ECSO.jsonl"
limit_args=()
if [ "$LIMIT" -gt 0 ]; then
  limit_args=(--limit "$LIMIT")
fi

"$PYTHON" run_generate_completions.py \
  --run_dir "$RUN_DIR" \
  --model "$MODEL" \
  --defense ECSO \
  --defense_config "$DEFENSE_CONFIG" \
  --save_path "$completion_path" \
  --batch_size 1 \
  --verbose \
  "${limit_args[@]}"

if [ "$EVALUATE" = "1" ]; then
  "$PYTHON" run_evaluate_completions.py \
    --run_dir "$RUN_DIR" \
    --model "$MODEL" \
    --completions_path "$completion_path" \
    --save_path "$RUN_DIR/results_${MODEL}__defense-ECSO.jsonl" \
    --judge_batch_size "$JUDGE_BATCH_SIZE"
fi
