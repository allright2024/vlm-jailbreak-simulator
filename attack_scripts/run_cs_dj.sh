#!/usr/bin/env bash
# Run CS-DJ through test-case generation, target inference, and common judging.
#
# Usage:
#   LIMIT=10 bash attack_scripts/run_cs_dj.sh
#   MODEL=gpt-4o-mini DATASETS="CS-DJ" ...
set -uo pipefail
export PYTHONNOUSERSITE=1
PYTHON=${PYTHON:-python}
cd "$(dirname "$0")/.."

ATTACK="CS-DJ"
MODEL=${MODEL:-gpt-4o-mini}
CONFIG=${CONFIG:-configs/attacks/CS-DJ.yaml}
LIMIT=${LIMIT:-0}
DATASETS=${DATASETS:-"CS-DJ"}
limit_flag() { [ "$LIMIT" -gt 0 ] && echo "--limit $LIMIT" || echo ""; }

for d in $DATASETS; do
  run_dir="runs/${ATTACK}_${d}"
  echo "=================================================================="
  echo ">>> $ATTACK x $d   (limit=$LIMIT)   -> $run_dir"
  echo "=================================================================="
  "$PYTHON" run_generate_test_cases.py --attack "$ATTACK" --dataset "$d" \
      --config "$CONFIG" --save_dir "$run_dir" $(limit_flag) \
      || { echo "!! stage1 failed: $ATTACK/$d"; continue; }
  "$PYTHON" run_generate_completions.py --run_dir "$run_dir" --model "$MODEL" \
      --max_new_tokens 1000 --temperature 0.1 --retry_limit 1 \
      || { echo "!! stage2 failed: $ATTACK/$d"; continue; }
  "$PYTHON" run_evaluate_completions.py --run_dir "$run_dir" --model "$MODEL" \
      --judge beaver_dam --judge_model PKU-Alignment/beaver-dam-7b \
      --judge_batch_size 1 \
      || { echo "!! stage3 failed: $ATTACK/$d"; continue; }
done
echo "Done. ASR summaries appended to asr_results.jsonl"
