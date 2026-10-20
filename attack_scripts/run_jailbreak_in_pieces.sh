#!/usr/bin/env bash
# Run Jailbreak In Pieces through generation, target inference, and judging.
#
# Usage:
#   bash attack_scripts/run_jailbreak_in_pieces.sh
# This reproduces the fixed released meth-pill demo once. Set EVALUATE=1 only
# for an additional simulator/HarmBench score; upstream publishes no judge.
set -uo pipefail
export PYTHONNOUSERSITE=1
PYTHON=${PYTHON:-python}
cd "$(dirname "$0")/.."

ATTACK="JailbreakInPieces"
MODEL=${MODEL:-llava_v1_5}
CONFIG=${CONFIG:-configs/attacks/JailbreakInPieces.yaml}
LIMIT=${LIMIT:-1}
DATASETS=${DATASETS:-"AdvBench"}
EVALUATE=${EVALUATE:-0}
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
      --max_new_tokens 100 --do_sample --temperature 0.6 --top_p 0.9 \
      || { echo "!! stage2 failed: $ATTACK/$d"; continue; }
  if [ "$EVALUATE" = "1" ]; then
    "$PYTHON" run_evaluate_completions.py --run_dir "$run_dir" --model "$MODEL" \
        || { echo "!! stage3 failed: $ATTACK/$d"; continue; }
  fi
done
echo "Done. ASR summaries appended to asr_results.jsonl"
