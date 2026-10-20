#!/usr/bin/env bash
# Run JOOD (OOD image/text mixing) on its compatible benchmark.
# JOOD OOD-mixes each harmful item with benign ones; it needs AdvBenchM (which
# carries the per-behavior scenario image + benign item pool). Builds one test
# case per (harmless item x lambda); Pass@k scores "succeed on any mix", so the
# test-case count is large (behaviors x #benign x #lambda).
#
# Usage:
#   bash attack_scripts/run_jood.sh                      # full official 4-augmentation grid
#   LIMIT=20 bash attack_scripts/run_jood.sh             # behavior subset (smoke)
#   DATASETS="AdvBenchM" bash attack_scripts/run_jood.sh # (only AdvBenchM is compatible)
#   MODEL=llava_v1_5 bash ...
set -uo pipefail
export PYTHONNOUSERSITE=1
PYTHON=${PYTHON:-python}
cd "$(dirname "$0")/.."

ATTACK="JOOD"
MODEL=${MODEL:-gpt-4-turbo-2024-04-09}
LIMIT=${LIMIT:-0}
DATASETS=${DATASETS:-"AdvBenchM"}
CONFIG=${CONFIG:-configs/attacks/JOOD.yaml}
AUGS=${AUGS:-"cutmix_original mixup cutmix_resizemix randaug1"}
limit_flag() { [ "$LIMIT" -gt 0 ] && echo "--limit $LIMIT" || echo ""; }
echo "config: $CONFIG"; cat "$CONFIG"

for aug in $AUGS; do
  AUG_CONFIG=$(mktemp /tmp/jood_official_XXXXXX.yaml)
  sed "s/^aug:.*/aug: ${aug}/" "$CONFIG" > "$AUG_CONFIG"
  for d in $DATASETS; do
  run_dir="runs/${ATTACK}_${aug}_${d}"
  echo "=================================================================="
  echo ">>> $ATTACK/$aug x $d   (limit=$LIMIT)   -> $run_dir"
  echo "=================================================================="
  "$PYTHON" run_generate_test_cases.py --attack "$ATTACK" --dataset "$d" \
      --config "$AUG_CONFIG" --save_dir "$run_dir" $(limit_flag) \
      || { echo "!! stage1 failed: $ATTACK/$d"; continue; }
  "$PYTHON" run_generate_completions.py --run_dir "$run_dir" --model "$MODEL" \
      --max_new_tokens 4096 --temperature 1.0 --retry_limit 10 \
      || { echo "!! stage2 failed: $ATTACK/$d"; continue; }
  "$PYTHON" run_evaluate_completions.py --run_dir "$run_dir" --model "$MODEL" \
      --judge llama_guard --judge_model meta-llama/Meta-Llama-Guard-2-8B \
      --save_path "$run_dir/results_${MODEL}_llamaguard2.jsonl" \
      || { echo "!! stage3 failed: $ATTACK/$d"; continue; }
done
  rm -f "$AUG_CONFIG"
done
echo "Done. ASR summaries appended to asr_results.jsonl"
