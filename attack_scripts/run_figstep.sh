#!/usr/bin/env bash
# Run FigStep on its compatible benchmark(s): test-cases -> completions -> judge.
# Offline typographic attack (fast).
#
# Usage:
#   bash attack_scripts/run_figstep.sh              # LIMIT=5 per dataset
#   LIMIT=50 bash attack_scripts/run_figstep.sh     # larger subset
#   LIMIT=0  bash attack_scripts/run_figstep.sh     # full dataset (slow)
#   DATASETS="AdvBench" bash attack_scripts/run_figstep.sh   # restrict datasets
#   MODEL=llava_v1_5 ...                                    # target model
set -uo pipefail
export PYTHONNOUSERSITE=1
PYTHON=${PYTHON:-python}
cd "$(dirname "$0")/.."                                      # repo root

ATTACK="FigStep"
MODEL=${MODEL:-llava_v1_5}
LIMIT=${LIMIT:-0}
DATASETS=${DATASETS:-"SafeBench AdvBench AdvBenchM HADES MM-SafetyBench"}
limit_flag() { [ "$LIMIT" -gt 0 ] && echo "--limit $LIMIT" || echo ""; }

for d in $DATASETS; do
  run_dir="runs/${ATTACK}_${d}"
    extra=""
  echo "=================================================================="
  echo ">>> $ATTACK x $d   (limit=$LIMIT)   -> $run_dir"
  echo "=================================================================="
  "$PYTHON" run_generate_test_cases.py --attack "$ATTACK" --dataset "$d" \
      --save_dir "$run_dir" $(limit_flag) $extra || { echo "!! stage1 failed: $ATTACK/$d"; continue; }
  "$PYTHON" run_generate_completions.py --run_dir "$run_dir" --model "$MODEL" \
      || { echo "!! stage2 failed: $ATTACK/$d"; continue; }
  "$PYTHON" run_evaluate_completions.py --run_dir "$run_dir" --model "$MODEL" \
      || { echo "!! stage3 failed: $ATTACK/$d"; continue; }
done
echo "Done. ASR summaries appended to asr_results.jsonl"
