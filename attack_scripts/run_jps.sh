#!/usr/bin/env bash
# Run JPS on its compatible benchmark(s): test-cases -> completions -> judge.
# Requires the upstream-only advbench_subset.csv and clean.jpeg under data/JPS.
#
# Usage:
#   bash attack_scripts/run_jps.sh              # full datasets, slow
#   LIMIT=50 bash attack_scripts/run_jps.sh     # larger subset
#   LIMIT=0  bash attack_scripts/run_jps.sh     # full dataset (slow)
#   DATASETS="AdvBench" bash attack_scripts/run_jps.sh   # restrict datasets
#   MODEL=llava_v1_5 ...                        # cross-model transfer target
set -uo pipefail
export PYTHONNOUSERSITE=1
PYTHON=${PYTHON:-python}
cd "$(dirname "$0")/.."                                      # repo root

ATTACK="JPS"
MODEL=${MODEL:-qwen2_vl_7b}
CONFIG=${CONFIG:-configs/attacks/JPS.yaml}
LIMIT=${LIMIT:-0}
DATASETS=${DATASETS:-"AdvBench HarmBench MM-SafetyBench"}
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
      --max_new_tokens 256 \
      || { echo "!! stage2 failed: $ATTACK/$d"; continue; }
  "$PYTHON" run_evaluate_completions.py --run_dir "$run_dir" --model "$MODEL" \
      --judge harmbench --judge_model cais/HarmBench-Llama-2-13b-cls \
      --judge_batch_size 12 --save_path "$run_dir/results_${MODEL}_harmbench.jsonl" \
      || { echo "!! stage3 failed: $ATTACK/$d"; continue; }
  "$PYTHON" run_evaluate_completions.py --run_dir "$run_dir" --model "$MODEL" \
      --judge llama_guard --judge_model meta-llama/Llama-Guard-3-8B \
      --save_path "$run_dir/results_${MODEL}_llamaguard3.jsonl" \
      || { echo "!! Llama Guard 3 stage failed: $ATTACK/$d"; continue; }
done
echo "Done. ASR summaries appended to asr_results.jsonl"
