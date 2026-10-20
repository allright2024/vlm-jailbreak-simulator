#!/usr/bin/env bash
# MML: released encrypted image -> target -> interleaved official CLAS judge.
#
# Usage:
#   bash attack_scripts/run_mml.sh
#   LIMIT=10 bash attack_scripts/run_mml.sh
#   VARIANTS="mirror" bash attack_scripts/run_mml.sh
set -uo pipefail
export PYTHONNOUSERSITE=1
PYTHON=${PYTHON:-python}
cd "$(dirname "$0")/.."

ATTACK="MML"
MODEL=${MODEL:-gpt-4o-2024-08-06}
JUDGE_MODEL=${JUDGE_MODEL:-gpt-4o-mini-2024-07-18}
LIMIT=${LIMIT:-0}
DATASETS=${DATASETS:-"SafeBench MM-SafetyBench HADES"}
CONFIG=${CONFIG:-configs/attacks/MML.yaml}
VARIANTS=${VARIANTS:-"word_replacement mirror rotate base64"}
EVALUATE_LLAMA_GUARD=${EVALUATE_LLAMA_GUARD:-0}
limit_flag() { [ "$LIMIT" -gt 0 ] && echo "--limit $LIMIT" || echo ""; }
subset_flag() { [ "$LIMIT" -gt 0 ] && echo "--allow_subset" || echo ""; }

for variant in $VARIANTS; do
  VARIANT_CONFIG=$(mktemp /tmp/mml_official_XXXXXX.yaml)
  sed "s/^variant:.*/variant: ${variant}/" "$CONFIG" > "$VARIANT_CONFIG"
  for d in $DATASETS; do
  run_dir="runs/${ATTACK}_${variant}_${d}"
  echo "=================================================================="
  echo ">>> $ATTACK/$variant x $d   (limit=$LIMIT)   -> $run_dir"
  echo "=================================================================="
  "$PYTHON" run_generate_test_cases.py --attack "$ATTACK" --dataset "$d" \
      --config "$VARIANT_CONFIG" --save_dir "$run_dir" $(limit_flag) \
      || { echo "!! stage1 failed: $ATTACK/$d"; continue; }
  "$PYTHON" run_generate_completions.py --protocol mml_official \
      --run_dir "$run_dir" --model "$MODEL" \
      --judge_model "$JUDGE_MODEL" --attempts 5 \
      --max_new_tokens 2000 --temperature 0.7 --retry_limit 3 \
      $(subset_flag) \
      || { echo "!! stage2 failed: $ATTACK/$d"; continue; }
  if [ "$EVALUATE_LLAMA_GUARD" = "1" ]; then
    echo ">>> paper cross-check with Llama Guard 3 (separate result file)"
    "$PYTHON" run_evaluate_completions.py --run_dir "$run_dir" --model "$MODEL" \
        --judge llama_guard --judge_model meta-llama/Llama-Guard-3-8B \
        --save_path "$run_dir/results_${MODEL}_llama_guard3.jsonl" \
        || { echo "!! stage3 failed: $ATTACK/$d"; continue; }
  fi
done
  rm -f "$VARIANT_CONFIG"
done

echo "Done. ASR summaries appended to asr_results.jsonl"
