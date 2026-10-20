#!/usr/bin/env bash
# Run DeltaJP (Niu et al., 2402.02309) on all compatible benchmarks.
# DeltaJP fits ONE universal L_inf perturbation (delta) on the LLaVA surrogate
# toward affirmative targets, then ADDS it to each behavior's own image. It
# therefore only runs on datasets whose behaviors carry a per-behavior image:
# AdvBenchM, HADES, MM-SafetyBench (AdvBench / SafeBench are text-only -> skipped).
#
# Note: delta is cached only within a process, so it is re-optimized once per
# dataset (train_num x steps on the surrogate; modest). To optimize it a single
# time and reuse, delta persistence must be added to the attack (ask if wanted).
#
# Usage:
#   bash attack_scripts/run_deltajp.sh                       # full, 3 datasets
#   DATASETS="AdvBenchM" bash attack_scripts/run_deltajp.sh  # one dataset
#   LIMIT=20 bash attack_scripts/run_deltajp.sh              # per-dataset subset (smoke)
#   STEPS=200 TRAIN_NUM=50 bash attack_scripts/run_deltajp.sh  # stronger delta
#   MODEL=llava_v1_5 bash ...
set -uo pipefail
export PYTHONNOUSERSITE=1
PYTHON=${PYTHON:-python}
cd "$(dirname "$0")/.."

ATTACK="DeltaJP"
MODEL=${MODEL:-llava_v1_5}
LIMIT=${LIMIT:-0}
DATASETS=${DATASETS:-"AdvBenchM HADES MM-SafetyBench"}
STEPS=${STEPS:-}
TRAIN_NUM=${TRAIN_NUM:-}
limit_flag() { [ "$LIMIT" -gt 0 ] && echo "--limit $LIMIT" || echo ""; }

# Build config: default DeltaJP.yaml, optionally overriding steps / train_num.
if [ -n "$STEPS" ] || [ -n "$TRAIN_NUM" ]; then
  CONFIG="/tmp/deltajp_$$.yaml"
  {
    echo "surrogate: LLaVA_v1_5"
    echo "train_dataset: AdvBenchM"
    echo "train_num: ${TRAIN_NUM:-25}"
    echo "epsilon: 16"
    echo "alpha: 0.5"
    echo "steps: ${STEPS:-100}"
    echo "batch_size: 8"
    echo "seed: 0"
  } > "$CONFIG"
else
  CONFIG="configs/attacks/DeltaJP.yaml"
fi
echo "config: $CONFIG"; cat "$CONFIG"

for d in $DATASETS; do
  run_dir="runs/${ATTACK}_${d}"
  echo "=================================================================="
  echo ">>> $ATTACK x $d   (limit=$LIMIT)   -> $run_dir"
  echo "=================================================================="
  "$PYTHON" run_generate_test_cases.py --attack "$ATTACK" --dataset "$d" \
      --config "$CONFIG" --save_dir "$run_dir" $(limit_flag) \
      || { echo "!! stage1 failed: $ATTACK/$d"; continue; }
  "$PYTHON" run_generate_completions.py --run_dir "$run_dir" --model "$MODEL" \
      || { echo "!! stage2 failed: $ATTACK/$d"; continue; }
  "$PYTHON" run_evaluate_completions.py --run_dir "$run_dir" --model "$MODEL" \
      || { echo "!! stage3 failed: $ATTACK/$d"; continue; }
done
echo "Done. ASR summaries appended to asr_results.jsonl"
