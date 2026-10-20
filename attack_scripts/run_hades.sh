#!/usr/bin/env bash
# Run HADES on its native benchmark (test-cases -> completions -> judge).
# The default config learns a universal top-quarter patch with white-box CE
# updates toward "Sure..." and excludes its training split from evaluation.
# Pass HADES_official.yaml to replay all six released image steps instead.
#
# Usage:
#   bash attack_scripts/run_hades.sh                       # white-box 3,000 epochs
#   CONFIG=configs/attacks/HADES_official.yaml bash ...    # released-image Pass@6 replay
#   CONFIG=configs/attacks/HADES_white_box_mid.yaml \
#     TAG=mid bash ...                                     # lower-cost 100-epoch run
#   LIMIT=50 bash ...                                      # per-dataset subset
#   TAG=mid bash ...                                       # write to runs/HADES_<dataset>_mid
set -uo pipefail
export PYTHONNOUSERSITE=1
PYTHON=${PYTHON:-python}
cd "$(dirname "$0")/.."                                      # repo root

ATTACK="HADES"
MODEL=${MODEL:-llava_v1_5}
LIMIT=${LIMIT:-0}
DATASETS=${DATASETS:-"HADES"}
limit_flag() { [ "$LIMIT" -gt 0 ] && echo "--limit $LIMIT" || echo ""; }

CONFIG=${CONFIG:-configs/attacks/HADES.yaml}
# Optional suffix so a run does not overwrite an existing result directory.
TAG=${TAG:-}
echo "config:"; cat "$CONFIG"

for d in $DATASETS; do
  run_dir="runs/${ATTACK}_${d}${TAG:+_$TAG}"
  echo "=================================================================="
  echo ">>> $ATTACK x $d   (limit=$LIMIT, config=$CONFIG)   -> $run_dir"
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
