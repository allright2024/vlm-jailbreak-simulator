#!/usr/bin/env bash
# Run UltraBreak on FULL datasets efficiently: optimize the universal image ONCE,
# then reuse it for every dataset (no per-dataset re-optimization).
# Optimizes ONE universal image on Qwen2-VL, then reuses it (transfer to the target).
#
# Usage:
#   bash attack_scripts/run_ultrabreak_full.sh
#   MODEL=llava_v1_5 bash attack_scripts/run_ultrabreak_full.sh
#   DATASETS="AdvBench SafeBench" bash attack_scripts/run_ultrabreak_full.sh
#   OPT_CONFIG=configs/attacks/UltraBreak.yaml bash ...   # optimization config (defaults shown below)
set -uo pipefail
export PYTHONNOUSERSITE=1
PYTHON=${PYTHON:-python}
cd "$(dirname "$0")/.."

ATTACK="UltraBreak"
MODEL=${MODEL:-llava_v1_5}
DATASETS=${DATASETS:-"SafeBench AdvBench AdvBenchM HADES MM-SafetyBench"}
RUNS=${RUNS:-runs/UltraBreak_full}
OPT_CONFIG=${OPT_CONFIG:-configs/attacks/UltraBreak.yaml}   # edit to lower iters/epochs for speed
OPT_DIR="$RUNS/__optimize__"
IMG="$OPT_DIR/images/universal_adv.png"

# ---- Step 1: optimize the universal image ONCE (eval set irrelevant; use 1 behavior) ----
echo "=================================================================="
echo ">>> $ATTACK : optimizing universal image once -> $IMG"
echo "=================================================================="
# Re-optimize if the artifact is missing OR the optimization config changed
# (SHA guard), so a stale image built from a different config is never reused.
SHA_FILE="$OPT_DIR/opt_config.sha256"
CUR_SHA=$(sha256sum "$OPT_CONFIG" | awk '{print $1}')
if [ -f "$IMG" ] && [ "$(cat "$SHA_FILE" 2>/dev/null)" = "$CUR_SHA" ] && \
    "$PYTHON" -m core.validation.artifact_lineage --attack "$ATTACK" --config "$OPT_CONFIG" --artifact "$IMG"; then
  echo ">>> verified optimization cache"
else
  [ -f "$IMG" ] && echo ">>> cache config/input/code/lineage mismatch -> re-optimizing"
  rm -f "$IMG" "$IMG.lineage.json"
  "$PYTHON" run_generate_test_cases.py --attack "$ATTACK" --dataset AdvBench \
      --config "$OPT_CONFIG" --save_dir "$OPT_DIR" --limit 1 \
      || { echo "!! optimization failed"; exit 1; }
  "$PYTHON" -m core.validation.artifact_lineage --attack "$ATTACK" --config "$OPT_CONFIG" --artifact "$IMG" \
      || { echo "!! optimized artifact lineage verification failed"; exit 1; }
  mkdir -p "$OPT_DIR"; echo "$CUR_SHA" > "$SHA_FILE"
fi
[ -f "$IMG" ] || { echo "!! universal image not found at $IMG"; exit 1; }

# ---- Step 2: reuse the image on each full dataset (skip optimization) ----
REUSE_CFG=$(mktemp /tmp/ultrabreak_reuse_XXXXXX.yaml)
trap 'rm -f "$REUSE_CFG"' EXIT
# Inherit the FULL optimization config (profile, surrogate, train_config/rows,
# ...) so the reuse run keeps the same train/eval split and provenance; only
# add the prebuilt artifact.
cat "$OPT_CONFIG" > "$REUSE_CFG"
printf '\nadv_image: RUNS/__optimize__/images/IMG\n' | sed "s#RUNS#$RUNS#g; s#IMG#universal_adv.png#g" >> "$REUSE_CFG"
echo "reuse config:"; cat "$REUSE_CFG"

for d in $DATASETS; do
  run_dir="$RUNS/$d"
  echo "=================================================================="
  echo ">>> $ATTACK x $d   (FULL)   -> $run_dir"
  echo "=================================================================="
  "$PYTHON" run_generate_test_cases.py --attack "$ATTACK" --dataset "$d" \
      --config "$REUSE_CFG" --save_dir "$run_dir" \
      || { echo "!! stage1 failed: $ATTACK/$d"; continue; }
  "$PYTHON" run_generate_completions.py --run_dir "$run_dir" --model "$MODEL" \
      || { echo "!! stage2 failed: $ATTACK/$d"; continue; }
  "$PYTHON" run_evaluate_completions.py --run_dir "$run_dir" --model "$MODEL" \
      || { echo "!! stage3 failed: $ATTACK/$d"; continue; }
done
echo "Done. ASR summaries in asr_results.jsonl"
