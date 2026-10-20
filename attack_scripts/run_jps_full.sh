#!/usr/bin/env bash
# Run JPS on FULL datasets efficiently: optimize the universal image ONCE,
# then reuse it for every dataset (no per-dataset re-optimization).
# Optimizes ONE image + steer prompt, then reuses both. The default retains the
# official attack settings and local/HF Qwen2.5-14B agents. The upstream-only
# advbench_subset.csv and clean.jpeg must be present under data/JPS.
#
# Usage:
#   bash attack_scripts/run_jps_full.sh
#   MODEL=llava_v1_5 bash attack_scripts/run_jps_full.sh  # transfer evaluation
#   DATASETS="AdvBench SafeBench" bash attack_scripts/run_jps_full.sh
#   OPT_CONFIG=configs/attacks/JPS.yaml bash ...   # optimization config (defaults shown below)
set -uo pipefail
export PYTHONNOUSERSITE=1
PYTHON=${PYTHON:-python}
cd "$(dirname "$0")/.."

ATTACK="JPS"
MODEL=${MODEL:-qwen2_vl_7b}
DATASETS=${DATASETS:-"AdvBench HarmBench MM-SafetyBench"}
RUNS=${RUNS:-runs/JPS_full}
OPT_CONFIG=${OPT_CONFIG:-configs/attacks/JPS.yaml}   # edit to lower iters/epochs for speed
OPT_DIR="$RUNS/__optimize__"
IMG="$OPT_DIR/images/jps_adv.png"
STEER="$OPT_DIR/images/jps_adv_steer.txt"

# ---- Step 1: optimize the universal image ONCE (eval set irrelevant; use 1 behavior) ----
echo "=================================================================="
echo ">>> $ATTACK : optimizing universal image once -> $IMG"
echo "=================================================================="
# Re-optimize if the artifact is missing OR the optimization config changed
# (SHA guard), so a stale image built from a different config is never reused.
SHA_FILE="$OPT_DIR/opt_config.sha256"
CUR_SHA=$(sha256sum "$OPT_CONFIG" | awk '{print $1}')
if [ -f "$IMG" ] && [ -f "$STEER" ] && \
    [ "$(cat "$SHA_FILE" 2>/dev/null)" = "$CUR_SHA" ] && \
    "$PYTHON" -m core.validation.artifact_lineage --attack "$ATTACK" --config "$OPT_CONFIG" --artifact "$IMG"; then
  echo ">>> verified optimization cache"
else
  [ -f "$IMG" ] && echo ">>> cache config/input/code/lineage mismatch -> re-optimizing"
  rm -f "$IMG" "$IMG.lineage.json" "$STEER"
  "$PYTHON" run_generate_test_cases.py --attack "$ATTACK" --dataset AdvBench \
      --config "$OPT_CONFIG" --save_dir "$OPT_DIR" --limit 1 \
      || { echo "!! optimization failed"; exit 1; }
  "$PYTHON" -m core.validation.artifact_lineage --attack "$ATTACK" --config "$OPT_CONFIG" --artifact "$IMG" \
      || { echo "!! optimized artifact lineage verification failed"; exit 1; }
  mkdir -p "$OPT_DIR"; echo "$CUR_SHA" > "$SHA_FILE"
fi
[ -f "$IMG" ] || { echo "!! universal image not found at $IMG"; exit 1; }
[ -f "$STEER" ] || { echo "!! JPS steer prompt not found at $STEER (artifact incomplete)"; exit 1; }

# ---- Step 2: reuse the image on each full dataset (skip optimization) ----
REUSE_CFG=$(mktemp /tmp/jps_reuse_XXXXXX.yaml)
trap 'rm -f "$REUSE_CFG"' EXIT
# Inherit the FULL optimization config (profile, surrogate, train_csv, ...) so
# the reuse run keeps the same train/eval split and provenance; only add the
# prebuilt image + steer prompt.
cat "$OPT_CONFIG" > "$REUSE_CFG"
printf '\nadv_image: RUNS/__optimize__/images/IMG\nsteer_path: RUNS/__optimize__/images/jps_adv_steer.txt\n' | sed "s#RUNS#$RUNS#g; s#IMG#jps_adv.png#g" >> "$REUSE_CFG"
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
echo "Done. ASR summaries in asr_results.jsonl"
