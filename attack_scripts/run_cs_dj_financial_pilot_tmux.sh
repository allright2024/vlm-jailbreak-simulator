#!/usr/bin/env bash
# Persistent, resumable CS-DJ Financial 150 x LLaVA-OneVision-Chat-7B pilot.
# Launch this script inside tmux; it prepares assets, builds attacks, generates
# completions, judges them with Beaver-Dam, and writes a final summary.
set -Eeuo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/home/jovyan/.conda/envs/jb_env/bin/python}
RUN_DIR=${RUN_DIR:-$REPO_ROOT/runs/cs_dj_financial_llava_onevision_seed2025_20261022}
CONFIG=${CONFIG:-$REPO_ROOT/configs/attacks/CS-DJ_financial_pilot.yaml}
LOG_PATH=$RUN_DIR/pipeline.log
STATUS_WRITER=$REPO_ROOT/attack_scripts/write_cs_dj_pilot_status.py
STAGES=$REPO_ROOT/attack_scripts/cs_dj_financial_pilot_stages.py
CURRENT_STAGE=initializing

mkdir -p "$RUN_DIR"
exec > >(tee -a "$LOG_PATH") 2>&1
cd "$REPO_ROOT"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

status() {
  "$PYTHON_BIN" "$STATUS_WRITER" \
    --run-dir "$RUN_DIR" --state "$1" --stage "$2" --detail "${3:-}"
}

on_error() {
  local code=$?
  "$PYTHON_BIN" "$STATUS_WRITER" \
    --run-dir "$RUN_DIR" --state failed --stage "$CURRENT_STAGE" \
    --detail "pipeline command failed; inspect pipeline.log" --exit-code "$code" || true
  echo "CS-DJ pilot failed in stage '$CURRENT_STAGE' with exit code $code"
  exit "$code"
}
trap on_error ERR

echo "CS-DJ Financial pilot started at $(date --iso-8601=seconds)"
echo "Repository: $REPO_ROOT"
echo "Run directory: $RUN_DIR"
echo "Python: $PYTHON_BIN"

CURRENT_STAGE=assets
status running "$CURRENT_STAGE" "preparing pinned datasets and model snapshots"
if [ ! -f "$RUN_DIR/.assets.ok" ]; then
  "$PYTHON_BIN" attack_scripts/prepare_cs_dj_financial_pilot.py
  touch "$RUN_DIR/.assets.ok"
fi

CURRENT_STAGE=stage1_attack_generation
status running "$CURRENT_STAGE" "building 150 Financial CS-DJ composite images"
if [ ! -f "$RUN_DIR/.stage1.ok" ]; then
  "$PYTHON_BIN" run_generate_test_cases.py \
    --attack CS-DJ \
    --dataset CS-DJ \
    --config "$CONFIG" \
    --save_dir "$RUN_DIR" \
    --categories Financial \
    --limit 150 \
    --verbose
  "$PYTHON_BIN" "$STAGES" validate --run-dir "$RUN_DIR"
  touch "$RUN_DIR/.stage1.ok"
else
  "$PYTHON_BIN" "$STAGES" validate --run-dir "$RUN_DIR"
fi

CURRENT_STAGE=stage2_llava_onevision
status running "$CURRENT_STAGE" "resumable greedy generation, max_new_tokens=1000"
"$PYTHON_BIN" "$STAGES" complete --run-dir "$RUN_DIR"
touch "$RUN_DIR/.stage2.ok"

CURRENT_STAGE=stage3_beaver_dam
status running "$CURRENT_STAGE" "resumable Beaver-Dam scoring at threshold 0.5"
"$PYTHON_BIN" "$STAGES" judge --run-dir "$RUN_DIR"
touch "$RUN_DIR/.stage3.ok"

CURRENT_STAGE=summary
status running "$CURRENT_STAGE" "computing ASR and paper-reference delta"
"$PYTHON_BIN" "$STAGES" summarize --run-dir "$RUN_DIR"
touch "$RUN_DIR/.summary.ok"

status complete complete "all 150 Financial examples generated and judged"
echo "CS-DJ Financial pilot completed at $(date --iso-8601=seconds)"
echo "Summary: $RUN_DIR/pilot_summary.json"
