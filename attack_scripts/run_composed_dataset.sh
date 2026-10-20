#!/usr/bin/env bash
set -euo pipefail

# Examples:
#   PER_RECIPE=3 bash attack_scripts/run_composed_dataset.sh
#   RECIPES=P03,P04 SAVE_DIR=runs/composed_offline_smoke \
#     bash attack_scripts/run_composed_dataset.sh
# Omit PER_RECIPE and RECIPES for the configured 3,000-sample full run.

PYTHON_BIN="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/compositions/jailbreak_dataset_v3.yaml}"
SAVE_DIR="${SAVE_DIR:-runs/composed_jailbreak_v3}"
RECIPES="${RECIPES:-}"
PER_RECIPE="${PER_RECIPE:-}"

# Allow the MML word-replacement stage to use a repository-local NLTK data
# install (for example: python -m nltk.downloader -d data/nltk_data
# averaged_perceptron_tagger). An explicitly supplied NLTK_DATA wins.
if [[ -z "${NLTK_DATA:-}" && -d "data/nltk_data" ]]; then
  export NLTK_DATA="$PWD/data/nltk_data"
fi

args=(
  run_generate_composed_dataset.py
  --config "$CONFIG"
  --save-dir "$SAVE_DIR"
)

if [[ -n "$RECIPES" ]]; then
  args+=(--recipes "$RECIPES")
fi
if [[ -n "$PER_RECIPE" ]]; then
  args+=(--per-recipe "$PER_RECIPE")
fi

"$PYTHON_BIN" "${args[@]}"
"$PYTHON_BIN" -m core.validation.composed_dataset "$SAVE_DIR"
