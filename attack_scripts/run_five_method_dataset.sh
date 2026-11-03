#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NLTK_DATA="${NLTK_DATA:-$PWD/data/nltk_data}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
PYTHON_BIN="${PYTHON:-/home/jovyan/.conda/envs/jb_env/bin/python}"
exec "$PYTHON_BIN" run_generate_five_method_dataset.py "$@"
