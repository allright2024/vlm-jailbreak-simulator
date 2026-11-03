#!/usr/bin/env bash
# All 500 SafeBench questions, FigStep and vanilla, five stochastic attempts each.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/figstep_triton_20261022}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/tmp/figstep_inductor_20261022}
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-2}
PYTHON=${PYTHON:-/home/jovyan/.conda/envs/jb_env/bin/python}
RUN_DIR=${RUN_DIR:-runs/figstep_reproduction_20261022}
if [[ ! -f "$RUN_DIR/manifest.json" ]]; then
  "$PYTHON" attack_scripts/figstep_reproduction.py prepare --run-dir "$RUN_DIR"
fi
"$PYTHON" attack_scripts/figstep_reproduction.py generate --run-dir "$RUN_DIR"
"$PYTHON" attack_scripts/figstep_reproduction.py judge --run-dir "$RUN_DIR"
"$PYTHON" attack_scripts/figstep_reproduction.py summary --run-dir "$RUN_DIR"
