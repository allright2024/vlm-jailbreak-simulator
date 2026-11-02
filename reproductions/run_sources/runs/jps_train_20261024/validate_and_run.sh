#!/usr/bin/env bash
set -Eeuo pipefail
run_dir=/home/jovyan/jailbreak/vlm-jailbreak-simulator/runs/jps_train_20261024
run_python=/home/jovyan/.conda/envs/jb_env/bin/python
cd /home/jovyan/jailbreak/vlm-jailbreak-simulator
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=/home/jovyan/jailbreak/runtime/jps_448_deps
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec >> "$run_dir/validation.log" 2>&1
trap 'run_exit=$?; printf "%s\n" "$run_exit" > "$run_dir/validation_exit.code"; date -u +%FT%TZ > "$run_dir/validation_finished_at.txt"; exit "$run_exit"' EXIT
date -u +%FT%TZ > "$run_dir/validation_started_at.txt"
"$run_python" -B -u "$run_dir/train_run.py" prepare
"$run_python" -B -u "$run_dir/train_run.py" responses --round 0 --smoke
"$run_python" -B -u "$run_dir/train_run.py" revise --round 1 --smoke
"$run_python" -B -u "$run_dir/train_run.py" optimize --round 1 --smoke
printf '0\n' > "$run_dir/validation_exit.code"
date -u +%FT%TZ > "$run_dir/validation_finished_at.txt"
exec bash "$run_dir/run_background.sh"
