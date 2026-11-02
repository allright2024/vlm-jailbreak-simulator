#!/usr/bin/env bash
set -Eeuo pipefail
run_dir=/home/jovyan/jailbreak/vlm-jailbreak-simulator/runs/ultrabreak_train_20261024
run_python=/home/jovyan/.conda/envs/jb_env/bin/python
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=/home/jovyan/jailbreak/runtime/ultrabreak_451_deps
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec 9> "$run_dir/worker.lock"
flock -n 9 || exit 73
exec >> "$run_dir/background.log" 2>&1
trap 'run_exit=$?; printf "%s\n" "$run_exit" > "$run_dir/exit.code"; date -u +%FT%TZ > "$run_dir/finished_at.txt"; exit "$run_exit"' EXIT
date -u +%FT%TZ > "$run_dir/started_at.txt"
"$run_python" -B -u "$run_dir/train.py"
"$run_python" -B -u "$run_dir/pipeline.py" infer --model qwen2
"$run_python" -B -u "$run_dir/pipeline.py" infer --model qwen25
"$run_python" -B -u "$run_dir/pipeline.py" judge
"$run_python" -B -u "$run_dir/pipeline.py" summarize
