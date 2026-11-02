#!/usr/bin/env bash
# Paper-era target (LLaVA v1 13B 336px) for the Table 6 reproduction, run detached in tmux:
#   tmux new -d -s jip-table6-paper 'bash attack_scripts/jip_table6_paper_worker.sh'
# Assumes `prepare` already ran (behaviors/test_cases/images under $RUN_DIR). Generation uses the
# original LLaVA runtime in asd_llava_venv; judge/report use jb_env, each in its own process.
set -uo pipefail
cd "$(dirname "$0")/.."
RUN_DIR=${RUN_DIR:-runs/jip_table6_advbench_20261023}
JB=${JB:-/home/jovyan/.conda/envs/jb_env/bin/python}
LLAVA_PY=${LLAVA_PY:-/home/jovyan/.conda/envs/asd_llava_venv/bin/python}
TARGET=llava_v1_13b_336px
ROUNDS=${ROUNDS:-25}
ARMS=${ARMS:-"released regenerated"}
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p "$RUN_DIR"
exec 9>"$RUN_DIR/paper_worker.lock"
flock -n 9 || { echo "paper worker already running"; exit 75; }
exec >>"$RUN_DIR/paper_worker.log" 2>&1
status() { printf '{"status":"%s","stage":"%s","target":"%s","updated_at":"%s","pid":%d}\n' "$1" "$2" "$TARGET" "$(date -Is)" "$$" > "$RUN_DIR/paper_worker_status.json"; }
status running generate; echo "===== $(date -Is) generate ($TARGET) ====="
"$LLAVA_PY" attack_scripts/jip_table6_llava_v1_generate.py --out "$RUN_DIR" --rounds "$ROUNDS" --arms $ARMS || { status failed generate; exit 1; }
status running judge; echo "===== $(date -Is) judge ====="
"$JB" attack_scripts/jailbreak_in_pieces_table6_reproduction.py judge --out "$RUN_DIR" --rounds "$ROUNDS" --arms $ARMS --target_model "$TARGET" || { status failed judge; exit 1; }
status running report; echo "===== $(date -Is) report ====="
"$JB" attack_scripts/jailbreak_in_pieces_table6_reproduction.py report --out "$RUN_DIR" --rounds "$ROUNDS" --arms $ARMS --target_model "$TARGET" || { status failed report; exit 1; }
status done report; echo "===== $(date -Is) done ====="
