#!/usr/bin/env bash
# Jailbreak in Pieces Table 6 reproduction, run detached in tmux:
#   tmux new -d -s jip-table6 'bash attack_scripts/jip_table6_background_worker.sh'
# Each stage runs in its own process so CLIP, LLaVA and the judge never share GPU memory.
set -uo pipefail
cd "$(dirname "$0")/.."
RUN_DIR=${RUN_DIR:-runs/jip_table6_advbench_20261023}
PYTHON=${PYTHON:-/home/jovyan/.conda/envs/jb_env/bin/python}
ROUNDS=${ROUNDS:-25}
ARMS=${ARMS:-"released regenerated"}
TARGET_MODEL=${TARGET_MODEL:-llava_v1_5}
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
mkdir -p "$RUN_DIR"
exec 9>"$RUN_DIR/background.lock"
flock -n 9 || { echo "JIP Table 6 worker is already running."; exit 75; }
exec >>"$RUN_DIR/background.log" 2>&1
SCRIPT=attack_scripts/jailbreak_in_pieces_table6_reproduction.py
status() { printf '{"status":"%s","stage":"%s","updated_at":"%s","pid":%d}\n' "$1" "$2" "$(date -Is)" "$$" > "$RUN_DIR/background_status.json"; }
for stage in prepare generate judge report; do
  status running "$stage"
  echo "===== $(date -Is) stage $stage ====="
  "$PYTHON" "$SCRIPT" "$stage" --out "$RUN_DIR" --rounds "$ROUNDS" --arms $ARMS --target_model "$TARGET_MODEL" || { status failed "$stage"; echo "!! stage $stage failed"; exit 1; }
done
status done report
echo "===== $(date -Is) all stages done ====="
