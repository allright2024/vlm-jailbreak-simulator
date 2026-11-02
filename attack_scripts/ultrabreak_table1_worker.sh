#!/usr/bin/env bash
# UltraBreak Table 1 reproduction, detached in tmux:
#   tmux new -d -s ub-table1 'bash attack_scripts/ultrabreak_table1_worker.sh'
# Phase 1: optimise the paper-iteration image while the released image is evaluated (sequential targets).
# Phase 2: evaluate the paper1300 image, one process per target in parallel.
# Phase 3: evaluate the official5000 image the same way. Phase 4: judge everything, report.
set -uo pipefail
cd "$(dirname "$0")/.."
RUN_DIR=${RUN_DIR:-runs/ultrabreak_table1_20261024}
PY=${PY:-/home/jovyan/.conda/envs/jb_env/bin/python}
S=attack_scripts/ultrabreak_table1_reproduction.py
export PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} TOKENIZERS_PARALLELISM=false
mkdir -p "$RUN_DIR/logs"
exec 9>"$RUN_DIR/worker.lock"; flock -n 9 || { echo "already running"; exit 75; }
exec >>"$RUN_DIR/worker.log" 2>&1
status() { printf '{"status":"%s","phase":"%s","updated_at":"%s","pid":%d}\n' "$1" "$2" "$(date -Is)" "$$" > "$RUN_DIR/worker_status.json"; }
run() { local name=$1; shift; "$PY" "$S" "$@" --out "$RUN_DIR" > "$RUN_DIR/logs/$name.log" 2>&1; local rc=$?; echo "[$(date -Is)] $name exit=$rc"; return $rc; }

status running phase1; echo "===== $(date -Is) phase 1: optimize paper1300 || generate released ====="
run optimize optimize &
P_OPT=$!
run gen_released generate --images released &
P_REL=$!
wait $P_OPT; RC_OPT=$?
wait $P_REL; RC_REL=$?
[ $RC_OPT -eq 0 ] || { status failed optimize; echo "optimize failed"; }

status running phase2; echo "===== $(date -Is) phase 2: generate paper1300 (parallel targets) ====="
if [ $RC_OPT -eq 0 ]; then
  for t in qwen2_vl_7b qwen2_5_vl_7b llava_v1_6_mistral; do run "gen_paper1300_$t" generate --images paper1300 --targets $t & done; wait
fi

status running phase3; echo "===== $(date -Is) phase 3: generate official5000 (parallel targets) ====="
for t in qwen2_vl_7b qwen2_5_vl_7b llava_v1_6_mistral; do run "gen_official5000_$t" generate --images official5000 --targets $t & done; wait

status running phase4; echo "===== $(date -Is) phase 4: judge + report ====="
run judge judge --images released paper1300 official5000 || { status failed judge; exit 1; }
run report report || { status failed report; exit 1; }
status done report; echo "===== $(date -Is) done ====="
