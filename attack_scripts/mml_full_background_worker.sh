#!/usr/bin/env bash
# Detached worker for full-500 MML preparation, inference, and validation.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_DIR=${RUN_DIR:-runs/mml_full_reproduction_20261022}
PYTHON=${PYTHON:-/home/jovyan/.conda/envs/jb_env/bin/python}
MML_WORKERS=${MML_WORKERS:-8}
export PYTHONNOUSERSITE=1
mkdir -p "$RUN_DIR"
exec 9>"$RUN_DIR/background.lock"
flock -n 9 || { echo "MML full worker is already running."; exit 75; }
exec >>"$RUN_DIR/background.log" 2>&1

write_status() {
    "$PYTHON" - "$RUN_DIR" "$1" "$2" "$$" <<'PY'
import datetime
import json
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

root = Path(sys.argv[1])
path = root / 'background_status.json'
state = json.loads(path.read_text()) if path.exists() else {}
now = datetime.datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')
if sys.argv[2] == 'running':
    state = {'started_at': now, 'ppl_excluded': True,
             'tmux_socket': 'mml-full-20261022', 'tmux_session': 'mml-full'}
state.update(status=sys.argv[2], updated_at=now, worker_pid=int(sys.argv[4]))
if sys.argv[2] != 'running':
    state['exit_code'] = int(sys.argv[3])
state['log_path'] = str((root / 'background.log').resolve())
temp = root / 'background_status.json.tmp'
temp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + '\n')
os.replace(temp, path)
PY
}

finish() {
    result=$?
    trap - EXIT
    if [[ "$result" -eq 0 ]]; then
        write_status completed "$result"
    else
        write_status failed "$result"
    fi
    exit "$result"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
write_status running 0
if [[ ! -f "$RUN_DIR/manifest.json" ]]; then
    "$PYTHON" -u attack_scripts/prepare_mml_full_reproduction.py
fi
"$PYTHON" attack_scripts/validate_mml_full_reproduction.py --run-dir "$RUN_DIR"
"$PYTHON" -u attack_scripts/mml_full_reproduction.py --run-dir "$RUN_DIR" --workers "$MML_WORKERS"
"$PYTHON" attack_scripts/validate_mml_full_reproduction.py --run-dir "$RUN_DIR" --completed
