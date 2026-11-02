#!/usr/bin/env bash
# Preserve the JOOD reproduction through editor disconnection.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_DIR=runs/jood_reproduction_20261022
PYTHON=/home/jovyan/.conda/envs/jb_env/bin/python
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1
mkdir -p "$RUN_DIR"
exec 9>"$RUN_DIR/background.lock"
flock -n 9 || { echo "JOOD reproduction worker already running."; exit 75; }
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
value = json.loads(path.read_text()) if path.exists() else {}
now = datetime.datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')
if sys.argv[2] == 'running':
    value = {'started_at': now, 'ppl_excluded': True,
             'tmux_socket': 'jood-repro-20261022', 'tmux_session': 'jood'}
value.update(status=sys.argv[2], updated_at=now, worker_pid=int(sys.argv[4]))
if sys.argv[2] != 'running':
    value['exit_code'] = int(sys.argv[3])
temp = root / 'background_status.json.tmp'
temp.write_text(json.dumps(value, indent=2) + '\n')
os.replace(temp, path)
PY
}

finish() {
    result=$?
    trap - EXIT
    if [[ "$result" -eq 0 ]]; then write_status completed "$result";
    else write_status failed "$result"; fi
    exit "$result"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
write_status running 0
if [[ ! -f "$RUN_DIR/manifest.json" ]]; then
    "$PYTHON" -u attack_scripts/prepare_jood_reproduction.py
fi
"$PYTHON" -u attack_scripts/jood_reproduction.py generate --workers 8
"$PYTHON" -u attack_scripts/download_jood_guard.py
"$PYTHON" -u attack_scripts/jood_reproduction.py guard --guard-batch-size 1
"$PYTHON" -u attack_scripts/jood_reproduction.py summarize
