#!/usr/bin/env bash
# Independently supervised full five-category CS-DJ reproduction.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_DIR=runs/cs_dj_reproduction_20261022
PYTHON=/home/jovyan/.conda/envs/jb_env/bin/python
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1
mkdir -p "$RUN_DIR"
exec 9>"$RUN_DIR/background.lock"
flock -n 9 || exit 75
exec >>"$RUN_DIR/background.log" 2>&1
write_status() {
    "$PYTHON" - "$RUN_DIR" "$1" "$2" "$$" <<'PY'
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,sys
p=Path(sys.argv[1])/'background_status.json'
d=json.loads(p.read_text()) if p.exists() else {}
now=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')
if sys.argv[2]=='running':
    d={'started_at':now,'tmux_socket':'cs-dj-repro-20261022','tmux_session':'csdj'}
d.update(status=sys.argv[2],exit_code=int(sys.argv[3]),worker_pid=int(sys.argv[4]),updated_at=now)
t=p.with_suffix('.tmp')
t.write_text(json.dumps(d,indent=2)+'\n')
t.replace(p)
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
for phase in prepare preflight retrieve build generate judge summarize; do
    "$PYTHON" -u attack_scripts/cs_dj_reproduction.py "$phase"
done
