#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_DIR=runs/hades_reproduction_preparation_20261022
PYTHON=/home/jovyan/.conda/envs/jb_env/bin/python
export PYTHONNOUSERSITE=1
export PYTHONPATH="/tmp/hades_preparation_deps:$PWD"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1
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
d.update(status=sys.argv[2],exit_code=int(sys.argv[3]),worker_pid=int(sys.argv[4]),updated_at=now,
         tmux_socket='hades-prep-20261022',tmux_session='hades-prep')
if sys.argv[2]=='running': d['started_at']=now
t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
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
"$PYTHON" -u attack_scripts/hades_reproduction_preparation.py wait
# Install the parquet reader in isolation, after the dependency has finished.
if ! "$PYTHON" -c 'import pyarrow.parquet' >/dev/null 2>&1; then
    "$PYTHON" -m pip install --disable-pip-version-check --no-deps --target /tmp/hades_preparation_deps pyarrow==20.0.0
fi
"$PYTHON" -u attack_scripts/hades_reproduction_preparation.py prepare
export HF_HUB_OFFLINE=1
for phase in smoke_generate smoke_judge finish; do
    "$PYTHON" -u attack_scripts/hades_reproduction_preparation.py "$phase"
done
