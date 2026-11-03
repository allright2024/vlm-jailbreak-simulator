#!/usr/bin/env bash
set -euo pipefail
RUN=/home/jovyan/jailbreak/vlm-jailbreak-simulator/runs/hades_paper_llava15_20261022
PYTHON=/home/jovyan/.conda/envs/jb_env/bin/python
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec 8>"$RUN/background.lock"
flock -n 8 || exit 75
exec >>"$RUN/background.log" 2>&1
phase=starting
write_status() {
    "$PYTHON" - "$RUN" "$1" "$2" "$$" "$phase" <<'PY'
import json,sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
p=Path(sys.argv[1])/'background_status.json'
v={'status':sys.argv[2],'exit_code':int(sys.argv[3]),'worker_pid':int(sys.argv[4]),
   'phase':sys.argv[5],'updated_at':datetime.now(ZoneInfo('Asia/Seoul')).isoformat()}
t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
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
stage() { phase="$1"; write_status running 0; }
stage validating_official_data
"$PYTHON" -u "$RUN/paper_run.py" prepare
stage judge_preflight
PYTHONPATH=/home/jovyan/jailbreak/reference_repos/HADES "$PYTHON" -u "$RUN/paper_run.py" judge_preflight
for part in smoke generate; do
    stage "$part"
    PYTHONPATH=/home/jovyan/jailbreak/runtime/hades_llava15_deps:/home/jovyan/jailbreak/runtime/llava_next_deps \
      "$PYTHON" -u "$RUN/paper_run.py" "$part"
done
stage beaver_dam_evaluation
PYTHONPATH=/home/jovyan/jailbreak/reference_repos/HADES "$PYTHON" -u "$RUN/paper_run.py" judge
stage complete
