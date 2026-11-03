#!/usr/bin/env bash
set -euo pipefail
RUN=/home/jovyan/jailbreak/vlm-jailbreak-simulator/runs/cs_dj_reference_onevision_20261022
ROOT=/home/jovyan/jailbreak/vlm-jailbreak-simulator
PYTHON=/home/jovyan/.conda/envs/jb_env/bin/python
NATIVE=/home/jovyan/jailbreak/runtime/llava_next_deps:/home/jovyan/jailbreak/reference_repos/LLaVA-NeXT
export PYTHONNOUSERSITE=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec 9>"$RUN/background.lock"
flock -n 9 || exit 75
exec >>"$RUN/background.log" 2>&1
phase=starting
write_status() {
    "$PYTHON" - "$RUN" "$1" "$2" "$$" "$phase" <<'PY'
import json,sys,time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
p=Path(sys.argv[1])/'background_status.json'
v={'status':sys.argv[2],'exit_code':int(sys.argv[3]),'worker_pid':int(sys.argv[4]),
   'phase':sys.argv[5],'updated_at':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
   'tmux_socket':'csdj-reference-20261022','tmux_session':'csdj-reference'}
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
cd "$ROOT"
stage cs_dj_judge_preflight
"$PYTHON" -u "$RUN/evaluate_cs_dj.py" preflight
stage validating_native_checkpoint
if "$PYTHON" - "$RUN" <<'PY'
import hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1])
try:
    v=json.loads((r/'math_sdpa_fix_validation.json').read_text())
    assert v['passed'] and v['smoke_cases']==7 and v['after_The_only']==0
    assert hashlib.sha256((r/'native_onevision.py').read_bytes()).hexdigest()==v['runner_sha256']
    assert hashlib.sha256((r/'native_smoke.json').read_bytes()).hexdigest()==v['smoke_sha256']
    assert json.loads((r/'native_smoke.json').read_text())['passed']
    for name in ['original_model_validation.json','vision_model_validation.json']:
        assert json.loads((r/name).read_text())['all_weights_verified']
except (AssertionError,KeyError,ValueError,OSError):
    raise SystemExit(1)
print('REUSE_VERIFIED_NATIVE_SMOKE',flush=True)
PY
then
    :
else
    stage downloading_original_weights
    "$PYTHON" -u "$RUN/orchestrate.py" download
    stage native_onevision_validation
    HF_HUB_OFFLINE=1 PYTHONPATH="$NATIVE:$RUN" "$PYTHON" -u "$RUN/native_onevision.py" smoke
fi
export HF_HUB_OFFLINE=1
for part in embedding retrieval samples; do
    stage "official_cs_dj_$part"
    PYTHONPATH="/tmp/csdj_pillow95:$RUN" "$PYTHON" -u "$RUN/orchestrate.py" "$part"
done
stage cs_dj_target_generation
PYTHONPATH="$NATIVE:$RUN" "$PYTHON" -u "$RUN/native_onevision.py" generate "$RUN/test_cases.jsonl" "$RUN/cs_dj_responses.jsonl"
stage cs_dj_beaver_dam_evaluation
"$PYTHON" -u "$RUN/evaluate_cs_dj.py" evaluate
stage hades_official_data_preparation
unset HF_HUB_OFFLINE
# Existing preparation validates its historical dependency; also require the
# current native OneVision run's complete, sealed evaluation before HADES.
"$PYTHON" - "$RUN" <<'PY'
import hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1]);v=json.loads((r/'evaluation_validation.json').read_text())
assert v['passed'] and v['n']==750
for name,digest in v['sha256'].items():
    assert hashlib.sha256((r/name).read_bytes()).hexdigest()==digest,name
PY
PYTHONPATH="/tmp/hades_preparation_deps:$ROOT:$RUN" "$PYTHON" -u "$RUN/prepare_hades_data.py"
stage hades_paper_llava15_reproduction
bash "$ROOT/runs/hades_paper_llava15_20261022/background_worker.sh"
stage complete
