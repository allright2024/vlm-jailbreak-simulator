#!/usr/bin/env bash
set -Eeuo pipefail
run_dir=/home/jovyan/jailbreak/vlm-jailbreak-simulator/runs/ultrabreak_released_20261024
run_python=/home/jovyan/.conda/envs/jb_env/bin/python
cd /home/jovyan/jailbreak/vlm-jailbreak-simulator
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=/home/jovyan/jailbreak/runtime/ultrabreak_451_deps
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec 9> "$run_dir/worker.lock"
flock -n 9 || exit 73
exec >> "$run_dir/background.log" 2>&1
trap 'run_exit=$?; printf "%s\n" "$run_exit" > "$run_dir/exit.code"; date -u +%FT%TZ > "$run_dir/finished_at.txt"; exit "$run_exit"' EXIT
date -u +%FT%TZ > "$run_dir/started_at.txt"
run_stage() {
    "$run_python" -B - "$run_dir/stage.json" "$@" <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import json,sys
p=Path(sys.argv[1]); obj={'stage':sys.argv[2], 'arguments':sys.argv[3:], 'started_at':datetime.now(timezone.utc).isoformat()}
t=p.with_suffix('.tmp'); t.write_text(json.dumps(obj,indent=2)+'\n'); t.replace(p)
print(json.dumps(obj),flush=True)
PY
    "$run_python" -B -u "$run_dir/paper_run.py" "$@"
}
run_stage infer --model qwen2 --smoke
run_stage infer --model qwen25 --smoke
run_stage judge --smoke
"$run_python" -B - <<'PY'
from pathlib import Path
import ast,json,hashlib
p=Path('runs/ultrabreak_released_20261024').resolve()
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
cases={x['case_id']:x for x in json.loads((p/'smoke_cases.json').read_text())}
for key in ['qwen2','qwen25']:
    rows=[json.loads(x) for x in (p/'smoke'/f'responses_{key}.jsonl').read_text().splitlines()]
    assert len(rows)==6 and len({x['case_id'] for x in rows})==6
    controls=[x for x in rows if x['dataset']=='control']; assert len(controls)==2
    assert all(cases[x['case_id']]['expected'] in x['assistant_response'] for x in controls)
    complete=json.loads((p/'smoke'/f'inference_{key}_complete.json').read_text())
    assert complete['native_csv_decode_exact'] and complete['n']==6
    for protocol in ['official_extraction','assistant_tokens_only']:
        labels=[json.loads(x) for x in (p/'smoke'/f'judgments_{key}_{protocol}.jsonl').read_text().splitlines()]
        assert len(labels)==4 and all(x['raw_verdict'].strip().lower() in ('yes','no') for x in labels)
ast.parse((p/'paper_run.py').read_text())
files=[p/x for x in ['paper_run.py','run_background.sh','plan.json','test_cases.json','smoke_cases.json','source_manifest.json','input_manifest.json']]
files.extend((p/'smoke').glob('inference_*_complete.json'))
seal={str(f):sha(f) for f in files}
dest=p/'launch_manifest.json'
if dest.exists():assert json.loads(dest.read_text())==seal
else:dest.write_text(json.dumps(seal,indent=2)+'\n')
print('Both native inference paths, OCR controls, CSV capture and both judge protocols passed',flush=True)
PY
date -u +%FT%TZ > "$run_dir/full_started_at.txt"
run_stage infer --model qwen2
run_stage infer --model qwen25
run_stage judge
run_stage summarize
