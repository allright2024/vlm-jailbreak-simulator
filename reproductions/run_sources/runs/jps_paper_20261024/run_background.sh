#!/usr/bin/env bash
set -Eeuo pipefail
run_dir=/home/jovyan/jailbreak/vlm-jailbreak-simulator/runs/jps_paper_20261024
run_python=/home/jovyan/.conda/envs/jb_env/bin/python
cd /home/jovyan/jailbreak/vlm-jailbreak-simulator
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=/home/jovyan/jailbreak/runtime/jps_448_deps
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec >> "$run_dir/background.log" 2>&1
trap 'run_exit=$?; printf "%s\n" "$run_exit" > "$run_dir/exit.code"; date -u +%FT%TZ > "$run_dir/finished_at.txt"; exit "$run_exit"' EXIT
date -u +%FT%TZ > "$run_dir/started_at.txt"
"$run_python" -B - <<'PY'
from pathlib import Path
import json, hashlib
p=Path('runs/jps_paper_20261024')
control=json.loads((p/'numerics_probe.json').read_text())
relevant=[x for x in control if 'ocr_expected' in x or 'control' in x]
assert len(relevant)==6 and all(x['correct'] for x in relevant)
smoke=[json.loads(x) for x in (p/'smoke_responses.jsonl').read_text().splitlines()]
assert len(smoke)==4 and all(x['answer'].strip() and x['answer'].strip()!='The' for x in smoke)
token=json.loads((p/'tokenization_check.json').read_text())
assert token['input_ids_equal'] and token['target_ids_equal']
files=[p/'paper_run.py', p/'test_cases.json', p/'plan.json', p/'run_background.sh',
       Path('multi_attacks/jps/jps.py'), Path('configs/attacks/JPS.yaml')]
seal={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
(p/'launch_manifest.json').write_text(json.dumps(seal,indent=2)+'\n')
(p/'status.json').write_text(json.dumps({'stage':'smoke_judge','total_responses':1200})+'\n')
print('Preflight controls and tokenization passed',flush=True)
PY
"$run_python" -B -u "$run_dir/paper_run.py" smoke_judge
"$run_python" -B -u "$run_dir/paper_run.py" infer
"$run_python" -B -u "$run_dir/paper_run.py" judge
"$run_python" -B -u "$run_dir/paper_run.py" summarize
