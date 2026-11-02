#!/usr/bin/env bash
set -Eeuo pipefail
run_dir=/home/jovyan/jailbreak/vlm-jailbreak-simulator/runs/jps_train_20261024
run_python=/home/jovyan/.conda/envs/jb_env/bin/python
cd /home/jovyan/jailbreak/vlm-jailbreak-simulator
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=/home/jovyan/jailbreak/runtime/jps_448_deps
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec 9> "$run_dir/training.lock"
flock -n 9 || exit 73
exec >> "$run_dir/background.log" 2>&1
trap 'run_exit=$?; printf "%s\n" "$run_exit" > "$run_dir/exit.code"; date -u +%FT%TZ > "$run_dir/finished_at.txt"; exit "$run_exit"' EXIT
date -u +%FT%TZ > "$run_dir/started_at.txt"
"$run_python" -B - <<'PY'
from pathlib import Path
import ast, hashlib, json
p=Path('runs/jps_train_20261024').resolve()
baseline=Path('runs/jps_paper_20261024').resolve()
sha=lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
for name, value in json.loads((baseline/'output_manifest.json').read_text()).items():
    assert sha(baseline/name)==value, name
memory=json.loads((p/'memory_probe.json').read_text())
assert memory['gradient_equal'] and memory['loss_equal'] and memory['gradient_finite']
smoke=json.loads((p/'smoke/round_0_optimization.json').read_text())
assert smoke['gradient_calls']==26 and smoke['gradient_norm_max']>0
agent=json.loads((p/'smoke/steering/round_1.json').read_text())
assert len(agent['components'])==2 and all(x.strip() for x in agent['components'])
assert len(agent['calls'])==6
second=json.loads((p/'smoke/round_1_optimization.json').read_text())
assert second['gradient_calls'] in (13,26)
assert second['gradient_norm_max']>0 or second['best_mean_target_loss']<=0.01
ast.parse((p/'train_run.py').read_text())
files=[p/x for x in ['train_run.py','run_background.sh','plan.json','resolved_config.json',
                     'input_manifest.json','official_source_manifest.json','inputs/provenance.json',
                     'memory_probe.json','smoke/round_0_optimization.json','smoke/steering/round_1.json',
                     'smoke/round_1_optimization.json']]
seal={str(f):sha(f) for f in files}
dest=p/'launch_manifest.json'
if dest.exists():
    assert json.loads(dest.read_text())==seal
else:
    dest.write_text(json.dumps(seal,indent=2)+'\n')
print('Gradient equivalence, native optimization and both native agent paths validated',flush=True)
PY
run_stage() {
    "$run_python" -B - "$run_dir/stage.json" "$@" <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import json, sys
p=Path(sys.argv[1])
obj={'stage':sys.argv[2], 'arguments':sys.argv[3:], 'started_at':datetime.now(timezone.utc).isoformat()}
t=p.with_suffix('.tmp'); t.write_text(json.dumps(obj,indent=2)+'\n'); t.replace(p)
print(json.dumps(obj),flush=True)
PY
    "$run_python" -B -u "$run_dir/train_run.py" "$@"
}
run_stage optimize --round 0
for round_id in 1 2 3 4 5; do
    run_stage responses --round "$((round_id - 1))"
    run_stage revise --round "$round_id"
    run_stage optimize --round "$round_id"
done
run_stage eval_prepare
run_stage eval_infer
run_stage eval_judge
run_stage eval_summarize
