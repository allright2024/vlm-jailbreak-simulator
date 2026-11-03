#!/usr/bin/env bash
# Detached worker. Invoke through run_five_method_dataset_background.sh.
set -euo pipefail
trap '' HUP
RUN_ROOT="$1"
PYTHON_BIN="$2"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export TZ=Asia/Seoul
export PYTHON="$PYTHON_BIN"
mkdir -p "$RUN_ROOT/logs"
printf '%s\n' "$$" > "$RUN_ROOT/worker.pid.tmp"
mv "$RUN_ROOT/worker.pid.tmp" "$RUN_ROOT/worker.pid"
date -Iseconds > "$RUN_ROOT/worker.started_at"
printf 'running\n' > "$RUN_ROOT/worker.status"
rm -f "$RUN_ROOT/worker.exit_code" "$RUN_ROOT/worker.finished_at"
finish() {
  local code=$?
  trap - EXIT
  printf '%s\n' "$code" > "$RUN_ROOT/worker.exit_code"
  date -Iseconds > "$RUN_ROOT/worker.finished_at"
  if [[ "$code" == 0 ]]; then
    printf 'complete\n' > "$RUN_ROOT/worker.status"
  elif [[ "$code" == 130 || "$code" == 143 ]]; then
    printf 'stopped\n' > "$RUN_ROOT/worker.status"
  else
    printf 'failed\n' > "$RUN_ROOT/worker.status"
  fi
  exit "$code"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
echo "Started: $(date -Iseconds); PID=$$; SID=$(ps -o sid= -p $$ | tr -d ' ')"
bash attack_scripts/run_five_method_dataset.sh full --output "$RUN_ROOT"
PYTHONNOUSERSITE=1 "$PYTHON_BIN" attack_scripts/five_method_gallery.py "$RUN_ROOT"
echo "Completed: $(date -Iseconds); dataset=$RUN_ROOT/full/test_cases.jsonl"
