#!/usr/bin/env bash
# SSH/VS Code independent launcher with verified PID identity and resumable inputs.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKER="$REPO_ROOT/attack_scripts/five_method_dataset_worker.sh"
RUN_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/runs/five_methods_3000_jood_image_mix_20261103}"
RUN_ROOT="$(realpath -m -- "$RUN_ROOT")"
PYTHON_BIN="${PYTHON:-/home/jovyan/.conda/envs/jb_env/bin/python}"
ACTION="${1:-start}"
if [[ "$ACTION" == --help || "$ACTION" == -h ]]; then
  echo "Usage: bash attack_scripts/run_five_method_dataset_background.sh {start|status|stop}"
  echo "Optional: OUTPUT_ROOT=/absolute/output/path PYTHON=/path/to/python"
  exit 0
fi
mkdir -p "$RUN_ROOT/logs"
LOG_FILE="$RUN_ROOT/logs/worker.log"
PID_FILE="$RUN_ROOT/worker.pid"
pid_matches() {
  [[ -f "$PID_FILE" ]] || return 1
  IFS= read -r WORKER_PID < "$PID_FILE"
  [[ "$WORKER_PID" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$WORKER_PID" 2>/dev/null || return 1
  [[ -r "/proc/$WORKER_PID/cmdline" ]] || return 1
  local -a argv=()
  mapfile -d '' -t argv < "/proc/$WORKER_PID/cmdline"
  [[ "${argv[1]:-}" == "$WORKER" && "${argv[2]:-}" == "$RUN_ROOT" ]]
}
case "$ACTION" in
  start)
    exec 8>"$RUN_ROOT/.background_launch.lock"
    flock -n 8 || { echo "Another launcher is active." >&2; exit 2; }
    if pid_matches; then
      echo "Already running: PID=$WORKER_PID; log=$LOG_FILE"
      exit 0
    fi
    [[ -x "$PYTHON_BIN" ]] || { echo "Python not executable: $PYTHON_BIN" >&2; exit 2; }
    [[ -f "$RUN_ROOT/preview/visual_review.json" ]] || {
      echo "A validated, visually reviewed preview is required before background production." >&2
      exit 2
    }
    exec 9>"$RUN_ROOT/.generation.lock"
    flock -n 9 || { echo "A generator already holds this output lock; no duplicate was started." >&2; exit 2; }
    flock -u 9
    exec 9>&-
    command -v setsid >/dev/null
    command -v nohup >/dev/null
    nohup setsid bash "$WORKER" "$RUN_ROOT" "$PYTHON_BIN" </dev/null >>"$LOG_FILE" 2>&1 8>&- &
    candidate=$!
    for _ in 1 2 3 4 5; do
      if pid_matches; then
        echo "Started detached: PID=$WORKER_PID; log=$LOG_FILE"
        exit 0
      fi
      if ! kill -0 "$candidate" 2>/dev/null; then break; fi
      sleep 1
    done
    echo "Worker did not stay running. See $LOG_FILE" >&2
    exit 3
    ;;
  status)
    if pid_matches; then
      echo "RUNNING PID=$WORKER_PID"
      ps -o pid,ppid,sid,tty,etime,args -p "$WORKER_PID"
    elif [[ -f "$RUN_ROOT/worker.status" ]]; then
      cat "$RUN_ROOT/worker.status"
    else
      echo "NOT STARTED"
    fi
    if [[ -f "$RUN_ROOT/worker.exit_code" ]]; then
      echo "Exit code: $(cat "$RUN_ROOT/worker.exit_code")"
    fi
    if [[ -d "$RUN_ROOT/full/cases" ]]; then
      count=$(find "$RUN_ROOT/full/cases" -type f -name '*.json' | wc -l)
      echo "Checkpointed inputs: $count / 3000"
    fi
    echo "Log: $LOG_FILE"
    if [[ -f "$LOG_FILE" ]]; then tail -n 8 "$LOG_FILE"; fi
    ;;
  stop)
    if ! pid_matches; then echo "No matching live worker."; exit 0; fi
    pgid=$(ps -o pgid= -p "$WORKER_PID" | tr -d ' ')
    [[ "$pgid" == "$WORKER_PID" ]] || { echo "Unexpected process group; refusing group signal." >&2; exit 3; }
    kill -TERM -- "-$WORKER_PID"
    echo "Stop requested for worker group $WORKER_PID. Completed inputs are kept; start resumes them."
    ;;
  *) echo "Unknown action: $ACTION" >&2; exit 2 ;;
esac
