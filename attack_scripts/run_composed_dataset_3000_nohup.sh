#!/usr/bin/env bash
set -euo pipefail

# Start the exact 3,600-raw -> 3,000-final harmful-only composition run in a
# detached session.  The worker performs CUDA/data/config preflight, executes
# configured memory-bounded groups, validates them, and writes SUCCESS only after the
# final merge passes validation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKER="$SCRIPT_DIR/run_composed_dataset_3000.sh"
LAUNCHER="$SCRIPT_DIR/run_composed_dataset_3000_nohup.sh"
cd "$REPO_ROOT"

usage() {
  cat <<'EOF'
Usage:
  bash attack_scripts/run_composed_dataset_3000_nohup.sh
  bash attack_scripts/run_composed_dataset_3000_nohup.sh --status
  bash attack_scripts/run_composed_dataset_3000_nohup.sh --dry-run

Optional environment variables:
  OUTPUT_ROOT  Dataset/output directory (default: runs/composed_jailbreak_3000_v3)
  CONFIG       Composition YAML (default: configs/compositions/jailbreak_dataset_v3.yaml)
  PYTHON       Python executable (worker defaults to the jb_env interpreter)
  LOG_FILE     Combined nohup log (default: <OUTPUT_ROOT>/nohup.log)
  PID_FILE     Detached worker PID (default: <OUTPUT_ROOT>/nohup.pid)
  VLM_JAILBREAK_DATA_ROOT, CUDA_VISIBLE_DEVICES, HF_HOME, etc. are inherited.

The launcher survives an SSH disconnect, but not a machine reboot. Completed
recipe groups are reused. Finalized incomplete groups are integrity-checked,
backed up, and resumed; manifest-less or corrupt partial groups are untouched.
EOF
}

resolve_path() {
  if [[ "$1" = /* ]]; then
    realpath -m -- "$1"
  else
    realpath -m -- "$REPO_ROOT/$1"
  fi
}

MODE="${1:-start}"
if [[ $# -gt 1 ]]; then
  usage >&2
  exit 2
fi
case "$MODE" in
  start) ;;
  --status) ;;
  --dry-run) ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown option: $MODE" >&2
    usage >&2
    exit 2
    ;;
esac

RUN_ROOT="$(resolve_path "${OUTPUT_ROOT:-runs/composed_jailbreak_3000_v3}")"
CONFIG_PATH="$(resolve_path "${CONFIG:-configs/compositions/jailbreak_dataset_v3.yaml}")"
LOG_PATH="$(resolve_path "${LOG_FILE:-$RUN_ROOT/nohup.log}")"
PID_PATH="$(resolve_path "${PID_FILE:-$RUN_ROOT/nohup.pid}")"

if [[ "$RUN_ROOT" == "/" || "$RUN_ROOT" == "$REPO_ROOT" ]]; then
  echo "Refusing unsafe OUTPUT_ROOT: $RUN_ROOT" >&2
  exit 2
fi

read_pid() {
  local value=""
  if [[ -f "$PID_PATH" ]]; then
    IFS= read -r value < "$PID_PATH" || true
  fi
  printf '%s' "$value"
}

is_worker_pid() {
  local pid="$1"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  # If procfs is readable, reject a PID that has since been reused by an
  # unrelated process. If it is not readable, conservatively treat it as live.
  if [[ -r "/proc/$pid/cmdline" ]]; then
    local command_line
    command_line="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
    [[ "$command_line" == *"run_composed_dataset_3000.sh"* ]]
    return
  fi
  return 0
}

verify_completed_config() {
  local interpreter="${PYTHON:-/home/jovyan/.conda/envs/jb_env/bin/python}"
  if [[ ! -x "$interpreter" ]] && [[ -z "${PYTHON:-}" ]]; then
    interpreter="python"
  fi
  "$interpreter" -c \
    "import sys; from core.composition import require_matching_run_config; require_matching_run_config(*sys.argv[1:])" \
    "$CONFIG_PATH" "$RUN_ROOT"
}

show_status() {
  if [[ -f "$RUN_ROOT/SUCCESS" ]]; then
    verify_completed_config || return 2
    echo "COMPLETE: $RUN_ROOT"
    echo "Final dataset: $RUN_ROOT/test_cases.jsonl"
    return 0
  fi

  local pid
  pid="$(read_pid)"
  if is_worker_pid "$pid"; then
    echo "RUNNING: PID=$pid"
    ps -p "$pid" -o pid=,etime=,stat=,cmd=
    echo "Log: $LOG_PATH"
    return 0
  fi

  if [[ -n "$pid" ]]; then
    echo "NOT RUNNING: stale PID file $PID_PATH (PID=$pid)"
  else
    echo "NOT STARTED: no live PID and no SUCCESS marker under $RUN_ROOT"
  fi
  if [[ -f "$LOG_PATH" ]]; then
    echo "Last log lines:"
    tail -n 12 "$LOG_PATH"
  fi
  return 1
}

if [[ "$MODE" == "--status" ]]; then
  show_status
  exit $?
fi

if [[ ! -f "$WORKER" ]]; then
  echo "Worker script not found: $WORKER" >&2
  exit 2
fi
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Composition config not found: $CONFIG_PATH" >&2
  exit 2
fi

RUN_ENV=(
  "OUTPUT_ROOT=$RUN_ROOT"
  "CONFIG=$CONFIG_PATH"
  "PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}"
)
if [[ -n "${PYTHON:-}" ]]; then
  RUN_ENV+=("PYTHON=$PYTHON")
fi

DETACH_PREFIX=()
if command -v setsid >/dev/null 2>&1; then
  DETACH_PREFIX+=(setsid)
fi
COMMAND=("${DETACH_PREFIX[@]}" env "${RUN_ENV[@]}" bash "$WORKER")

print_command() {
  printf 'nohup'
  printf ' %q' "${COMMAND[@]}"
  printf ' >> %q 2>&1 < /dev/null &\n' "$LOG_PATH"
}

if [[ "$MODE" == "--dry-run" ]]; then
  echo "No process was started. Resolved launch command:"
  print_command
  echo "PID file: $PID_PATH"
  exit 0
fi

if [[ -f "$RUN_ROOT/SUCCESS" ]]; then
  verify_completed_config || exit 2
  echo "Dataset is already complete: $RUN_ROOT"
  exit 0
fi

existing_pid="$(read_pid)"
if is_worker_pid "$existing_pid"; then
  echo "A 3,000-sample run is already active for this output root (PID=$existing_pid)." >&2
  echo "Check it with: bash $LAUNCHER --status" >&2
  exit 4
fi

mkdir -p "$RUN_ROOT" "$(dirname "$LOG_PATH")" "$(dirname "$PID_PATH")"
timestamp="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
{
  echo
  echo "[$timestamp] Launching composed 3,000-sample run"
  print_command
} >> "$LOG_PATH"

nohup "${COMMAND[@]}" >> "$LOG_PATH" 2>&1 < /dev/null &
pid=$!
pid_tmp="$PID_PATH.tmp.$$"
printf '%s\n' "$pid" > "$pid_tmp"
mv -f -- "$pid_tmp" "$PID_PATH"

if ! kill -0 "$pid" 2>/dev/null; then
  set +e
  wait "$pid"
  status=$?
  set -e
  if [[ "$status" -eq 0 ]]; then
    echo "Worker completed before the launcher returned. See $LOG_PATH"
    exit 0
  fi
  echo "Worker exited immediately with status $status. See $LOG_PATH" >&2
  exit "$status"
fi

echo "Started the 3,000-sample run."
echo "PID: $pid"
echo "Output: $RUN_ROOT"
echo "Log: $LOG_PATH"
echo "Status: bash $LAUNCHER --status"
echo "Follow log: tail -f $LOG_PATH"
echo "Completion marker: $RUN_ROOT/SUCCESS"
