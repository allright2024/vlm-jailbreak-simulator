#!/usr/bin/env bash
set -euo pipefail

# Build the configured 3,000-sample harmful-only composed jailbreak dataset.
#
# Foreground:
#   bash attack_scripts/run_composed_dataset_3000.sh
#
# Survive SSH disconnect with PID/log management:
#   bash attack_scripts/run_composed_dataset_3000_nohup.sh
#
# Optional overrides:
#   PYTHON=/path/to/python
#   CONFIG=configs/compositions/jailbreak_dataset_v3.yaml
#   OUTPUT_ROOT=runs/composed_jailbreak_3000_v3
#   PREFLIGHT_ONLY=1  # validate prerequisites/quotas, then stop
#
# A completed recipe group is validated and reused on a later invocation. A
# finalized group whose manifest is explicitly "incomplete" is resumed only
# after the generator verifies its config, seed, QC, hashes, and lineage. Other
# invalid or manifest-less partial directories are never overwritten.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif [[ -x "/home/jovyan/.conda/envs/jb_env/bin/python" ]]; then
  PYTHON_BIN="/home/jovyan/.conda/envs/jb_env/bin/python"
else
  PYTHON_BIN="python"
fi

CONFIG="${CONFIG:-configs/compositions/jailbreak_dataset_v3.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/composed_jailbreak_3000_v3}"
OUTPUT_ROOT="$(realpath -m -- "$OUTPUT_ROOT")"
PARTS_DIR="$OUTPUT_ROOT/_parts"
LOG_DIR="$OUTPUT_ROOT/logs"

# Prevent ~/.local packages (notably the incompatible timm 0.6.13 currently
# installed on this host) from shadowing the selected Conda environment.
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NLTK_DATA="${NLTK_DATA:-$REPO_ROOT/data/nltk_data}"
export VLM_COMPOSITION_CACHE_ROOT="${VLM_COMPOSITION_CACHE_ROOT:-$OUTPUT_ROOT/_universal_cache}"

if [[ "$PYTHON_BIN" == */* ]]; then
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found or not executable: $PYTHON_BIN" >&2
    exit 2
  fi
elif ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found on PATH: $PYTHON_BIN" >&2
  exit 2
else
  PYTHON_BIN="$(command -v "$PYTHON_BIN")"
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "Composition config not found: $CONFIG" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT" "$PARTS_DIR" "$LOG_DIR" "$NLTK_DATA"

# Hold this descriptor for the lifetime of the worker. This closes the race
# between two foreground/nohup launches targeting the same output directory.
if ! command -v flock >/dev/null 2>&1; then
  echo "Required command not found: flock" >&2
  exit 2
fi
exec 9>"$OUTPUT_ROOT/.generation.lock"
if ! flock -n 9; then
  echo "Another generation process already holds $OUTPUT_ROOT/.generation.lock" >&2
  echo "Use the nohup launcher's --status command or choose another OUTPUT_ROOT." >&2
  exit 4
fi

# An already merged and independently valid dataset is a successful no-op.
if [[ -f "$OUTPUT_ROOT/composition_manifest.json" ]]; then
  "$PYTHON_BIN" -c \
    "import sys; from core.composition import require_matching_run_config; require_matching_run_config(*sys.argv[1:])" \
    "$CONFIG" "$OUTPUT_ROOT"
  "$PYTHON_BIN" -m core.validation.composed_dataset "$OUTPUT_ROOT"
  "$PYTHON_BIN" -c \
    "import json,sys; m=json.load(open(sys.argv[1])); assert m['n_final_test_cases']==3000 and m['n_raw_structurally_valid']==3600" \
    "$OUTPUT_ROOT/composition_manifest.json"
  echo "Dataset is already complete: $OUTPUT_ROOT"
  exit 0
fi

"$PYTHON_BIN" -c \
  "import torch; assert torch.cuda.is_available(), 'CUDA GPU is required for the production composition run'; print('CUDA:', torch.cuda.get_device_name(0))"

# MML word replacement uses the tagger name expected by requirements.txt's
# pinned NLTK 3.8.1. Download it only when it is absent.
if ! "$PYTHON_BIN" -c \
  "import nltk,sys; nltk.data.path.insert(0, sys.argv[1]); nltk.pos_tag(['smoke'])" \
  "$NLTK_DATA" \
  >/dev/null 2>&1; then
  "$PYTHON_BIN" -m nltk.downloader -d "$NLTK_DATA" averaged_perceptron_tagger
fi

echo "Preflight: production plan, harmful-only capacity, and stage configs"
"$PYTHON_BIN" run_generate_composed_dataset.py \
  --config "$CONFIG" \
  --preflight

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "PREFLIGHT_ONLY=1: checks passed; generation was not started."
  exit 0
fi

declare -a COMPLETED_PARTS=()

run_group() {
  local group_name="$1"
  local recipe_ids="$2"
  local group_dir="$PARTS_DIR/$group_name"
  local group_log="$LOG_DIR/$group_name.log"

  if [[ -f "$group_dir/composition_manifest.json" ]]; then
    "$PYTHON_BIN" -c \
      "import sys; from core.composition import require_matching_run_config; require_matching_run_config(*sys.argv[1:])" \
      "$CONFIG" "$group_dir" "$recipe_ids"
    if "$PYTHON_BIN" -m core.validation.composed_dataset --quiet "$group_dir"; then
      echo "Reuse complete group: $group_name ($recipe_ids)"
      COMPLETED_PARTS+=("$group_dir")
      return
    fi
    local manifest_status
    if ! manifest_status="$("$PYTHON_BIN" -c \
      "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8')).get('status', 'missing'))" \
      "$group_dir/composition_manifest.json")"; then
      echo "Existing group manifest is unreadable: $group_dir" >&2
      exit 3
    fi
    if [[ "$manifest_status" != "incomplete" ]]; then
      echo "Existing group is invalid and is not resume-eligible: $group_dir" >&2
      exit 3
    fi
    echo "Resume finalized incomplete group: $group_name ($recipe_ids)"
    "$PYTHON_BIN" run_generate_composed_dataset.py \
      --config "$CONFIG" \
      --recipes "$recipe_ids" \
      --save-dir "$group_dir" \
      --resume \
      --verbose 2>&1 | tee -a "$group_log"
    "$PYTHON_BIN" -m core.validation.composed_dataset "$group_dir"
    COMPLETED_PARTS+=("$group_dir")
    return
  fi
  if [[ -d "$group_dir" ]]; then
    echo "Partial group directory has no manifest: $group_dir" >&2
    echo "Move it aside or choose a new OUTPUT_ROOT; nothing was overwritten." >&2
    exit 3
  fi

  echo "Run group: $group_name ($recipe_ids)"
  "$PYTHON_BIN" run_generate_composed_dataset.py \
    --config "$CONFIG" \
    --recipes "$recipe_ids" \
    --save-dir "$group_dir" \
    --verbose 2>&1 | tee -a "$group_log"
  "$PYTHON_BIN" -m core.validation.composed_dataset "$group_dir"
  COMPLETED_PARTS+=("$group_dir")
}

# Separate processes bound peak memory. Within a group, universal artifacts
# and identical component instances are reused rather than re-optimized.
GROUP_PLAN="$("$PYTHON_BIN" - "$CONFIG" <<'PY'
import sys
from core.composition import load_composition_config
config = load_composition_config(sys.argv[1])
for group in config["execution_groups"]:
    print(group["name"] + "\t" + ",".join(group["recipes"]))
PY
)"
while IFS=$'\t' read -r group_name recipe_ids; do
  run_group "$group_name" "$recipe_ids"
done <<< "$GROUP_PLAN"

echo "Merge ${#COMPLETED_PARTS[@]} validated groups into the final 3,000-sample bundle"
"$PYTHON_BIN" merge_composed_runs.py \
  --config "$CONFIG" \
  --output-dir "$OUTPUT_ROOT" \
  "${COMPLETED_PARTS[@]}"

"$PYTHON_BIN" -m core.validation.composed_dataset "$OUTPUT_ROOT"
"$PYTHON_BIN" -c \
  "import json,sys; m=json.load(open(sys.argv[1])); assert m['status']=='complete'; assert len(m['recipes'])==18; assert m['n_final_test_cases']==3000; assert m['n_raw_structurally_valid']==3600; print('FINAL OK: 18 recipes, 3,600 raw, 3,000 selected')" \
  "$OUTPUT_ROOT/composition_manifest.json"
touch "$OUTPUT_ROOT/SUCCESS"

echo "Completed dataset: $OUTPUT_ROOT"
echo "Final cases: $OUTPUT_ROOT/test_cases.jsonl"
echo "Manifest: $OUTPUT_ROOT/composition_manifest.json"
