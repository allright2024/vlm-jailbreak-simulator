#!/usr/bin/env bash
# Run VisualAdv and DeltaJP in TRANSFER mode: craft on the Qwen2-VL-7B surrogate,
# evaluate on the LLaVA target. Compare with their white-box (LLaVA-surrogate)
# runs to measure surrogate->target transfer.
#
# VisualAdv: one universal image, optimized ONCE and reused on every dataset
#            (cache is accepted only if its config SHA and lineage verify).
# DeltaJP:   one universal delta added to each behavior's own image; the delta is
#            re-optimized per dataset run (it is not persisted), so it only runs
#            on datasets with per-behavior images.
#
# Usage:
#   bash attack_scripts/run_transfer.sh                       # both attacks, all compatible datasets
#   ATTACKS="VisualAdv" bash attack_scripts/run_transfer.sh   # one attack
#   VISUALADV_DS="AdvBench SafeBench" DELTAJP_DS="AdvBenchM" bash ...
#   LIMIT=20 bash attack_scripts/run_transfer.sh              # per-dataset behavior cap (smoke)
#   PYTHON=/path/to/python bash ...                           # default: python of the active env
set -uo pipefail
export PYTHONNOUSERSITE=1
PYTHON=${PYTHON:-python}
cd "$(dirname "$0")/.."

MODEL=${MODEL:-llava_v1_5}
LIMIT=${LIMIT:-0}
ATTACKS=${ATTACKS:-"VisualAdv DeltaJP"}
VISUALADV_DS=${VISUALADV_DS:-"AdvBench SafeBench AdvBenchM HADES MM-SafetyBench"}
DELTAJP_DS=${DELTAJP_DS:-"AdvBenchM HADES MM-SafetyBench"}   # needs per-behavior images
VA_CONFIG=${VA_CONFIG:-configs/attacks/VisualAdv_transfer.yaml}
DJ_CONFIG=${DJ_CONFIG:-configs/attacks/DeltaJP_transfer.yaml}
limit_flag() { [ "$LIMIT" -gt 0 ] && echo "--limit $LIMIT" || echo ""; }

"$PYTHON" -c "import transformers" 2>/dev/null \
  || { echo "!! '$PYTHON' has no transformers; activate jb_env or set PYTHON=..."; exit 1; }

run_stages() {  # attack dataset config run_dir
  local attack="$1" dataset="$2" config="$3" run_dir="$4"
  echo "=================================================================="
  echo ">>> $attack (transfer) x $dataset  (limit=$LIMIT) -> $run_dir"
  echo "=================================================================="
  "$PYTHON" run_generate_test_cases.py --attack "$attack" --dataset "$dataset" \
      --config "$config" --save_dir "$run_dir" $(limit_flag) \
      || { echo "!! stage1 failed: $attack/$dataset"; return 1; }
  "$PYTHON" run_generate_completions.py --run_dir "$run_dir" --model "$MODEL" \
      || { echo "!! stage2 failed: $attack/$dataset"; return 1; }
  "$PYTHON" run_evaluate_completions.py --run_dir "$run_dir" --model "$MODEL" \
      || { echo "!! stage3 failed: $attack/$dataset"; return 1; }
}

run_visualadv() {
  local runs=runs/VisualAdv_transfer
  local opt_dir="$runs/__optimize__"
  local img="$opt_dir/images/visual_adv.png"
  local sha_file="$opt_dir/opt_config.sha256"
  local cur_sha; cur_sha=$(sha256sum "$VA_CONFIG" | awk '{print $1}')

  echo ">>> VisualAdv (transfer): universal image -> $img"
  if [ -f "$img" ] && [ "$(cat "$sha_file" 2>/dev/null)" = "$cur_sha" ] && \
      "$PYTHON" -m core.validation.artifact_lineage --attack VisualAdv --config "$VA_CONFIG" --artifact "$img"; then
    echo ">>> verified optimization cache"
  else
    [ -f "$img" ] && echo ">>> cache config/input/code/lineage mismatch -> re-optimizing"
    rm -f "$img" "$img.lineage.json"
    "$PYTHON" run_generate_test_cases.py --attack VisualAdv --dataset AdvBench \
        --config "$VA_CONFIG" --save_dir "$opt_dir" --limit 1 \
        || { echo "!! VisualAdv optimization failed"; return 1; }
    "$PYTHON" -m core.validation.artifact_lineage --attack VisualAdv --config "$VA_CONFIG" --artifact "$img" \
        || { echo "!! optimized artifact lineage verification failed"; return 1; }
    echo "$cur_sha" > "$sha_file"
  fi

  # reuse: inherit the full optimization config, only add the prebuilt image
  local reuse_cfg; reuse_cfg=$(mktemp /tmp/visualadv_transfer_reuse_XXXXXX.yaml)
  cat "$VA_CONFIG" > "$reuse_cfg"
  printf '\nadv_image: %s\n' "$img" >> "$reuse_cfg"
  for d in $VISUALADV_DS; do
    run_stages VisualAdv "$d" "$reuse_cfg" "$runs/$d"
  done
  rm -f "$reuse_cfg"
}

run_deltajp() {
  for d in $DELTAJP_DS; do
    run_stages DeltaJP "$d" "$DJ_CONFIG" "runs/DeltaJP_transfer/$d"
  done
}

for a in $ATTACKS; do
  case "$a" in
    VisualAdv) run_visualadv ;;
    DeltaJP)   run_deltajp ;;
    *) echo "!! unknown attack for transfer: $a (expected VisualAdv or DeltaJP)" ;;
  esac
done
echo "Done. Transfer ASR summaries appended to asr_results.jsonl"
