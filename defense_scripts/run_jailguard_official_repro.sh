#!/usr/bin/env bash
# Launch the JailGuard official-dataset reproduction as tmux shards that share
# one GPU (each shard loads its own MiniGPT-4 copy, ~29 GB fp16).
#
#   SHARDS=3 MUTATOR=PL bash defense_scripts/run_jailguard_official_repro.sh
#   EXTRA="--limit 5" SHARDS=1 bash defense_scripts/run_jailguard_official_repro.sh   # smoke test
#
# Outputs: $RUN_DIR/<mutator>_shard<k>of<n>.jsonl (resumable) and logs under $RUN_DIR/logs.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-$HOME/.conda/envs/jailguard_mgpt4_venv/bin/python}
TMUX_BIN=${TMUX_BIN:-$HOME/.local/bin/tmux}
SHARDS=${SHARDS:-3}
MUTATOR=${MUTATOR:-PL}
RUN_DIR=${RUN_DIR:-runs/jailguard_official_repro}
EXTRA=${EXTRA:-}
mutator_lc=$(echo "$MUTATOR" | tr '[:upper:]' '[:lower:]')

mkdir -p "$RUN_DIR/logs"
for ((k = 0; k < SHARDS; k++)); do
  name="jg-${mutator_lc}-${k}"
  out="$RUN_DIR/${mutator_lc}_shard${k}of${SHARDS}.jsonl"
  log="$RUN_DIR/logs/${mutator_lc}_shard${k}of${SHARDS}.log"
  if "$TMUX_BIN" has-session -t "$name" 2>/dev/null; then
    echo "tmux session $name already exists; skipping" >&2
    continue
  fi
  "$TMUX_BIN" new -d -s "$name" \
    "cd '$(pwd)' && PYTHONNOUSERSITE=1 '$PYTHON' defense_scripts/jailguard_official_repro.py \
       --out '$out' --shard $k/$SHARDS --mutator $MUTATOR $EXTRA 2>&1 | tee -a '$log'; \
     echo EXIT_CODE=\${PIPESTATUS[0]} >> '$log'"
  echo "started tmux session $name -> $out (log: $log)"
done
