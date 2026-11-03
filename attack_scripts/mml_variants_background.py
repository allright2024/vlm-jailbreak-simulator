#!/usr/bin/env python3
"""Detached supervisor for the prepared four-variant MML evaluation."""
import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/mml_variants_reproduction_20261102")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    def record(state, **extra):
        temp = run / "background_status.json.tmp"
        temp.write_text(json.dumps({"status": state, "pid": os.getpid(), "session_id": os.getsid(0), "updated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra}, indent=2) + "\n")
        temp.replace(run / "background_status.json")
    record("starting")
    with (run / "evaluation.log").open("ab") as log:
        child = subprocess.Popen([sys.executable, "-u", "-B", str(ROOT / "attack_scripts/mml_variants_reproduction.py"), "--run-dir", str(run), "--workers", "8"], cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        record("running", child_pid=child.pid)
        code = child.wait()
    record("completed" if code == 0 else "failed", exit_code=code)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
