#!/usr/bin/env python3
"""Persistent supervisor for the single-benchmark VisualAdv comparison.

Launch with stdin=/dev/null, stdout/stderr=logfile and start_new_session=True.
The supervisor and its workers ignore SIGHUP and hold no VS Code terminal FD.
"""

from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from attack_scripts.visualadv_protocol import write_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, default=ROOT / "runs/visualadv_background")
    parser.add_argument("--released-out", type=Path, default=ROOT / "runs/visualadv_rtp_released")
    parser.add_argument("--port-out", type=Path, default=ROOT / "runs/visualadv_rtp_port")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/VisualAdv_paper.yaml")
    args = parser.parse_args()
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    args.job_dir.mkdir(parents=True, exist_ok=True)
    with (args.job_dir / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (args.job_dir / "supervisor.pid").write_text(str(os.getpid()) + "\n")

        def update(phase, **extra):
            stamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(timespec="seconds")
            value = {"phase": phase, "updated_kst": stamp, "supervisor_pid": os.getpid(),
                     "process_group": os.getpgrp(), "released_out": str(args.released_out),
                     "port_out": str(args.port_out), **extra}
            write_json(args.job_dir / "status.json", value)
            print(f"[{stamp}] {phase} {extra}", flush=True)

        def run(stage, out, image_source):
            phase = f"{image_source}:{stage}"
            command = [sys.executable, "-u", str(ROOT / "attack_scripts/visualadv_reproduction.py"),
                       "--stage", stage, "--out", str(out), "--config", str(args.config),
                       "--image-source", image_source]
            env = {**os.environ, "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "PYTHONNOUSERSITE": "1"}
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL)
            update(phase, running=True, child_pid=child.pid)
            result = child.wait()
            if result:
                update("failed", running=False, failed_phase=phase, returncode=result)
                raise SystemExit(result)

        update("starting", running=True)
        for stage in ("generate", "score", "report"):
            run(stage, args.released_out, "released")
        released = json.loads((args.released_out / "report.json").read_text())
        if not released["protocol_complete"]:
            update("failed", running=False, failed_phase="released:coverage", returncode=2)
            return 2
        # The worker validates the copied baseline's complete manifest and hashes
        # before reusing it. No model generation is duplicated between comparisons.
        source = args.released_out / "cells/rtp/clean/clean/minigpt4"
        target = args.port_out / "cells/rtp/clean/clean/minigpt4"
        if not target.exists():
            shutil.copytree(source, target)
        for stage in ("plan", "audit", "optimize", "generate", "score", "report"):
            run(stage, args.port_out, "trained")
        report = json.loads((args.port_out / "report.json").read_text())
        update("complete" if report["protocol_complete"] else "incomplete",
               running=False, protocol_complete=report["protocol_complete"],
               report=str(args.port_out / "report.md"))
        return 0 if report["protocol_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
