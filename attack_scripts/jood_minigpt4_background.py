#!/usr/bin/env python3
"""Detached JOOD supervisor: target + API HF, then GPU Guard and reports."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

from jood_minigpt4_common import ROOT, RUN, status, validate, write

TARGET_PY = "/home/jovyan/.conda/envs/visualadv_venv/bin/python"
EVAL_PY = "/home/jovyan/.conda/envs/jb_env/bin/python"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--defer-hf", action="store_true", help="Generate responses and Guard labels while HF evaluator choice is pending")
    args = parser.parse_args()
    lock = (RUN / "background.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    validate()
    assert json.loads((RUN / "smoke/local_validation.json").read_text())["passed"]
    if not args.defer_hf:
        assert json.loads((RUN / "smoke/validation.json").read_text())["passed"]
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
    children, logs = {}, []
    started = time.monotonic()

    def launch(name, python, script, *args):
        stream = (RUN / (name + ".log")).open("a", buffering=1)
        logs.append(stream)
        process = subprocess.Popen([python, "-u", str(ROOT / "attack_scripts" / script), *args],
            cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
        children[name] = process
        return process

    def report(state, stage, **extra):
        detail = {}
        for name in ["target", "hf", "guard"]:
            path = RUN / (name + "_status.json")
            if path.exists():
                detail[name] = json.loads(path.read_text())
        status("background", status=state, current_stage=stage, pid=os.getpid(),
            session_id=os.getsid(0), elapsed_seconds=round(time.monotonic() - started, 1),
            children={n: {"pid": p.pid, "exit_code": p.poll()} for n, p in children.items()},
            progress=detail, **extra)

    def interrupt(signum, frame):
        raise InterruptedError(f"Supervisor received signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        (RUN / "target_failed.json").unlink(missing_ok=True)
        target = launch("target", TARGET_PY, "jood_minigpt4_target.py", "generate")
        hf = None if args.defer_hf else launch("hf", EVAL_PY, "jood_minigpt4_evaluate.py", "hf", "--follow")
        report("running", "target" if hf is None else "target_and_hf")
        guard = None
        while True:
            # A later evaluator decision can attach HF scoring without stopping
            # a long GPU run. The corresponding smoke check must already pass.
            activation = RUN / "hf_activation.json"
            if hf is None and activation.exists():
                requested = json.loads(activation.read_text())
                current, _ = validate()
                checked = json.loads((RUN / "smoke/validation.json").read_text())
                assert checked["passed"] and checked["hf_model"] == current["hf_model"] == requested["hf_model"]
                hf = launch("hf", EVAL_PY, "jood_minigpt4_evaluate.py", "hf", "--follow")
            for name, proc in children.items():
                rc = proc.poll()
                if rc is not None and rc != 0:
                    if name == "target":
                        write(RUN / "target_failed.json", {"exit_code": rc})
                    raise RuntimeError(f"{name} failed with exit code {rc}; see {name}.log")
            if target.poll() == 0 and guard is None:
                assert (RUN / "target_complete.json").exists()
                guard = launch("guard", EVAL_PY, "jood_minigpt4_evaluate.py", "guard")
            if guard is not None and guard.poll() == 0 and (hf is None or hf.poll() == 0):
                break
            stage = "target" if guard is None else "guard"
            report("running", stage + ("_and_hf" if hf is not None else ""))
            time.sleep(15)
        if hf is None:
            report("awaiting_hf", "local_stages_complete", reason="HF evaluator choice pending; final paper-style ASR/HF requires maximum-HF selection")
            return
        report("running", "summarize")
        summary = launch("summarize", EVAL_PY, "jood_minigpt4_evaluate.py", "summarize")
        while summary.poll() is None:
            time.sleep(5)
        if summary.returncode != 0:
            raise RuntimeError("Summary validation failed; see summarize.log")
        assert json.loads((RUN / "final_validation.json").read_text())["passed"]
        report("complete", "complete")
    except BaseException:
        for process in children.values():
            if process.poll() is None:
                process.terminate()
        for process in children.values():
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        report("failed", "stopped_after_error")
        traceback.print_exc()
        raise
    finally:
        for stream in logs:
            stream.close()


if __name__ == "__main__":
    main()
