"""Detached supervisor for the existing Financial Advice experiment.

It observes the already running workers, resumes a stopped main worker from
its verified cache, then replays the report and runs the CPU artifact verifier.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def active(pid, module):
    if not pid:
        return False
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        return fields[2] != "Z" and module in command
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False


def update(folder, state, **changes):
    state.update(changes)
    state["updated_at"] = stamp()
    temporary = folder / "supervisor_status.json.tmp"
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(folder / "supervisor_status.json")


def repair_interrupted_writes(folder, skip_ids=()):
    recovery = folder / "resume_recovery"
    changes = []
    for name in ("baseline.jsonl", "evaluation.jsonl"):
        path = folder / name
        if not path.exists():
            continue
        raw = path.read_bytes()
        lines = raw.splitlines(keepends=True)
        repaired = raw
        for index, line in enumerate(lines):
            try:
                json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if index != len(lines) - 1 or line.endswith(b"\n"):
                    raise ValueError(f"Invalid complete JSONL record in {name}; refusing automatic repair") from None
                repaired = b"".join(lines[:-1])
        if repaired and not repaired.endswith(b"\n"):
            repaired += b"\n"
        if repaired != raw:
            recovery.mkdir(exist_ok=True)
            backup = recovery / f"{name}.{time.time_ns()}.before_repair"
            backup.write_bytes(raw)
            path.write_bytes(repaired)
            changes.append({"file": name, "backup": str(backup), "reason": "interrupted final JSONL write"})
    for path in (folder / "inputs").glob("*/*/optimization.json"):
        if path.parent.parent.name in skip_ids:
            continue
        try:
            meta = json.loads(path.read_text())
            valid = (digest(path.parent / "input.safetensors") == meta["tensor_sha256"]
                     and digest(path.parent / "image.png") == meta["png_sha256"])
        except (json.JSONDecodeError, FileNotFoundError, KeyError):
            valid = False
        if not valid:
            recovery.mkdir(exist_ok=True)
            backup = recovery / f"{path.parent.parent.name}_{path.parent.name}.{time.time_ns()}.optimization.json"
            shutil.copyfile(path, backup)
            path.unlink()
            changes.append({"file": str(path), "backup": str(backup), "reason": "interrupted optimization cache write; rerun its 100 steps"})
    return changes


def run_command(command, folder, log_name, env):
    with (folder / log_name).open("a") as output:
        subprocess.run(command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--labels-from", type=Path, required=True)
    parser.add_argument("--main-pid", type=int, required=True)
    parser.add_argument("--aux-pid", type=int, default=0)
    args = parser.parse_args()
    folder = args.run.resolve()
    lock = (folder / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(os.environ)
    env.update({"PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1", "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"})
    base = [sys.executable, "-u", "-m", "attack_scripts.jailbound_financial_reproduction",
            "--labels-from", str(args.labels_from.resolve()), "--out", str(folder)]
    state = {"started_at": stamp(), "supervisor_pid": os.getpid(), "main_pid": args.main_pid,
             "aux_pid": args.aux_pid, "resume_attempts": 0, "recovery_changes": [],
             "supervisor_script_sha256": digest(Path(__file__)), "status": "observing_existing_workers"}
    child = None
    main_pid = args.main_pid
    update(folder, state)
    try:
        while True:
            if active(main_pid, "attack_scripts.jailbound_financial_reproduction"):
                update(folder, state, main_pid=main_pid, status="running")
                time.sleep(10)
                continue
            if child is not None:
                child.wait()
            if (folder / "summary.json").exists():
                break
            if state["resume_attempts"] >= 3:
                raise RuntimeError("Main worker stopped without a full summary after three resume attempts")
            # Preserve cached artifacts and avoid touching the auxiliary work front.
            skip_ids = set()
            if active(args.aux_pid, "attack_scripts.jailbound_financial_aux_optimize"):
                source = json.loads((folder / "source_cases.json").read_text())
                skip_ids = {row["id"] for row in source["evaluation"][-50:]}
            metadata_repairs = repair_interrupted_writes(folder, skip_ids)
            state["recovery_changes"].extend(metadata_repairs)
            state["resume_attempts"] += 1
            update(folder, state, status="resuming_main_worker")
            output = (folder / "worker.log").open("a")
            child = subprocess.Popen(base + ["--stage", "all"], cwd=ROOT, env=env,
                                     stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            output.close()
            main_pid = child.pid
        while active(args.aux_pid, "attack_scripts.jailbound_financial_aux_optimize"):
            update(folder, state, status="waiting_for_auxiliary_exit")
            time.sleep(10)
        update(folder, state, status="replaying_report")
        watched = [folder / name for name in ("baseline.jsonl", "evaluation.jsonl", "boundaries.pt")]
        watched += list((folder / "inputs").glob("*/*/input.safetensors"))
        before = {str(path.relative_to(folder)): digest(path) for path in watched}
        run_command(base + ["--stage", "report"], folder, "report_replay.log", env)
        after = {str(path.relative_to(folder)): digest(path) for path in watched}
        if before != after:
            raise RuntimeError("Report replay modified generated inputs, responses or boundaries")
        (folder / "report_replay_integrity.json").write_text(json.dumps({"status": "passed", "unchanged_files": len(before), "sha256": before}, indent=2) + "\n")
        oracle = folder / "heldout_self_assessment"
        if (oracle / "summary.json").exists():
            saved = json.loads((oracle / "summary.json").read_text())
            changed = [name for name, value in saved["source_sha256"].items() if digest(folder / name) != value]
            if changed:
                if changed != ["heldout_probe_review.json"]:
                    raise RuntimeError(f"Unexpected oracle validation source changes: {changed}")
                oracle.rename(folder / f"heldout_self_assessment_before_resume_{time.time_ns()}")
        if not (oracle / "summary.json").exists():
            update(folder, state, status="validating_self_assessment")
            run_command([sys.executable, "-u", "-m", "attack_scripts.jailbound_financial_probe_validation", "--run", str(folder)],
                        folder, "probe_validation_after_resume.log", env)
        update(folder, state, status="verifying_artifacts")
        run_command([sys.executable, "-m", "attack_scripts.jailbound_verify_financial", "--run", str(folder)],
                    folder, "verification.log", env)
        update(folder, state, status="complete", completed_at=stamp(), verification_file="verification.json",
               summary_file="summary.md", summary_sha256=digest(folder / "summary.json"))
    except Exception as error:
        update(folder, state, status="error", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
