#!/usr/bin/env python3
"""Run the frozen JOOD MiniGPT-4 protocol on two isolated GPU workers.

The original protocol modules are reused without editing them. Binding their
output directory and input validator gives each worker a separate journal.
Whole original batches are assigned alternately, preserving each batch's seed.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

import jood_minigpt4_common as common

ROOT = common.ROOT
SOURCE = ROOT / "runs/jood_minigpt4_7b_20261102"
RUN = ROOT / "runs/jood_minigpt4_7b_90g2_20261102"
SCRIPT = Path(__file__).resolve()
TARGET_PY = "/home/jovyan/.conda/envs/visualadv_venv/bin/python"
EVAL_PY = "/home/jovyan/.conda/envs/jb_env/bin/python"
sha, write, digest = common.sha, common.write, common.digest


def rows(path):
    """JSONL records end at LF, not at Unicode line separators inside strings."""
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_bytes().split(b"\n") if line.strip()]


def live_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    end = data.rfind(b"\n")
    if end < 0:
        return []
    return [json.loads(line) for line in data[:end].split(b"\n") if line.strip()]


def partition(cases, worker):
    return [c for start in range(0, len(cases), 8) if (start // 8) % 2 == worker
            for c in cases[start:start + 8]]


def validate_all():
    m = json.loads((RUN / "manifest.json").read_text())
    for name, expected in m["code_sha256"].items():
        assert sha(ROOT / name) == expected, name
    for name, expected in m["frozen_inputs_sha256"].items():
        assert sha(RUN / name) == expected, name
    cases = rows(RUN / "test_cases.jsonl")
    assert len(cases) == len({c["custom_id"] for c in cases}) == 7820
    groups = collections.Counter((c["scenario"], c["arm"], c["prompt_idx"]) for c in cases)
    assert len(groups) == 340
    assert all(n == (1 if arm == "Vanilla" else 45) for (_, arm, _), n in groups.items())
    parts = [partition(cases, i) for i in range(2)]
    ids = [{c["custom_id"] for c in part} for part in parts]
    assert not ids[0] & ids[1] and ids[0] | ids[1] == {c["custom_id"] for c in cases}
    assert [len(part) for part in parts] == [3912, 3908]
    return m, cases


def prepare():
    RUN.mkdir(exist_ok=True)
    if (RUN / "manifest.json").exists():
        return validate_all()
    original = json.loads((SOURCE / "manifest.json").read_text())
    assert original["target_batch_size"] == 8
    for name, expected in original["code_sha256"].items():
        assert sha(ROOT / name) == expected, name
    for name, expected in original["frozen_inputs_sha256"].items():
        assert sha(SOURCE / name) == expected, name
        shutil.copyfile(SOURCE / name, RUN / name)
    for name in ["asset_provenance.json", "image_provenance.json"]:
        for record in rows_of_json(SOURCE / name):
            assert sha(record["path"]) == record["sha256"], record["path"]
    data = SOURCE / "official_data"
    (RUN / "official_data").symlink_to(data, target_is_directory=True)
    for i in range(2):
        worker = RUN / ("worker" + str(i))
        worker.mkdir()
        (worker / "official_data").symlink_to(data, target_is_directory=True)
        (worker / "model_assets.json").symlink_to(RUN / "model_assets.json")
    m = dict(original)
    m.update(run_directory=str(RUN), host=socket.gethostname(), independent_run=True,
             source_input_directory=str(SOURCE), source_results_reused=False,
             target_workers=2, worker_case_counts=[3912, 3908],
             partition="Original batch index modulo 2; batch size 8 and seed derivation unchanged",
             hf_status="awaiting_evaluator_preference")
    m["code_sha256"] = dict(original["code_sha256"])
    m["code_sha256"][str(SCRIPT.relative_to(ROOT))] = sha(SCRIPT)
    m["limitations"] = [s for s in m["limitations"] if "returned model_not_found" not in s]
    m["limitations"] += [
        "Independent two-GPU run on B200 MIG 3g.90gb; whole batches assigned alternately, with unchanged target settings and per-batch seeds.",
        "Original GPT-4 Turbo HF judge is unavailable. HF and maximum-HF-selected ASR remain pending until an alternative evaluator is selected and validated.",
    ]
    write(RUN / "manifest.json", m)
    write(RUN / "source_state_at_prepare.json", {
        "background": json.loads((SOURCE / "background_status.json").read_text()),
        "target": json.loads((SOURCE / "target_status.json").read_text()),
        "note": "Source results preserved and not reused; this is a separate complete run.",
    })
    m, cases = validate_all()
    write(RUN / "input_validation.json", {"passed": True, "cases": len(cases),
        "workers": [3912, 3908], "disjoint_and_exhaustive": True,
        "source_input_hashes_preserved": True, "all_asset_and_image_hashes_verified": True})
    return m, cases


def rows_of_json(path):
    return json.loads(path.read_text())


def bind(folder, selected_cases):
    """Set only this child process's module globals; no shared files are edited."""
    common.RUN = folder
    common.DATA = RUN / "official_data"
    common.rows = rows

    def checked():
        manifest, all_cases = validate_all()
        return manifest, selected_cases(all_cases)

    common.validate = checked


def worker_stage(stage, worker):
    folder = RUN / ("worker" + str(worker))
    smoke = stage.startswith("smoke")
    bind(folder, (lambda c: c) if smoke else (lambda c: partition(c, worker)))
    if stage in {"smoke-target", "target"}:
        import jood_minigpt4_target as target
        sys.argv = [str(SCRIPT), "smoke" if smoke else "generate"]
        target.main()
    else:
        import jood_minigpt4_evaluate as evaluate
        manifest, cases = common.validate()
        if smoke:
            cases = common.smoke_cases(cases)
            evaluate.guard(manifest, cases, folder / "smoke")
            evaluate.validate_smoke(cases, manifest, local_only=True)
        else:
            evaluate.guard(manifest, cases, folder)


def merge_local():
    _, cases = validate_all()
    merged = {}
    for name in ["responses.jsonl", "guard_results.jsonl"]:
        records = [r for i in range(2) for r in rows(RUN / ("worker" + str(i)) / name)]
        found = common.index(records)
        assert set(found) == {c["custom_id"] for c in cases}
        ordered = [found[c["custom_id"]] for c in cases]
        tmp = RUN / (name + ".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered))
        tmp.replace(RUN / name)
        merged[name] = found
    responses, guards = merged["responses.jsonl"], merged["guard_results.jsonl"]
    for cid, response in responses.items():
        assert guards[cid]["response_sha256"] == digest(response["response"])
        assert guards[cid]["output_is_standard"]
    write(RUN / "local_results_validation.json", {"passed": True, "responses": 7820,
        "guard_labels": 7820, "no_duplicates_or_missing_cases": True,
        "hf_and_final_asr_pending": True, "ppl_excluded": True})


def report(state, stage, children, started, **extra):
    progress = {}
    for i in range(2):
        detail = {}
        for name in ["target", "guard", "smoke_target", "smoke_guard"]:
            path = RUN / ("worker" + str(i)) / (name + "_status.json")
            if path.exists():
                detail[name] = json.loads(path.read_text())
        progress[str(i)] = detail
    value = {"status": state, "current_stage": stage, "host": socket.gethostname(),
        "updated_at": datetime.datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds"),
        "pid": os.getpid(), "session_id": os.getsid(0), "elapsed_seconds": round(time.monotonic() - started, 1),
        "children": {name: {"pid": p.pid, "exit_code": p.poll()} for name, p in children.items()},
        "progress": progress, **extra}
    write(RUN / "background_status.json", value)
    print(json.dumps(value), flush=True)


def background():
    lock = (RUN / "background.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    validate_all()
    assert socket.gethostname() == "yong-90g-2ea-0", "Wrong execution host"
    assert json.loads((RUN / "model_assets.json").read_text())["status"] == "complete"
    children, streams = {}, []
    started = time.monotonic()

    def stop(signum, frame):
        raise InterruptedError(f"Supervisor received signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    manifest, cases = validate_all()
    smoke_verified = True
    for worker in range(2):
        folder = RUN / ("worker" + str(worker))
        try:
            checked = json.loads((folder / "smoke/local_validation.json").read_text())
            runtime = json.loads((folder / "target_runtime.json").read_text())
            guard_runtime = json.loads((folder / "smoke/guard_runtime.json").read_text())
            responses = common.index(rows(folder / "smoke/responses.jsonl"))
            guards = common.index(rows(folder / "smoke/guard_results.jsonl"))
            expected = {c["custom_id"] for c in common.smoke_cases(cases)}
            assert checked["passed"] and set(responses) == set(guards) == expected
            assert runtime["generation"] == manifest["target_generation"]
            assert runtime["linear8bit_layers"] == 224
            assert guard_runtime["revision"] == manifest["guard_revision"]
            assert all(guards[c]["output_is_standard"] and guards[c]["response_sha256"] == digest(responses[c]["response"]) for c in expected)
        except (OSError, ValueError, KeyError, AssertionError):
            smoke_verified = False
    try:
        for stage in ["smoke-target", "smoke-guard", "target", "guard"]:
            if smoke_verified and stage.startswith("smoke-"):
                report("running", stage + "_already_validated", children, started)
                continue
            jobs = []
            for worker in range(2):
                name = stage + "-" + str(worker)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(worker), PYTHONNOUSERSITE="1",
                    PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                    TOKENIZERS_PARALLELISM="false")
                log = (RUN / ("worker" + str(worker)) / (stage + ".log")).open("a")
                streams.append(log)
                python = TARGET_PY if stage.endswith("target") or stage == "target" else EVAL_PY
                p = subprocess.Popen([python, "-u", str(SCRIPT), stage, "--worker", str(worker)],
                    cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                children[name] = p
                jobs.append(p)
            while True:
                report("running", stage, children, started)
                for p in jobs:
                    if p.poll() is not None and p.returncode != 0:
                        raise RuntimeError(f"{stage} PID {p.pid} failed with exit code {p.returncode}")
                if all(p.poll() == 0 for p in jobs):
                    break
                time.sleep(15)
        merge_local()
        report("awaiting_hf", "local_stages_complete", children, started,
               reason="Target and Guard complete; HF evaluator choice remains pending")
    except BaseException:
        for p in children.values():
            if p.poll() is None:
                p.terminate()
        for p in children.values():
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        report("failed", "stopped_after_error", children, started)
        traceback.print_exc()
        raise
    finally:
        for log in streams:
            log.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prepare", "background", "smoke-target", "smoke-guard", "target", "guard", "merge", "hf", "summarize"])
    p.add_argument("--worker", type=int, choices=[0, 1])
    a = p.parse_args()
    if a.stage == "prepare":
        m, cases = prepare()
        print(json.dumps({"prepared": len(cases), "worker_cases": m["worker_case_counts"]}))
    elif a.stage == "background":
        background()
    elif a.stage == "merge":
        merge_local()
    elif a.stage in {"hf", "summarize"}:
        bind(RUN, lambda c: c)
        import jood_minigpt4_evaluate as evaluate
        evaluate.live_rows = live_rows
        m, cases = common.validate()
        assert m["hf_status"] == "alternative_evaluator_selected", "HF evaluator choice required"
        if a.stage == "hf":
            evaluate.hf(m, cases, RUN, follow=False)
        else:
            evaluate.summarize(m, cases)
    else:
        assert a.worker is not None
        worker_stage(a.stage, a.worker)


if __name__ == "__main__":
    main()
