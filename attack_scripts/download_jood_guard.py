#!/usr/bin/env python3
"""Fetch the pinned JOOD judge with resumable HTTP ranges and verify SHA-256."""

from __future__ import annotations

import concurrent.futures
import contextlib
import datetime
import hashlib
import json
import os
import shutil
import threading
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from filelock import FileLock
from huggingface_hub import HfApi, constants, get_token, hf_hub_url

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/jood_reproduction_20261022"
CHUNK_SIZE = 32 * 1024 * 1024
HTTP = threading.local()


def status(**fields):
    value = {
        "status": "running",
        "phase": "guard_weights_download",
        "api_cases_completed": 782,
        "updated_at": datetime.datetime.now(ZoneInfo("Asia/Seoul")).isoformat(
            timespec="seconds"
        ),
        **fields,
    }
    temp = RUN / "status.json.tmp"
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(RUN / "status.json")
    print(json.dumps(value), flush=True)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def fetch_chunk(task):
    model, revision, item, begin, end, path = task
    expected = end - begin + 1
    if path.exists() and path.stat().st_size == expected:
        return expected
    if not hasattr(HTTP, "session"):
        HTTP.session = requests.Session()
    last_error = None
    for attempt in range(8):
        temp = path.with_suffix(".partial")
        try:
            url = hf_hub_url(model, item.rfilename, revision=revision)
            # A unique query avoids stale, expired redirects cached by gateways.
            url += f"?download=true&request_id={time.time_ns()}"
            headers = {
                "Authorization": "Bearer " + get_token(),
                "Range": f"bytes={begin}-{end}",
            }
            with HTTP.session.get(
                url, headers=headers, stream=True, timeout=(10, 40)
            ) as response:
                response.raise_for_status()
                assert response.status_code == 206, "Range request was not honored"
                assert (
                    response.headers.get("Content-Range")
                    == f"bytes {begin}-{end}/{item.size}"
                ), "Incorrect byte range"
                path.parent.mkdir(parents=True, exist_ok=True)
                size = 0
                with temp.open("wb") as stream:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        stream.write(block)
                        size += len(block)
            assert size == expected, "Incomplete HTTP range"
            temp.replace(path)
            return expected
        except (requests.RequestException, AssertionError, OSError) as exc:
            last_error = type(exc).__name__
            if attempt < 7:
                time.sleep(min(2**attempt, 15))
    raise RuntimeError(f"{item.rfilename} range {begin}-{end} failed ({last_error})")


def main():
    manifest = json.loads((RUN / "manifest.json").read_text())
    model, revision = manifest["guard_model"], manifest["guard_revision"]
    info = HfApi().model_info(model, revision=revision, files_metadata=True)
    files = [item for item in info.siblings if item.rfilename.endswith(".safetensors")]
    assert len(files) == 4
    cache = Path(constants.HF_HUB_CACHE) / ("models--" + model.replace("/", "--"))
    blobs, snapshot = cache / "blobs", cache / "snapshots" / revision
    locks = (
        Path(constants.HF_HUB_CACHE)
        / ".locks"
        / ("models--" + model.replace("/", "--"))
    )
    for path in [blobs, snapshot, locks]:
        path.mkdir(parents=True, exist_ok=True)
    checked, pending = [], []
    workers = int(os.environ.get("JOOD_DOWNLOAD_WORKERS", "16"))
    status(method="resumable HTTP ranges", http_concurrency=workers)
    for item in files:
        target = blobs / item.lfs.sha256
        if (
            target.exists()
            and target.stat().st_size == item.size
            and digest(target) == item.lfs.sha256
        ):
            checked.append(
                {
                    "file": item.rfilename,
                    "size": item.size,
                    "sha256": item.lfs.sha256,
                    "matches_official_lfs": True,
                }
            )
        else:
            pending.append(item)
    tasks = []
    for item in pending:
        for begin in range(0, item.size, CHUNK_SIZE):
            end = min(begin + CHUNK_SIZE, item.size) - 1
            path = (
                RUN
                / "guard_http_parts"
                / item.rfilename
                / f"{begin:012d}-{end:012d}.part"
            )
            tasks.append((model, revision, item, begin, end, path))
    total_bytes = sum(end - begin + 1 for _, _, _, begin, end, _ in tasks)
    downloaded = completed = 0
    errors = []
    started = time.monotonic()
    with contextlib.ExitStack() as stack:
        for item in pending:
            stack.enter_context(
                FileLock(locks / (item.lfs.sha256 + ".lock"), timeout=30)
            )
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fetch_chunk, task): task for task in tasks}
            for future in concurrent.futures.as_completed(futures):
                try:
                    downloaded += future.result()
                    completed += 1
                except Exception as exc:  # noqa: BLE001 - journal all failed ranges, never publish incomplete weights
                    errors.append(
                        {
                            "file": futures[future][2].rfilename,
                            "begin": futures[future][3],
                            "error_type": type(exc).__name__,
                        }
                    )
                status(
                    completed_chunks=completed,
                    total_chunks=len(tasks),
                    downloaded_bytes=downloaded,
                    total_bytes=total_bytes,
                    completed_weight_files=len(checked),
                    elapsed_seconds=round(time.monotonic() - started, 1),
                    failed_chunks=len(errors),
                )
        if errors:
            (RUN / "guard_download_errors.json").write_text(
                json.dumps(errors, indent=2)
            )
            raise SystemExit(
                "Some HTTP ranges failed; successful parts are preserved for retry."
            )
        for item in pending:
            assembling = blobs / (item.lfs.sha256 + ".http-assembling")
            checksum = hashlib.sha256()
            with assembling.open("wb") as out:
                for begin in range(0, item.size, CHUNK_SIZE):
                    end = min(begin + CHUNK_SIZE, item.size) - 1
                    part = (
                        RUN
                        / "guard_http_parts"
                        / item.rfilename
                        / f"{begin:012d}-{end:012d}.part"
                    )
                    with part.open("rb") as stream:
                        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                            checksum.update(block)
                            out.write(block)
            assert assembling.stat().st_size == item.size
            assert checksum.hexdigest() == item.lfs.sha256, (
                "Full official weight checksum mismatch"
            )
            assembling.replace(blobs / item.lfs.sha256)
            checked.append(
                {
                    "file": item.rfilename,
                    "size": item.size,
                    "sha256": item.lfs.sha256,
                    "matches_official_lfs": True,
                }
            )
    for item in files:
        link = snapshot / item.rfilename
        target = blobs / item.lfs.sha256
        if not link.exists():
            link.symlink_to(os.path.relpath(target, snapshot))
    assert len(checked) == 4
    value = {
        "model": model,
        "revision": revision,
        "snapshot_path": str(snapshot),
        "download_method": "resumable verified HTTP byte ranges, Xet disabled",
        "all_weight_sha256_verified": True,
        "files": sorted(checked, key=lambda x: x["file"]),
    }
    for name in ["guard_snapshot.json", "guard_weights_validation.json"]:
        (RUN / name).write_text(json.dumps(value, indent=2) + "\n")
    # Only successful, checksum-verified intermediate chunks are removed.
    if (RUN / "guard_http_parts").exists():
        shutil.rmtree(RUN / "guard_http_parts")
    print(
        json.dumps({"phase": "guard_weights_verified", "files": len(checked)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
