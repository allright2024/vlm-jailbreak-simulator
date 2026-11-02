"""Resume public HF LFS files with bounded ranges; verify published whole-file SHA."""
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import threading
import time
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlparse

import requests

RUN = Path(__file__).resolve().parent


class DownloadGate:
    """Share a request pace and server-directed cooldown across all workers."""

    def __init__(self):
        self.lock = threading.Lock()
        self.next_request = 0.0
        self.cooldown_until = 0.0

    def wait(self):
        while True:
            with self.lock:
                now = time.monotonic()
                delay = max(self.next_request, self.cooldown_until) - now
                if delay <= 0:
                    self.next_request = now + 0.5  # At most two requests/second.
                    return
            time.sleep(min(delay, 1.0))

    def cooldown(self, response, attempt):
        retry = response.headers.get("Retry-After", "")
        try:
            seconds = float(retry)
        except ValueError:
            try:
                seconds = parsedate_to_datetime(retry).timestamp() - time.time()
            except (TypeError, ValueError, OverflowError):
                seconds = min(30 * 2**attempt, 300)
        seconds = max(seconds, 1.0)
        reset = re.search(r"(?:^|;)\s*t=(\d+)", response.headers.get("RateLimit", ""))
        if reset:
            seconds = max(seconds, float(reset.group(1)) + 1)
        with self.lock:
            self.cooldown_until = max(self.cooldown_until, time.monotonic() + seconds)
        print("DOWNLOAD_RATE_LIMIT_COOLDOWN", round(seconds, 1), "seconds", flush=True)


class PublicDownloadURL:
    """Reuse the standard Hub redirect instead of resolving every byte range."""

    def __init__(self, url, gate):
        self.url = url
        self.gate = gate
        self.lock = threading.Lock()
        self.cached = None
        self.cached_at = 0.0

    def get(self):
        with self.lock:
            if self.cached and time.monotonic() - self.cached_at < 1200:
                return self.cached
            for attempt in range(12):
                self.gate.wait()
                with requests.get(self.url, allow_redirects=False, stream=True, timeout=(10, 20)) as response:
                    if response.status_code == 429:
                        self.gate.cooldown(response, attempt)
                        continue
                    if response.status_code not in (301, 302, 303, 307, 308):
                        raise RuntimeError(f"Public model resolver returned HTTP {response.status_code}")
                    destination = response.headers["Location"]
                    parsed = urlparse(destination)
                    assert parsed.scheme == "https"
                    assert parsed.hostname and (parsed.hostname.endswith(".hf.co") or parsed.hostname.endswith(".huggingface.co"))
                    self.cached, self.cached_at = destination, time.monotonic()
                    return destination
            raise RuntimeError("Model resolver remains rate-limited after server-directed cooldowns")

    def invalidate(self):
        with self.lock:
            self.cached = None


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def fetch_weight(meta, entry, snapshot):
    expected = entry["lfs"]["sha256"]
    size = entry["lfs"]["size"]
    blob = Path(snapshot).parents[1] / "blobs" / expected
    link = Path(snapshot) / entry["name"]
    if blob.exists() and blob.stat().st_size == size and digest(blob) == expected:
        if not link.exists():
            link.symlink_to(os.path.relpath(blob, link.parent))
        return
    pieces = RUN / "download_parts" / expected
    pieces.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{meta['id']}/resolve/{meta['sha']}/{entry['name']}"
    gate = DownloadGate()
    download_url = PublicDownloadURL(url, gate)
    chunk_size = 1024 * 1024
    tasks = [(start, min(start + chunk_size, size)) for start in range(0, size, chunk_size)]

    def fetch(task):
        start, end = task
        path = pieces / str(start)
        if path.exists() and path.stat().st_size == end - start:
            return end - start
        tmp = path.with_suffix(".part")
        for attempt in range(12):
            try:
                destination = download_url.get()
                gate.wait()
                began = time.monotonic()
                with requests.get(destination, headers={"Range": f"bytes={start}-{end-1}", "Accept-Encoding": "identity"},
                                  timeout=(10, 15), stream=True) as response:
                    if response.status_code == 429:
                        gate.cooldown(response, attempt)
                        continue
                    if response.status_code == 403:
                        download_url.invalidate()  # Refresh an expired public signed URL.
                        continue
                    if response.status_code >= 400:
                        raise RuntimeError(f"Public model range returned HTTP {response.status_code}")
                    assert response.status_code == 206
                    assert response.headers["Content-Range"] == f"bytes {start}-{end-1}/{size}"
                    with tmp.open("wb") as f:
                        for block in response.iter_content(65536):
                            f.write(block)
                            if time.monotonic() - began > 35:
                                raise TimeoutError("Range transfer exceeded 35 seconds")
                assert tmp.stat().st_size == end - start
                tmp.replace(path)
                return end - start
            except (requests.RequestException, TimeoutError, RuntimeError, AssertionError, OSError) as exc:
                if attempt == 11:
                    raise RuntimeError(f"Range {start}-{end-1} failed after retries ({type(exc).__name__}); completed ranges preserved") from None
                time.sleep(min(2**attempt, 30))
        raise RuntimeError(f"Range {start}-{end-1} exhausted retries; completed ranges preserved")

    done = 0
    began = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for index, future in enumerate(concurrent.futures.as_completed([pool.submit(fetch, task) for task in tasks]), 1):
            done += future.result()
            if index % 32 == 0 or index == len(tasks):
                status = {"phase": "downloading_original_weights", "file": entry["name"],
                          "repo": meta["id"], "completed_bytes": done, "total_bytes": size,
                          "completed_parts": index, "parts": len(tasks), "seconds": time.monotonic()-began,
                          "updated_at_epoch": time.time()}
                tmp_status = RUN / "download_status.json.tmp"
                tmp_status.write_text(json.dumps(status, indent=2))
                tmp_status.replace(RUN / "download_status.json")
                print("DOWNLOAD", entry["name"], f"{done}/{size}", flush=True)
    temp_blob = blob.with_suffix(".ranged_incomplete")
    with temp_blob.open("wb") as target:
        for start, _ in tasks:
            with (pieces / str(start)).open("rb") as f:
                shutil.copyfileobj(f, target)
    assert temp_blob.stat().st_size == size
    assert digest(temp_blob) == expected, "Published original checkpoint SHA mismatch"
    temp_blob.replace(blob)
    if link.is_symlink() or link.exists():
        assert link.resolve() == blob.resolve()
    else:
        link.symlink_to(os.path.relpath(blob, link.parent))
    for p in pieces.iterdir():
        p.unlink()
    pieces.rmdir()
    print("WEIGHT_VERIFIED", entry["name"], expected, flush=True)
