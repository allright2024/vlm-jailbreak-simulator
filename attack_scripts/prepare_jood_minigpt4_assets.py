#!/usr/bin/env python3
"""Fetch the official JOOD data and MiniGPT-4 Vicuna-v0 7B assets."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/jood_minigpt4_7b_20261102"
MODEL_ROOT = Path("/tmp/jood_minigpt4_7b")
REVISION = "0abeb8872d75421e58ec146c39b31f5dc78c7613"
SCENARIOS = ["bomb_explosive", "drugs", "firearms_weapons", "hack_information",
             "kill_someone", "social_violence", "suicide"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def folder(fid):
    cache = RUN / "drive_index" / (fid + ".json")
    if cache.exists():
        return json.loads(cache.read_text())
    parser = importlib.import_module("gdown.download_folder")._parse_embedded_folder_view
    name, children = parser(sess=requests.Session(), folder_id=fid, verify=True, timeout=30)
    result = {"id": fid, "name": name, "children": children}
    write(cache, result)
    return result


def child(fid, name):
    return next(c[0] for c in folder(fid)["children"] if c[1] == name)


def drive_download(item):
    fid, path = item
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        for attempt in range(6):
            try:
                endpoint = ["https://drive.google.com/uc", "https://drive.usercontent.google.com/download"][attempt % 2]
                r = requests.get(endpoint, params={"id": fid, "export": "download"}, timeout=(10, 45))
                r.raise_for_status()
                tmp = path.with_suffix(path.suffix + ".download")
                tmp.write_bytes(r.content)
                if path.suffix == ".json":
                    json.loads(tmp.read_text())
                else:
                    with Image.open(tmp) as im:
                        im.verify()
                tmp.replace(path)
                break
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(2)
    return {"drive_id": fid, "path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def data():
    root_id = "1Mgx3jRd9slB4FhwCDrwuMTTifB6NtNAO"
    target = RUN / "official_data"
    tasks = [(child(root_id, n), target / n) for n in ["scenario_def.json", "scenario_repr.json"]]
    prompts = child(root_id, "prompts")
    for group in ["all_instructions", "eval_all_instructions"]:
        parent = child(prompts, group)
        tasks.extend((child(parent, s + ".json"), target / "prompts" / group / (s + ".json")) for s in SCENARIOS)
    images = child(root_id, "images")
    for fid, name, kind in folder(child(images, "harmless"))["children"]:
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            tasks.append((fid, target / "images/harmless" / name))
    harmful = child(images, "harmful")
    for s in SCENARIOS:
        items = [c for c in folder(child(harmful, s))["children"] if c[1].lower().endswith((".jpg", ".jpeg", ".png"))]
        assert len(items) == 1, (s, items)
        tasks.extend((fid, target / "images/harmful" / s / name) for fid, name, _ in items)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(drive_download, tasks))
    write(RUN / "asset_provenance.json", records)
    counts = {}
    for s in SCENARIOS:
        a = json.loads((target / "prompts/all_instructions" / (s + ".json")).read_text())["instructions"]
        b = json.loads((target / "prompts/eval_all_instructions" / (s + ".json")).read_text())["instructions"]
        assert len(a) == len(b)
        counts[s] = len(a)
    assert sum(counts.values()) == 170
    write(RUN / "data_preparation.json", {"status": "complete", "n_instructions": 170, "counts": counts, "files": len(records)})
    print(json.dumps({"data": "complete", "counts": counts}), flush=True)


def models():
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    metadata = requests.get("https://huggingface.co/api/models/Vision-CAIR/vicuna-7b/revision/" + REVISION + "?blobs=true", timeout=30)
    metadata.raise_for_status()
    info = metadata.json()
    assert info["sha"] == REVISION
    write(MODEL_ROOT / "vicuna_repo_metadata.json", info)
    output = MODEL_ROOT / "vicuna-7b"
    output.mkdir(exist_ok=True)
    large = []
    for f in info["siblings"]:
        name = f["rfilename"]
        if name == ".gitattributes":
            continue
        p = output / name
        if f.get("size", 0) > 100_000_000:
            large.append(name + ":" + f["lfs"]["sha256"])
            continue
        if not p.exists():
            r = requests.get(f"https://huggingface.co/Vision-CAIR/vicuna-7b/resolve/{REVISION}/{name}", timeout=60)
            r.raise_for_status()
            p.write_bytes(r.content)
        if f.get("lfs"):
            assert sha(p) == f["lfs"]["sha256"]
        else:
            b = p.read_bytes()
            assert hashlib.sha1(b"blob " + str(len(b)).encode() + b"\0" + b).hexdigest() == f["blobId"]
    checkpoint = MODEL_ROOT / "pretrained_minigpt4_7b.pth"
    if not checkpoint.exists():
        r = requests.get("https://drive.usercontent.google.com/download",
                         params={"id": "1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R", "export": "download", "confirm": "t"}, timeout=90)
        r.raise_for_status()
        assert r.content[:2] == b"PK" and len(r.content) > 1_000_000
        tmp = checkpoint.with_suffix(".download")
        tmp.write_bytes(r.content)
        tmp.replace(checkpoint)
    # Use the existing resumable range downloader with a pinned revision URL.
    sys.path.insert(0, str(ROOT.parent / "models"))
    import hf_range_download as downloader
    downloader.RESOLVE = "https://huggingface.co/{repo}/resolve/" + REVISION + "/{name}"
    sys.argv = ["hf_range_download", "--repo", "Vision-CAIR/vicuna-7b", "--out", str(output), "--threads", "12", *large]
    downloader.main()
    records = []
    for f in info["siblings"]:
        p = output / f["rfilename"]
        if not p.exists():
            continue
        value = sha(p)
        if f.get("lfs"):
            assert value == f["lfs"]["sha256"]
        records.append({"path": str(p), "sha256": value})
    records.append({"path": str(checkpoint), "sha256": sha(checkpoint), "official_drive_id": "1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R"})
    cfg = json.loads((output / "config.json").read_text())
    assert cfg["hidden_size"] == 4096 and cfg["num_hidden_layers"] == 32
    write(RUN / "model_assets.json", {"status": "complete", "model": "MiniGPT-4 Vicuna-v0 7B", "llm_path": str(output), "checkpoint": str(checkpoint), "hf_revision": REVISION, "files": records})
    print('MODEL_ASSETS_COMPLETE', flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["data", "models"])
    a = p.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    (data if a.stage == "data" else models)()
