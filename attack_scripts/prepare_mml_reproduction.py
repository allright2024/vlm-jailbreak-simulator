#!/usr/bin/env python3
"""Prepare a paired, category-balanced MML-M/FigStep reproduction subset."""

from __future__ import annotations

import concurrent.futures
import hashlib
import importlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import gdown
import requests
import yaml
from PIL import Image, ImageChops, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.datasets.safebench import SafeBenchDataset
from core.io import write_jsonl
from core.schema import TestCase
from multi_attacks.mml.mml import MMLAttack
from multi_attacks.mml.official_source import load_upstream_constants

RUN = ROOT / "runs/mml_reproduction_20261022"
DATA = ROOT / "data/MML"
REV = "5966fca529011e8b5ccff61d70365997d6466abc"
SOURCE_HASH = "89d7480ff98a4872b50f16c8bf4271f37074127fa0c1cfdd46a1f50b2cdce9a7"
PARSER = importlib.import_module("gdown.download_folder")._parse_embedded_folder_view


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def folder(fid):
    cache = DATA / ".drive_index" / f"{fid}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    name, children = PARSER(
        sess=requests.Session(), folder_id=fid, verify=True, timeout=25
    )
    value = {"id": fid, "name": name, "children": children}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(value, indent=2))
    return value


def download(item):
    fid, path = item
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temp = Path(str(path) + ".download")
        result = gdown.download(
            id=fid,
            output=str(temp),
            quiet=True,
            use_cookies=False,
            timeout=(10, 35),
            retries=1,
        )
        if result is None:
            raise RuntimeError(f"Official Drive download failed for {fid}")
        temp.replace(path)
    return {
        "drive_file_id": fid,
        "path": str(path.relative_to(ROOT)),
        "sha256": sha(path),
        "bytes": path.stat().st_size,
    }


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    existing = RUN / "manifest.json"
    if existing.exists():
        raise SystemExit(
            "Prepared manifest already exists; refusing to replace frozen inputs"
        )
    index = json.loads((DATA / "drive_safebench_index.json").read_text())
    categories = [
        (fid, name)
        for fid, name, kind in index["children"]
        if kind == "application/vnd.google-apps.folder"
    ]
    assert len(categories) == 10
    category_indices = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        tasks = {pool.submit(folder, fid): name for fid, name in categories}
        for future in concurrent.futures.as_completed(tasks):
            category_indices[tasks[future]] = future.result()
    metadata_tasks = []
    image_folder_tasks = []
    for category, value in category_indices.items():
        entries = {name: (fid, kind) for fid, name, kind in value["children"]}
        assert "data.json" in entries, category
        metadata_tasks.append(
            (entries["data.json"][0], DATA / "safebench" / category / "data.json")
        )
        for method in ["images_mirror", "images_figstep"]:
            assert method in entries, (category, method, list(entries))
            image_folder_tasks.append((category, method, entries[method][0]))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        provenance = list(pool.map(download, metadata_tasks))
    print(json.dumps({"stage": "metadata_ready", "categories": 10}), flush=True)
    image_indices = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        tasks = {
            pool.submit(folder, fid): (category, method)
            for category, method, fid in image_folder_tasks
        }
        for future in concurrent.futures.as_completed(tasks):
            image_indices[tasks[future]] = {
                name: fid for fid, name, _ in future.result()["children"]
            }
    behaviors = SafeBenchDataset().load()
    assert len(behaviors) == 500
    # The MML release contains a few typography-related spelling/spacing
    # differences. Its metadata is the authoritative title used by its judge.
    release_rows = {}
    for category in category_indices:
        cid = int(category.split("-", 1)[0])
        items = json.loads((DATA / "safebench" / category / "data.json").read_text())
        assert {int(item["id"]) for item in items} == set(range(1, 51))
        for item in items:
            release_rows[(cid, int(item["id"]))] = item
    title_differences = []
    for behavior in behaviors:
        released = release_rows[
            (behavior.meta["category_id"], behavior.meta["task_id"])
        ]["original_prompt"]
        if behavior.statement != released:
            title_differences.append(
                {
                    "behavior_id": behavior.behavior_id,
                    "figstep_statement": behavior.statement,
                    "mml_release_title": released,
                }
            )
        behavior.statement = released
        behavior.meta["statement_source"] = (
            "official MML data.json; category/task IDs matched"
        )
    (RUN / "release_title_differences.json").write_text(
        json.dumps(title_differences, indent=2)
    )
    rng = random.Random(20261022)
    chosen = []
    for cid in range(1, 11):
        group = [b for b in behaviors if b.meta["category_id"] == cid]
        chosen.extend(sorted(rng.sample(group, 5), key=lambda b: b.behavior_id))
    assert len(chosen) == 50
    assert set(Counter(b.category for b in chosen).values()) == {5}
    config = yaml.safe_load((ROOT / "configs/attacks/MML.yaml").read_text())
    config["variant"] = "mirror"
    attack = MMLAttack(config)
    downloads, records = [], {}
    for behavior in chosen:
        record = attack._find_precomputed_record(behavior)
        assert record is not None, behavior.behavior_id
        records[behavior.behavior_id] = record
        for method in ["images_mirror", "images_figstep"]:
            name = str(record["row"]["id"]) + ".png"
            category = record["subclass"]
            fid = image_indices[(category, method)][name]
            downloads.append((fid, DATA / "safebench" / category / method / name))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for i, result in enumerate(pool.map(download, downloads), 1):
            provenance.append(result)
            if i % 20 == 0:
                print(
                    json.dumps(
                        {"stage": "official_images", "completed": i, "total": 100}
                    ),
                    flush=True,
                )
    fs_prompt = load_upstream_constants(
        str(DATA / "upstream/const.py"), {"fs_prompt"}, expected_sha256=SOURCE_HASH
    )["fs_prompt"]
    cases, audit = [], []
    for behavior in chosen:
        record = records[behavior.behavior_id]
        filename = str(record["row"]["id"]) + ".png"
        fs_image = DATA / "safebench" / record["subclass"] / "images_figstep" / filename
        mirror_image = (
            DATA / "safebench" / record["subclass"] / "images_mirror" / filename
        )
        with Image.open(fs_image) as image:
            fs = image.convert("RGB")
        with Image.open(mirror_image) as image:
            mirror = image.convert("RGB")
        exact_mirror = (
            fs.size == mirror.size
            and ImageChops.difference(ImageOps.mirror(fs), mirror).getbbox() is None
        )
        audit.append(
            {
                "behavior_id": behavior.behavior_id,
                "exact_mirror_of_baseline": exact_mirror,
                "figstep_size": list(fs.size),
                "mirror_size": list(mirror.size),
            }
        )
        for case in attack.build(behavior, str(RUN / "images")):
            case.attack = "MML-M"
            case.test_case_id = f"MML-M:{behavior.behavior_id}:{case.attempt}"
            cases.append(case)
        for attempt in range(5):
            cases.append(
                TestCase(
                    f"FigStep:{behavior.behavior_id}:{attempt}",
                    behavior.behavior_id,
                    "FigStep",
                    fs_prompt,
                    [str(fs_image)],
                    attempt=attempt,
                    meta={
                        "original_title": record["row"]["original_prompt"],
                        "require_content_marker": False,
                    },
                )
            )
    assert len(cases) == len({c.test_case_id for c in cases}) == 500
    write_jsonl(str(RUN / "behaviors.jsonl"), chosen)
    write_jsonl(str(RUN / "test_cases.jsonl"), cases)
    (RUN / "asset_provenance.json").write_text(json.dumps(provenance, indent=2))
    (RUN / "image_audit.json").write_text(json.dumps(audit, indent=2))
    (RUN / "config.yaml").write_text(yaml.safe_dump(config))
    manifest = {
        "paper": "https://aclanthology.org/2025.acl-long.74.pdf",
        "paper_table": 1,
        "official_repository": "https://github.com/wangyu-ovo/MML",
        "official_commit": REV,
        "official_const_sha256": SOURCE_HASH,
        "official_drive_root": "10QlhyFvytmUsapcQf2xBBBsiB9N0zrpl",
        "dataset": "SafeBench",
        "n_behaviors": 50,
        "sampling_seed": 20261022,
        "sampling": "5 uniformly sampled without replacement from each of 10 categories; frozen before inference",
        "categories": dict(Counter(b.category for b in chosen)),
        "arms": ["FigStep", "MML-M"],
        "max_attempts": 5,
        "target_model": "gpt-4o-2024-08-06",
        "judge_model": "gpt-4o-mini-2024-07-18",
        "target_temperature": 0.7,
        "target_max_tokens": 2000,
        "system_message": None,
        "judge": "Official CLAS prompt + case-sensitive refusal gates; score 5 only is success",
        "content_marker_gate": "MML-M only; FigStep exempt as upstream",
        "stop_on_success": True,
        "image_input": "Official Drive images without regeneration",
        "paper_500_reference_asr": {"FigStep": 0.33, "MML-M": 0.976},
        "hint_randomness": "Unseeded like upstream; concrete shuffled prompts frozen in test_cases.jsonl",
        "exact_mirror_pairs": sum(a["exact_mirror_of_baseline"] for a in audit),
        "released_title_differences_from_figstep_csv": len(title_differences),
        "local_adapter_sha256": sha(ROOT / "multi_attacks/mml/mml.py"),
        "limitations": [
            "50-item subset rather than full 500: assess protocol and effect, not exact table decimals.",
            "Historical API backend state and nondeterministic responses cannot be recreated.",
        ],
    }
    existing.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "stage": "prepared",
                "behaviors": 50,
                "potential_calls": 500,
                "official_images": 100,
                "exact_mirror_pairs": manifest["exact_mirror_pairs"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
