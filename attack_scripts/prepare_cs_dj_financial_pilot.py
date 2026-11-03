#!/usr/bin/env python3
"""Prepare pinned public assets for the CS-DJ Financial pilot.

This script is intentionally resumable.  Hugging Face downloads use their
content-addressed cache, ZIP extraction is marked complete only after every
member is extracted, and existing user files are never silently replaced with
different content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import random
import shutil
import subprocess
import sys
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "CS-DJ"
UPSTREAM_REPO = "https://github.com/TeamPigeonLab/CS-DJ.git"
UPSTREAM_COMMIT = "4577b05fe93e86030ba2e4c3339e3aa59cdbb5ff"
CC3M_REPO = "liuhaotian/LLaVA-CC3M-Pretrain-595K"
CC3M_REVISION = "814894e93db9e12a1dee78b9669e20e8606fd590"
CC3M_ARCHIVE_BYTES = 6_456_430_373
CC3M_ARCHIVE_SHA256 = "1919904167348370212c61ffee36544639c6efbc9b8109cb367bc16095a5b6d0"
CC3M_IMAGE_COUNT = 595_375
PILOT_SEED = 2025
PILOT_IMAGE_COUNT = 10_000
LLAVA_REPO = "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf"
LLAVA_REVISION = "fa83f9361d2d053d9cfa12f2ceea23425762a8d3"
BEAVER_REPO = "PKU-Alignment/beaver-dam-7b"
BEAVER_REVISION = "3a5b70888869f04336d0d8344185ed74268ef896"
QWEN_REPO = "Qwen/Qwen2.5-3B-Instruct"
QWEN_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
CLIP_REPO = "sentence-transformers/clip-ViT-L-14"
CLIP_REVISION = "1b4b2e899178706d1b9905460ac21de1e0ba86a5"
IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def run(command: list[str], **kwargs) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, **kwargs)


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def copy_verified(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(source) != sha256(destination):
            raise RuntimeError(
                f"Refusing to replace a different existing asset: {destination}"
            )
        return
    shutil.copy2(source, destination)


def prepare_upstream() -> dict:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    upstream = DATA_ROOT / "upstream"
    if not (upstream / ".git").is_dir():
        if upstream.exists():
            raise RuntimeError(f"Non-git path already exists: {upstream}")
        run(["git", "clone", UPSTREAM_REPO, str(upstream)])
    run(["git", "-C", str(upstream), "fetch", "origin", UPSTREAM_COMMIT, "--depth=1"])
    run(["git", "-C", str(upstream), "checkout", "--detach", UPSTREAM_COMMIT])
    actual = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != UPSTREAM_COMMIT:
        raise RuntimeError(f"Unexpected CS-DJ commit: {actual}")

    instructions = DATA_ROOT / "instructions"
    instructions.mkdir(exist_ok=True)
    instruction_hashes = {}
    for source in sorted((upstream / "instructions").glob("*.json")):
        destination = instructions / source.name
        copy_verified(source, destination)
        instruction_hashes[source.name] = sha256(destination)
    financial = json.loads((instructions / "Financial.json").read_text(encoding="utf-8"))
    if len(financial) != 150:
        raise RuntimeError(f"Expected 150 Financial queries, found {len(financial)}")

    font = DATA_ROOT / "Super Moods.ttf"
    copy_verified(upstream / "Super Moods.ttf", font)
    return {
        "repo": UPSTREAM_REPO,
        "commit": actual,
        "instruction_hashes": instruction_hashes,
        "financial_count": len(financial),
        "font_sha256": sha256(font),
    }


def direct_image_count(path: Path) -> int:
    return sum(
        1 for entry in os.scandir(path)
        if entry.is_file() and Path(entry.name).suffix.lower() in IMAGE_SUFFIXES
    )


def prepare_cc3m() -> dict:
    from huggingface_hub import hf_hub_download

    archive = Path(hf_hub_download(
        repo_id=CC3M_REPO,
        repo_type="dataset",
        filename="images.zip",
        revision=CC3M_REVISION,
    ))
    if archive.stat().st_size != CC3M_ARCHIVE_BYTES:
        raise RuntimeError(
            f"Unexpected CC3M archive size: {archive.stat().st_size}"
        )
    archive_hash = sha256(archive)
    if archive_hash != CC3M_ARCHIVE_SHA256:
        raise RuntimeError(f"Unexpected CC3M archive SHA-256: {archive_hash}")

    image_root = DATA_ROOT / "llava_images"
    marker = image_root / ".selection-complete.json"
    selection_path = DATA_ROOT / f"cc3m_selection_seed{PILOT_SEED}.json"
    if marker.is_file():
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        if metadata.get("archive_sha256") != archive_hash:
            raise RuntimeError("Existing CC3M selection came from a different archive")
        image_count = direct_image_count(image_root)
        if image_count != PILOT_IMAGE_COUNT:
            raise RuntimeError(
                f"Expected {PILOT_IMAGE_COUNT} selected images, found {image_count}"
            )
        full_image_count = int(metadata["full_image_count"])
        selection_sha = str(metadata["selection_sha256"])
    else:
        if image_root.exists():
            raise RuntimeError(
                f"Unverified llava_images path already exists; refusing to replace it: {image_root}"
            )
        staging = DATA_ROOT / ".llava_images_seed2025.partial"
        staging.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            image_members = []
            for member in bundle.infolist():
                pure = PurePosixPath(member.filename)
                if pure.is_absolute() or ".." in pure.parts:
                    raise RuntimeError(f"Unsafe ZIP member: {member.filename!r}")
                if Path(member.filename).suffix.lower() in IMAGE_SUFFIXES:
                    image_members.append(member)
            full_image_count = len(image_members)
            if full_image_count != CC3M_IMAGE_COUNT:
                raise RuntimeError(
                    f"Expected {CC3M_IMAGE_COUNT} CC3M images, found {full_image_count}"
                )
            basenames = [Path(member.filename).name for member in image_members]
            if len(basenames) != len(set(basenames)):
                raise RuntimeError("CC3M archive image basenames are not unique")
            image_members.sort(key=lambda member: Path(member.filename).name)
            random.Random(PILOT_SEED).shuffle(image_members)
            selected = image_members[:PILOT_IMAGE_COUNT]
            selected_names = [Path(member.filename).name for member in selected]
            selection_sha = hashlib.sha256(
                ("\n".join(selected_names) + "\n").encode("utf-8")
            ).hexdigest()
            for index, member in enumerate(selected, start=1):
                destination = staging / Path(member.filename).name
                with bundle.open(member) as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
                if index % 500 == 0 or index == len(selected):
                    print(f"Extracted selected image {index}/{len(selected)}", flush=True)
        if direct_image_count(staging) != PILOT_IMAGE_COUNT:
            raise RuntimeError("Selective CC3M extraction did not produce exactly 10,000 images")
        selection = {
            "dataset_repo": CC3M_REPO,
            "dataset_revision": CC3M_REVISION,
            "archive_sha256": archive_hash,
            "full_image_count": full_image_count,
            "selection_algorithm": "sort basename; random.Random(2025).shuffle; first 10000",
            "seed": PILOT_SEED,
            "selected_count": PILOT_IMAGE_COUNT,
            "selected_basenames": selected_names,
            "selection_sha256": selection_sha,
        }
        selection_path.write_text(
            json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (staging / marker.name).write_text(json.dumps({
            "archive_sha256": archive_hash,
            "full_image_count": full_image_count,
            "seed": PILOT_SEED,
            "selected_count": PILOT_IMAGE_COUNT,
            "selection_sha256": selection_sha,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        staging.replace(image_root)
        image_count = direct_image_count(image_root)
    return {
        "repo": CC3M_REPO,
        "revision": CC3M_REVISION,
        "archive_path": str(archive),
        "archive_sha256": archive_hash,
        "image_root": str(image_root),
        "image_count": image_count,
        "full_image_count": full_image_count,
        "selection_seed": PILOT_SEED,
        "selection_manifest": str(selection_path),
        "selection_sha256": selection_sha,
    }


def prepare_models() -> dict:
    from huggingface_hub import snapshot_download

    models = {}
    for role, repo_id, revision in (
        ("query_decomposer", QWEN_REPO, QWEN_REVISION),
        ("retrieval_encoder", CLIP_REPO, CLIP_REVISION),
        ("target", LLAVA_REPO, LLAVA_REVISION),
        ("judge", BEAVER_REPO, BEAVER_REVISION),
    ):
        print(f"Preparing {role}: {repo_id}@{revision}", flush=True)
        path = snapshot_download(repo_id=repo_id, revision=revision)
        models[role] = {"repo": repo_id, "revision": revision, "snapshot_path": path}
    return models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-cc3m", action="store_true")
    parser.add_argument("--skip-models", action="store_true")
    args = parser.parse_args()

    manifest = {"schema_version": 1, "upstream": prepare_upstream()}
    if not args.skip_cc3m:
        manifest["cc3m"] = prepare_cc3m()
    if not args.skip_models:
        manifest["models"] = prepare_models()
    destination = DATA_ROOT / "assets_manifest.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    print(f"Assets ready: {destination}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ASSET PREPARATION FAILED: {exc}", file=sys.stderr, flush=True)
        raise
