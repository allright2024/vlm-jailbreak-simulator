"""Prepare the pinned Unicode font used by five-method benchmark compositions."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

from core.paths import resolve_path

REVISION = "f8d157532fbfaeda587e826d4cd5b21a49186f7c"
BASE_URL = f"https://raw.githubusercontent.com/notofonts/noto-cjk/{REVISION}/Sans/"
FILES = {
    "NotoSansCJKsc-Bold.otf": (
        "OTF/SimplifiedChinese/NotoSansCJKsc-Bold.otf",
        "b5f0d1a190a7f9b43c310a8850630af12553df32c4c050543f9059732d9b4c0a"),
    "NotoSansCJK-OFL.txt": (
        "LICENSE",
        "6a73f9541c2de74158c0e7cf6b0a58ef774f5a780bf191f2d7ec9cc53efe2bf2"),
}


def _sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def prepare():
    folder = Path(resolve_path("data/method_assets/fonts"))
    folder.mkdir(parents=True, exist_ok=True)
    for name, (relative, expected) in FILES.items():
        destination = folder / name
        if destination.exists():
            if _sha256(destination) != expected:
                raise ValueError(f"existing font asset has a different hash: {destination}")
            continue
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=folder, delete=False) as stream:
                temporary = Path(stream.name)
                with urllib.request.urlopen(BASE_URL + relative, timeout=120) as response:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            if _sha256(temporary) != expected:
                raise ValueError(f"downloaded font asset has a different hash: {name}")
            os.replace(temporary, destination)
        finally:
            if temporary and temporary.exists():
                temporary.unlink()
    manifest = {"repository": "https://github.com/notofonts/noto-cjk", "revision": REVISION,
                "files": {name: {"url": BASE_URL + rel, "sha256": sha}
                          for name, (rel, sha) in FILES.items()}}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(folder / "NotoSansCJKsc-Bold.otf")
    return folder


if __name__ == "__main__":
    prepare()

