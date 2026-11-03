#!/usr/bin/env python3
"""Freeze official EchoSafe Gen data, assets, and the Table 3 comparison scope."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT.parent / "reference_repos/EchoSafe"
RUN = ROOT / "runs/echosafe_reproduction_20261103"
ASSETS = Path("/tmp/echosafe_reproduction_20261103")
DATA_REPO = "EchoSafe-MLLM/MM-SafetyBench-plus-plus"
DATA_REV = "128ea182089b677a8d48d97c9f86425713351ac2"
REF_REV = "f956a15976c3256c7b38a1f0ff48957c5579ec59"
SCENARIOS = ["01-Illegal_Activity", "02-HateSpeech", "03-Malware_Generation", "04-Physical_Harm", "06-Fraud", "07-Sex"]
MODELS = [
    ("Qwen/Qwen2.5-VL-7B-Instruct", "cc594898137f460bfe9f0759e9844b3ce807cfb5", "Qwen2.5-VL-7B-Instruct"),
    ("openai/clip-vit-large-patch14", "32bd64288804d66eefd0ccbe215aa642df71cc41", "clip-vit-large-patch14"),
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n")
    tmp.replace(path)


def get(url):
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response


def download(url, path, expected=None):
    path = Path(path)
    if path.exists() and (expected is None or sha(path) == expected):
        return sha(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        try:
            tmp = path.with_suffix(path.suffix + ".part")
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with tmp.open("wb") as f:
                    for block in response.iter_content(1 << 20):
                        f.write(block)
            found = sha(tmp)
            assert expected is None or found == expected, str(path)
            tmp.replace(path)
            return found
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def link(path, target):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        assert path.resolve() == Path(target).resolve()
    elif path.exists():
        raise FileExistsError(path)
    else:
        path.symlink_to(target, target_is_directory=Path(target).is_dir())


def main():
    RUN.mkdir(exist_ok=True)
    ASSETS.mkdir(exist_ok=True)
    assert subprocess.check_output(["git", "-C", str(REF), "rev-parse", "HEAD"], text=True).strip() == REF_REV
    assert not subprocess.check_output(["git", "-C", str(REF), "diff", "--name-only"], text=True).strip()
    cases = []
    for scenario in SCENARIOS:
        unsafe = json.loads((REF / "processed_questions" / (scenario + ".json")).read_text())
        safe = json.loads((REF / "processed_questions_safe" / (scenario + ".json")).read_text())
        assert list(unsafe) == list(safe)
        for key in unsafe:
            for flag, data in [("unsafe", unsafe), ("safe", safe)]:
                image = f"images/{scenario}/{'Gen' if flag == 'unsafe' else 'Gen_SAFE'}/{key}.png"
                cases.append({"case_id": f"{scenario}/{key}/{flag}", "scenario": scenario,
                              "image": image, "image_name": key + ".png", "safe_flag": flag,
                              "query": data[key]["Rephrased Question(SD)"]})
    assert len(cases) == 1422
    write(RUN / "cases.json", cases)
    wanted = {c["image"] for c in cases}
    entries = []
    url = f"https://huggingface.co/api/datasets/{DATA_REPO}/tree/{DATA_REV}?recursive=true&expand=false"
    while url:
        response = get(url)
        entries.extend(x for x in response.json() if x.get("path") in wanted)
        url = response.links.get("next", {}).get("url")
    assert {x["path"] for x in entries} == wanted
    write(RUN / "dataset_files.json", entries)
    print(json.dumps({"stage": "data", "files": len(entries), "bytes": sum(x["size"] for x in entries)}), flush=True)

    def one(entry):
        path = ASSETS / entry["path"]
        expected = entry["lfs"]["oid"]
        found = download(f"https://huggingface.co/datasets/{DATA_REPO}/resolve/{DATA_REV}/{entry['path']}", path, expected)
        from PIL import Image
        with Image.open(path) as image:
            image.verify()
        return {"path": str(path), "relative_path": entry["path"], "sha256": found, "bytes": entry["size"]}

    images = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        for future in concurrent.futures.as_completed([pool.submit(one, e) for e in entries]):
            images.append(future.result())
            if len(images) % 50 == 0 or len(images) == len(entries):
                value = {"stage": "data", "completed": len(images), "total": len(entries)}
                write(RUN / "prepare_status.json", value)
                print(json.dumps(value), flush=True)
    write(RUN / "image_provenance.json", sorted(images, key=lambda x: x["relative_path"]))

    models = []
    for repo, rev, dirname in MODELS:
        info = get(f"https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true").json()
        cache = Path('/home/jovyan/.cache/huggingface/hub') / ('models--' + repo.replace('/', '--')) / 'snapshots' / rev
        dest = ASSETS / "models" / dirname
        dest.mkdir(parents=True, exist_ok=True)
        records = []
        for f in info["siblings"]:
            name = f["rfilename"]
            if name.startswith('.') or name.endswith(('.md', '.h5', '.msgpack')):
                continue
            if name.endswith('.bin') and (cache / 'model.safetensors').exists():
                continue
            if name.endswith('.safetensors') and not (cache / name).exists() and (cache / 'pytorch_model.bin').exists():
                continue
            target = dest / name
            if "lfs" in f and (cache / name).exists():
                expected = f["lfs"]["sha256"]
                found = sha(cache / name)
                assert found == expected, name
                link(target, (cache / name).resolve())
            else:
                found = download(f"https://huggingface.co/{repo}/resolve/{rev}/{name}", target, f.get('lfs', {}).get('sha256'))
                if 'lfs' not in f:
                    payload = target.read_bytes()
                    blob = hashlib.sha1(b'blob ' + str(len(payload)).encode() + b'\0' + payload).hexdigest()
                    assert blob == f['blobId'], name
            records.append({"name": name, "sha256": found, "size": target.stat().st_size})
        models.append({"repo": repo, "revision": rev, "path": str(dest), "files": records})
        print(json.dumps({"stage": "model", "repo": repo, "verified_files": len(records)}), flush=True)
    write(RUN / "model_assets.json", models)
    workspace = RUN / "workspace"
    workspace.mkdir(exist_ok=True)
    link(RUN / "models", ASSETS / "models")
    link(workspace / "images", ASSETS / "images")
    for name in ["processed_questions", "processed_questions_safe"]:
        link(workspace / name, REF / name)
    html = get('https://arxiv.org/html/2603.15800v1').text
    (RUN / 'paper.html').write_text(html)
    from bs4 import BeautifulSoup
    table = BeautifulSoup(html, 'html.parser').find(id='S5.T3')
    (RUN / 'paper_table3.txt').write_text(table.get_text(' ', strip=True) + '\n')
    write(RUN / "prepare_complete.json", {"passed": True, "images": len(images), "model_files_verified": True,
          "dataset_revision": DATA_REV, "reference_commit": REF_REV})
    print('PREPARATION COMPLETE', flush=True)


if __name__ == '__main__':
    main()
