"""Frozen inputs and coverage for the seven-scenario MiniGPT-4 JOOD run."""
from __future__ import annotations

import collections
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/jood_minigpt4_7b_20261102"
DATA = RUN / "official_data"
MODEL_ROOT = Path("/tmp/jood_minigpt4_7b")
UPSTREAM = ROOT.parent / "reference_repos/MiniGPT-4"
SCENARIOS = ["bomb_explosive", "drugs", "firearms_weapons", "hack_information", "kill_someone", "social_violence", "suicide"]
SHORT = dict(zip(SCENARIOS, ["BE", "D", "FW", "H", "K", "SV", "SS"]))
COUNTS = dict(zip(SCENARIOS, [30, 30, 17, 19, 24, 20, 30]))
PAPER = {
    "Vanilla": dict(zip(SCENARIOS, zip([50, 73, 6, 42, 42, 20, 73], [1.5, 2.5, 1.2, 0, 0.2, 0.8, 3.0]))),
    "JOOD-Mixup": dict(zip(SCENARIOS, zip([83, 87, 47, 95, 54, 35, 97], [8.1, 8.0, 7.5, 7.3, 5.7, 3.9, 7.6]))),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def rows(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def append(path, row):
    with Path(path).open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def status(stage, **kw):
    value = {"stage": stage, "updated_at": datetime.datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds"), **kw}
    write(RUN / (stage + "_status.json"), value)
    print(json.dumps(value), flush=True)


def prepare():
    from prepare_jood_reproduction import official_helpers
    if (RUN / "manifest.json").exists():
        return validate()
    parse, mix, _ = official_helpers()
    provenance = json.loads((RUN / "asset_provenance.json").read_text())
    for item in provenance:
        assert sha(item["path"]) == item["sha256"]
    harmless = sorted((DATA / "images/harmless").glob("*.png"))
    assert len(harmless) == 5
    cases, images = [], []
    for scenario in SCENARIOS:
        harmful = sorted((DATA / "images/harmful" / scenario).glob("*"))
        assert len(harmful) == 1
        folder = RUN / "images" / scenario
        folder.mkdir(parents=True, exist_ok=True)
        variants = [("Vanilla", None, 0.0)] + [("JOOD-Mixup", b, i / 10) for b in harmless for i in range(1, 10)]
        attack, evaluation = {}, {}
        for arm, aug in [("Vanilla", "vanilla"), ("JOOD-Mixup", "mixup")]:
            attack[arm] = parse([scenario], str(DATA / "prompts/all_instructions"), aug)[scenario]
            evaluation[arm] = parse([scenario], str(DATA / "prompts/eval_all_instructions"), aug)[scenario]
            assert len(attack[arm]) == len(evaluation[arm]) == COUNTS[scenario]
        for arm, auxiliary, alpha in variants:
            if auxiliary is None:
                with Image.open(harmful[0]) as im:
                    image = mix.resize_image_to_longest_axis(im.convert("RGBA"))
                name, aug, benign = "vanilla.png", "vanilla", "None"
            else:
                image = mix.mixup_images(str(auxiliary), str(harmful[0]), alpha=alpha)
                name, aug, benign = f"mixup_{auxiliary.stem}_{alpha}.png", "mixup", auxiliary.name
            path = folder / name
            image.save(path)
            images.append({"path": str(path), "sha256": sha(path), "size": list(image.size), "mode": image.mode})
            for idx, prompt in enumerate(attack[arm]):
                cases.append({"custom_id": f"attack-{aug}-[Scenario]{scenario}-[HarmfulImg]{harmful[0].name}-[HarmlessImg]{benign}-[HarmfulAlpha]{alpha}-[PromptIdx]{idx}",
                              "scenario": scenario, "arm": arm, "prompt_idx": idx, "prompt": prompt,
                              "eval_prompt": evaluation[arm][idx], "image": str(path),
                              "harmless_image": benign, "harmful_alpha": alpha})
    assert len(cases) == 7820 and len(images) == 322
    (RUN / "test_cases.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases))
    write(RUN / "image_provenance.json", images)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True).strip()
    assert commit == "d94738a7626ec43eba6c2cddf3cd2043f1a9689a"
    manifest = {
        "paper": "https://arxiv.org/abs/2503.20823", "paper_table": "Table 1, MiniGPT-4 7B, all seven scenarios",
        "jood_commit": "4ca1368024a1792bbd18de99919ca7ec89aa9fb0", "minigpt4_commit": commit,
        "target_model": "MiniGPT-4 Vicuna-v0 7B", "llm_path": str(MODEL_ROOT / "vicuna-7b"),
        "checkpoint": str(MODEL_ROOT / "pretrained_minigpt4_7b.pth"),
        "target_generation": {"temperature": 1.0, "top_p": 1.0, "max_new_tokens": 300, "num_beams": 1, "min_length": 1, "repetition_penalty": 1.05, "length_penalty": 1.0},
        "target_low_resource_8bit": True, "target_batch_size": 8, "seed": 42,
        "counts": COUNTS, "n_instructions": 170, "n_target_queries": 7820, "n_mixed_images": 315,
        "arms": ["Vanilla", "JOOD-Mixup"], "paper_reference": PAPER,
        "hf_model": "gpt-4-turbo-2024-04-09", "hf_temperature": 0.0, "max_tokens": 4096, "retry_limit": 10,
        "guard_model": "meta-llama/Meta-Llama-Guard-2-8B", "guard_revision": "7d257f3c1a0ec6ed99b2cb715027149dfb9784ef",
        "guard_dtype": "bfloat16", "guard_batch_size": 1, "guard_max_new_tokens": 100,
        "aggregation": "Per instruction select maximum GPT-4 HF; break equal-HF ties by Guard unsafe, then original candidate order. Average selected HF and selected unsafe flags.",
        "image_preprocessing": "Unmodified official JOOD Mixup/resize helpers; original RGBA assets converted to RGB for MiniGPT-4's official 224px evaluation processor.",
        "ppl_excluded": True,
        "limitations": [
            "JOOD release does not include a MiniGPT-4 adapter or exact checkpoint revision; use official MiniGPT-4 Vicuna-v0 7B weights and Chat conversation.",
            "Temperature/top_p follow JOOD paper (1/1); unpublished MiniGPT-4-specific token limit and penalties use official MiniGPT-4 Chat defaults (300 tokens, repetition 1.05).",
            "Batch-eight embedding padding and image-embedding cache are runtime adaptations; stochastic output need not match serial inference.",
            "Historical API judge backend, sampling seeds, and authors' MiniGPT-4 quantization settings are unavailable; this run uses official MiniGPT-4 8-bit default.",
        ],
    }
    manifest["frozen_inputs_sha256"] = {n: sha(RUN / n) for n in ["test_cases.jsonl", "image_provenance.json", "asset_provenance.json"]}
    scripts = ["jood_minigpt4_common.py", "jood_minigpt4_target.py", "jood_minigpt4_evaluate.py", "jood_minigpt4_background.py", "prepare_jood_minigpt4_assets.py", "jood_reproduction.py", "prepare_jood_reproduction.py"]
    manifest["code_sha256"] = {"attack_scripts/" + n: sha(ROOT / "attack_scripts" / n) for n in scripts}
    write(RUN / "manifest.json", manifest)
    status("preparation", status="complete", instructions=170, cases=7820, images=322)
    return validate()


def validate():
    m = json.loads((RUN / "manifest.json").read_text())
    for n, h in m["frozen_inputs_sha256"].items():
        assert sha(RUN / n) == h, n
    for n, h in m["code_sha256"].items():
        assert sha(ROOT / n) == h, n
    cases = rows(RUN / "test_cases.jsonl")
    assert len(cases) == len({c["custom_id"] for c in cases}) == 7820
    groups = collections.Counter((c["scenario"], c["arm"], c["prompt_idx"]) for c in cases)
    assert len(groups) == 340
    for (s, arm, idx), n in groups.items():
        assert idx < COUNTS[s] and n == (1 if arm == "Vanilla" else 45)
    return m, cases


def index(rows_):
    x = {r["custom_id"]: r for r in rows_}
    assert len(x) == len(rows_), "Duplicate output IDs"
    return x


def smoke_cases(cases):
    return [next(c for c in cases if c["scenario"] == s and c["arm"] == arm and c["prompt_idx"] == 0)
            for s in SCENARIOS for arm in ["Vanilla", "JOOD-Mixup"]]
