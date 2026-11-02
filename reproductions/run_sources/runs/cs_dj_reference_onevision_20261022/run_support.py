"""Persistence hooks only; attack construction remains in the official scripts."""
import hashlib
import json
import os
import random
import time
from pathlib import Path

RUN = Path(__file__).resolve().parent


def append(name, value):
    with (RUN / name).open("a") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_rows(name):
    p = RUN / name
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def seed_case(category, index):
    import torch
    seed = int.from_bytes(hashlib.sha256(f"2025:{category}:{index}".encode()).digest()[:8], "big") % (2**63 - 1)
    torch.manual_seed(seed)
    random.seed(seed)
    return seed


def is_done(category, index):
    return any(x["custom_id"] == f"{category}-{index:03d}" for x in read_rows("test_cases.jsonl"))


def record_attempt(category, index, attempt, text, subquestions):
    append("decomposition_attempts.jsonl", {"custom_id": f"{category}-{index:03d}",
           "attempt": attempt, "text": text, "subquestions": subquestions})


def record_sample(category, index, question, prompt, image_path, subquestions, attempts):
    from PIL import Image
    p = Path(image_path).resolve()
    assert len(subquestions) == 3
    assert 1 <= attempts <= 5
    assert not any("First sub-question" in q for q in subquestions)
    with Image.open(p) as im:
        assert im.size == (1500, 2120)
        im.verify()
    record = {"custom_id": f"{category}-{index:03d}", "category": category,
              "instruction": question, "prompt": prompt, "image": str(p),
              "image_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
              "subquestions": subquestions, "decomposition_attempts": attempts}
    append("test_cases.jsonl", record)
    update_progress()


def record_failure(category, index, question, prompt, attempts):
    assert attempts == 5
    assert not is_done(category, index)
    append("test_cases.jsonl", {"custom_id": f"{category}-{index:03d}", "category": category,
           "instruction": question, "prompt": prompt, "image": None, "image_sha256": None,
           "subquestions": [], "decomposition_attempts": attempts, "construction_failed": True,
           "failure_reason": "No valid three-question decomposition within five attempts"})
    update_progress()


def update_progress():
    records = read_rows("test_cases.jsonl")
    count = len(records)
    failed = sum(x.get("construction_failed", False) for x in records)
    tmp = RUN / "status.json.tmp"
    tmp.write_text(json.dumps({"phase": "official_sample_generation", "completed": count,
                              "total": 750, "images_generated": count - failed,
                              "construction_failed": failed, "updated_at_epoch": time.time()}, indent=2))
    tmp.replace(RUN / "status.json")
    print(f"SAMPLES {count}/750 images={count-failed} construction_failed={failed}", flush=True)
