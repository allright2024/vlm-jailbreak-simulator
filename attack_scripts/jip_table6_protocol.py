"""Model-free identity, coverage and sampling checks for the Table 6 experiment."""

from __future__ import annotations

import hashlib
import json
from collections import Counter

PAPER_CONFERENCE = "https://proceedings.iclr.cc/paper_files/paper/2024/file/83170ccee5543b872f4de71002f21aad-Paper-Conference.pdf"
SEED_STRATEGY = "sha256-test-case-v1"
SAMPLING_NOTE = (
    "25 repeats per image are specified for the main experiment in Section 4. "
    "Table 6 does not specify its denominator, number of images, or seeds. "
    "Two-decimal rates do not establish a denominator of 100. "
    "Repeats of one image per topic are an explicitly chosen replication protocol."
)


def case_seed(seed: int, test_case_id: str) -> int:
    """Same seed per case across arms, independent of resume order and batch size."""
    key = f"{SEED_STRATEGY}:{seed}:{test_case_id}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63 - 1)


def case_fingerprint(case: dict) -> str:
    payload = {key: case[key] for key in ("test_case_id", "behavior_id", "attempt", "text", "images", "meta")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def unique_records(records: list[dict], name: str) -> dict[str, dict]:
    counts = Counter(r["test_case_id"] for r in records)
    duplicates = [key for key, n in counts.items() if n != 1]
    if duplicates:
        raise ValueError(f"{name}: duplicate test_case_id(s): {duplicates[:3]}")
    return {r["test_case_id"]: r for r in records}


def validate_completions(cases: list[dict], records: list[dict], target: str) -> dict:
    expected = unique_records(cases, "test cases")
    actual = unique_records(records, "completions")
    for key, row in actual.items():
        if key not in expected:
            raise ValueError(f"completion absent from selected test cases: {key}")
        case = expected[key]
        for field in ("behavior_id", "attempt", "attack"):
            if row[field] != case[field]:
                raise ValueError(f"{key}: completion {field} differs from test case")
        if row["model"] != target:
            raise ValueError(f"{key}: wrong target {row['model']!r}")
    return {
        "expected": len(expected), "completed": len(actual),
        "missing_completions": len(expected.keys() - actual.keys()),
    }


def validate_results(cases: list[dict], completions: list[dict], results: list[dict], target: str) -> dict:
    coverage = validate_completions(cases, completions, target)
    validate_completions(cases, results, target)
    actual = unique_records(completions, "completions")
    scored = unique_records(results, "results")
    for key, row in scored.items():
        if key not in actual or row["response"] != actual[key]["response"]:
            raise ValueError(f"{key}: judged response differs from completion")
        if row.get("judge") != "HarmBench-Llama-2-13b-cls":
            raise ValueError(f"{key}: report expects HarmBench labels")
    coverage.update(
        result_rows=len(scored),
        missing_results=len(actual.keys() - scored.keys()),
        unjudged=sum(row.get("label") not in (0, 1) for row in scored.values()),
    )
    coverage["complete"] = (
        coverage["missing_completions"] == coverage["missing_results"] == coverage["unjudged"] == 0
    )
    return coverage


def validate_resume(cases: list[dict], records: list[dict], target: str,
                    seed: int, model_id: str, decoding: dict, extra_meta: dict | None = None) -> None:
    validate_completions(cases, records, target)
    by_id = unique_records(cases, "test cases")
    for row in records:
        meta = row.get("meta", {})
        expected = {"sampling_seed_strategy": SEED_STRATEGY,
                    "sampling_seed": case_seed(seed, row["test_case_id"]),
                    "model_id": model_id, "decoding": decoding,
                    "test_case_sha256": case_fingerprint(by_id[row["test_case_id"]]),
                    **(extra_meta or {})}
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ValueError(
                    f"Cannot resume {row['test_case_id']}: {key} differs or is missing. "
                    "Use a new output directory; legacy group-seeded results must not be mixed."
                )
