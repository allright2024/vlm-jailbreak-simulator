"""Audit category-balanced MM-SafetyBench splits using existing data only.

This utility writes sample identifiers and hashes. It does not label examples,
load a VLM, train a boundary, generate responses, or optimize attacks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("SD", "SD_TYPO", "TYPO")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def question_digest(text: str) -> str:
    normalized = " ".join(text.casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def read_categories(root: Path) -> tuple[dict, dict]:
    categories, sources = {}, {}
    for source in sorted(root.glob("*/*.json")):
        if source.stem != source.parent.name:
            continue
        category = source.parent.name
        sources[category] = {"file": str(source.relative_to(root)), "sha256": digest(source)}
        data = json.loads(source.read_text())
        records = []
        for key in sorted(data, key=int):
            text = data[key]["Question"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Empty question: {category}/{key}")
            images = {}
            for variant in VARIANTS:
                path = source.parent / variant / f"{key}.jpg"
                images[variant] = {"file": str(path.relative_to(root)), "sha256": digest(path)}
            records.append({"id": f"{category}:{key}", "category": category,
                            "question_id": key, "question_sha256": question_digest(text),
                            "images": images})
        categories[category] = records
    if len(categories) != 13:
        raise ValueError(f"Expected 13 categories; found {len(categories)}")
    return categories, sources


def build_split(categories: dict, per_category: int, seed: int, target: str) -> dict:
    if per_category < 1 or target not in categories:
        raise ValueError("Positive sampling count and an existing evaluation category are required")
    training, remainder = [], []
    for category, records in sorted(categories.items()):
        if len(records) <= per_category:
            raise ValueError(f"No held-out samples would remain in {category}")
        shuffled = list(records)
        random.Random(f"{seed}:{category}").shuffle(shuffled)
        training.extend(shuffled[:per_category])
        remainder.extend(shuffled[per_category:])
    training.sort(key=lambda row: row["id"])
    train_ids = {row["id"] for row in training}
    train_questions = {row["question_sha256"] for row in training}
    train_images = {image["sha256"] for row in training for image in row["images"].values()}
    heldout, excluded = [], []
    for row in sorted(remainder, key=lambda row: row["id"]):
        reasons = []
        if row["question_sha256"] in train_questions:
            reasons.append("normalized_question_matches_training")
        if any(image["sha256"] in train_images for image in row["images"].values()):
            reasons.append("image_matches_training")
        if reasons:
            excluded.append({"id": row["id"], "reasons": reasons})
        else:
            heldout.append(row)
    target_rows = [row for row in heldout if row["category"] == target]
    if not target_rows:
        raise ValueError("No evaluation candidates remain after overlap exclusions")
    if train_ids & {row["id"] for row in heldout}:
        raise ValueError("Training and held-out sample IDs overlap")
    counts = {category: {"total": len(records), "training_candidates": per_category,
                         "heldout_candidates": sum(r["category"] == category for r in heldout),
                         "excluded_for_overlap": sum(r["id"].startswith(category + ":") for r in excluded)}
              for category, records in sorted(categories.items())}
    return {"category_counts": counts, "training_candidates": training,
            "heldout_candidates": heldout, "target_evaluation_candidates": target_rows,
            "excluded_for_overlap": excluded}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/MM-SafetyBench")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-category", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-category", default="01-Illegal_Activitiy")
    args = parser.parse_args()
    categories, sources = read_categories(args.data)
    split = build_split(categories, args.per_category, args.seed, args.eval_category)
    metadata = {"scope": "Dataset split audit only", "target_model": "Qwen2.5-VL-7B",
                "source_root": str(args.data.resolve()), "sources": sources,
                "seed": args.seed, "per_category": args.per_category,
                "target_category": args.eval_category,
                "script_sha256": digest(Path(__file__)),
                "grouping": "Each original question ID keeps SD, SD_TYPO, and TYPO together",
                "overlap_checks": ["sample_id", "normalized_question_sha256", "all_image_variant_sha256"],
                "limitation": "Semantic paraphrase duplication is not detected by exact hashes",
                "labels_created": False, "boundary_training_run": False,
                "model_inference_run": False, "attack_run": False, "measured_asr": None}
    payload = {"metadata": metadata, **split}
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "split_audit.json"
    serialized = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path.exists() and path.read_text() != serialized:
        raise ValueError("Existing split differs; choose a new output directory")
    path.write_text(serialized)
    lines = ["# MM-SafetyBench category-balanced split audit", "",
             "Existing data only. No labels, boundary training, target inference, or attacks were run.",
             "Measured ASR: unavailable.", "",
             (f"Seed: {args.seed}. Training candidates: {len(split['training_candidates'])} "
              f"({args.per_category} per category)."), "",
             "| Category | Total | Training candidates | Held-out candidates | Overlap exclusions |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for category, counts in split["category_counts"].items():
        lines.append(f"| {category} | {counts['total']} | {counts['training_candidates']} | "
                     f"{counts['heldout_candidates']} | {counts['excluded_for_overlap']} |")
    lines.extend(["", (f"Target evaluation candidates ({args.eval_category}): "
                       f"{len(split['target_evaluation_candidates'])}."), "",
                  ("Original question IDs are grouped across all three image variants. "
                   "Held-out candidates matching a training question or image hash are excluded. "
                   "Semantic paraphrases require an additional review."), "",
                  "This split is a locally chosen configuration, not an author-provided paper split.", ""])
    (args.out / "summary.md").write_text("\n".join(lines))
    print(json.dumps({"training_candidates": len(split["training_candidates"]),
                      "target_evaluation_candidates": len(split["target_evaluation_candidates"]),
                      "overlap_exclusions": len(split["excluded_for_overlap"]),
                      "audit": str(path)}, indent=2))


if __name__ == "__main__":
    main()
