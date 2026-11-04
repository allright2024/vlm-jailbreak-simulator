"""Annotate completions for the twelve jailbreak/defense evaluation metrics."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import tempfile
from pathlib import Path

from core import Behavior, Completion, read_jsonl
from core.response_grading import ResponseGrader
from core.schema import EvalType


def load_cache(path: Path) -> dict:
    """Recover an interrupted final append, while rejecting interior corruption."""
    cache = {}
    if not path.exists():
        return cache
    lines = path.read_bytes().splitlines(keepends=True)
    offset = 0
    for i, line in enumerate(lines):
        try:
            item = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if i == len(lines) - 1 and not line.endswith(b"\n"):
                with path.open("r+b") as stream:
                    stream.truncate(offset)
                break
            raise ValueError("Corrupt grading cache; preserve it and use a new --cache path")
        key, annotation = item["key"], item["annotation"]
        if annotation.get("input_sha256") != key:
            raise ValueError("Cache key differs from the stored input fingerprint")
        if key in cache and cache[key] != annotation:
            raise ValueError("Conflicting cache entries for the same response")
        cache[key] = annotation
        offset += len(line)
    if path.stat().st_size and not path.read_bytes().endswith(b"\n"):
        with path.open("ab") as stream:
            stream.write(b"\n")
    return cache


def grade_file(args, *, grader=None) -> dict:
    source, output = Path(args.completions).resolve(), Path(args.output).resolve()
    cache_path = Path(args.cache).resolve() if args.cache else output.with_suffix(".cache.jsonl")
    protected = {source, Path(args.behaviors).resolve()}
    if output in protected or cache_path in protected or cache_path == output:
        raise ValueError("Output and cache must be separate from source/Behavior files and each other")
    rows = read_jsonl(str(source), Completion)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        rows = rows[:args.limit]
    behaviors = read_jsonl(args.behaviors, Behavior)
    lookup = {b.behavior_id: b for b in behaviors}
    if len(lookup) != len(behaviors):
        raise ValueError("Duplicate Behavior IDs")
    ids = set()
    for row in rows:
        if row.test_case_id in ids or not row.test_case_id:
            raise ValueError("Duplicate/empty test_case_id")
        ids.add(row.test_case_id)
        if (row.behavior_id not in lookup or not isinstance(lookup[row.behavior_id].instruction, str)
                or not lookup[row.behavior_id].instruction):
            raise ValueError(f"Missing original Behavior instruction: {row.behavior_id}")
        if not isinstance(row.response, str) or row.label not in (None, 0, 1):
            raise ValueError("Completion must have response text and a binary/null label")
        if type(row.blocked) is not bool or row.blocked and row.label == 1:
            raise ValueError("Invalid blocked completion")
        if args.kind == "jailbreak" and lookup[row.behavior_id].eval_type != EvalType.HARMFUL:
            raise ValueError("Jailbreak grading requires harmful-instruction Behaviors")
    if len({row.model for row in rows}) > 1:
        raise ValueError("Grade each target model separately")

    grader = grader or ResponseGrader(args.judge_model, retry_limit=args.retry_limit, timeout=args.timeout)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    reused, fresh = 0, 0
    # A cache is shared by at most one running grader. Reuse is keyed by the
    # original instruction, response, role, target, references and judge settings.
    with Path(str(output) + ".lock").open("a") as output_lock, Path(str(cache_path) + ".lock").open("a") as lock:
        fcntl.flock(output_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        cache = load_cache(cache_path)
        fd, temporary = tempfile.mkstemp(prefix=output.name + ".", dir=output.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as destination, cache_path.open("a", encoding="utf-8") as checkpoint:
                for index, row in enumerate(rows):
                    behavior = lookup[row.behavior_id]
                    key = grader.fingerprint(row, behavior, args.kind)
                    if key in cache:
                        annotation = cache[key]
                        reused += 1
                    else:
                        annotation = grader.grade(row, behavior, args.kind)
                        # Validate before persisting a cache entry.
                        grader.attach(row, behavior, args.kind, annotation)
                        checkpoint.write(json.dumps({"key": key, "annotation": annotation}, ensure_ascii=False, allow_nan=False) + "\n")
                        checkpoint.flush()
                        os.fsync(checkpoint.fileno())
                        cache[key] = annotation
                        fresh += 1
                    graded = grader.attach(row, behavior, args.kind, annotation)
                    destination.write(json.dumps(graded.to_dict(), ensure_ascii=False, allow_nan=False) + "\n")
                    if (index + 1) % 20 == 0:
                        print(json.dumps({"graded": index + 1, "new": fresh, "reused": reused}), flush=True)
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary, output)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return {"output": str(output), "cache": str(cache_path), "n_records": len(rows),
            "n_new_annotations": fresh, "n_cached_annotations": reused, "kind": args.kind}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completions", required=True, help="Raw or already judged Completion JSONL.")
    parser.add_argument("--behaviors", required=True, help="Original instructions; never attack-mutated prompts.")
    parser.add_argument("--kind", choices=("jailbreak", "benign"), default="jailbreak")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--output", required=True, help="New annotated Completion JSONL.")
    parser.add_argument("--cache", help="Resumable annotation cache; default beside --output.")
    parser.add_argument("--limit", type=int, help="Optional number of completions to grade.")
    parser.add_argument("--retry-limit", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).with_name(".env"), override=False)
    print(json.dumps(grade_file(args), ensure_ascii=False))


if __name__ == "__main__":
    main()

