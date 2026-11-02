"""Evaluate the completed SafeBench responses with the previously validated judge."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

RUN = Path(__file__).resolve().parent
PARENT = RUN.parent
spec = importlib.util.spec_from_file_location("released_ultrabreak_run", PARENT / "paper_run.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def write_once(path, data):
    if path.exists():
        assert path.read_bytes() == data, f"Existing file differs: {path}"
    else:
        path.write_bytes(data)


def json_once(path, value):
    write_once(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())


def prepare():
    base.verify()
    cases = [x for x in json.loads((PARENT / "test_cases.json").read_text()) if x["dataset"] == "safebench"]
    ids = [x["case_id"] for x in cases]
    assert len(ids) == len(set(ids)) == 315
    json_once(RUN / "test_cases.json", cases)
    selection = {}
    for key in base.MODELS:
        source = PARENT / f"responses_{key}.jsonl"
        rows = base.read_rows(source)
        selected = [x for x in rows if x["dataset"] == "safebench"]
        assert [x["case_id"] for x in selected] == ids
        assert all(x["model"] == key and x["assistant_response"].strip() for x in selected)
        # Preserve the exact original JSONL lines; no regeneration or answer editing.
        data = b"".join(line for line in source.read_bytes().splitlines(keepends=True)
                        if json.loads(line)["dataset"] == "safebench")
        write_once(RUN / source.name, data)
        write_once(RUN / f"runtime_{key}.json", (PARENT / f"runtime_{key}.json").read_bytes())
        selection[key] = {"source": str(source), "source_sha256": base.sha(source),
                          "source_rows": len(rows), "selected_rows": 315,
                          "excluded_rows": len(rows) - 315,
                          "selected_sha256": base.sha(RUN / source.name)}
    for name in ["launch_manifest.json", "input_manifest.json"]:
        write_once(RUN / ("parent_" + name), (PARENT / name).read_bytes())
    write_once(RUN / "source_manifest.json", (PARENT / "source_manifest.json").read_bytes())
    json_once(RUN / "selection_manifest.json", selection)
    plan = json.loads((PARENT / "plan.json").read_text())
    plan.update({"scope": "SafeBench-only evaluation of existing released-image responses",
                 "datasets": {"safebench": plan["datasets"]["safebench"]},
                 "total_responses": 630, "max_judgments": 1260,
                 "parent_run": str(PARENT), "additional_inference": False,
                 "selection": "All 315 SafeBench cases per model; no filtering by response or judgment",
                 "limits": [x for x in plan["limits"] if not x.startswith(("Paper calls AdvBench", "Only the two declared"))]
                           + ["Only the SafeBench rows of Table 1 for the two declared Qwen models are evaluated"]})
    json_once(RUN / "plan.json", plan)
    inputs = [RUN / "test_cases.json", base.OFF / "outputs/ultrabreak.png"]
    inputs += [RUN / f"responses_{key}.jsonl" for key in base.MODELS]
    inputs += [PARENT / f"responses_{key}.jsonl" for key in base.MODELS]
    json_once(RUN / "input_manifest.json", {str(p): base.sha(p) for p in inputs})
    files = [Path(__file__).resolve(), RUN / "run_background.sh", PARENT / "paper_run.py"]
    files += [RUN / name for name in ["plan.json", "selection_manifest.json", "input_manifest.json",
                                      "source_manifest.json", "parent_launch_manifest.json", "parent_input_manifest.json"]]
    files += [RUN / f"runtime_{key}.json" for key in base.MODELS]
    json_once(RUN / "launch_manifest.json", {str(p): base.sha(p) for p in files})
    print(json.dumps({"prepared_responses": 630, "judgments": 1260, "selection": selection}), flush=True)


def summarize():
    base.verify()
    ids = {x["case_id"] for x in json.loads((RUN / "test_cases.json").read_text())}
    assert len(ids) == 315
    comparison = []
    for key, model in base.MODELS.items():
        for protocol in ["official_extraction", "assistant_tokens_only"]:
            rows = base.read_rows(RUN / f"judgments_{key}_{protocol}.jsonl")
            assert len(rows) == 315 and {x["case_id"] for x in rows} == ids
            assert all(x["dataset"] == "safebench" and x["model"] == key and x["protocol"] == protocol for x in rows)
            successes = sum(x["success"] for x in rows)
            asr = successes / 315 * 100
            comparison.append({"dataset": "safebench", "model": model["id"], "setting": model["setting"],
                               "protocol": protocol, "n": 315, "successes": successes, "asr_percent": asr,
                               "paper_asr_percent": model["paper"]["safebench"],
                               "gap_pp": asr - model["paper"]["safebench"]})
    base.dump(RUN / "summary.json", {"status": "complete", "optimization_performed": False,
                                    "comparison": comparison,
                                    "limitations": json.loads((RUN / "plan.json").read_text())["limits"]})
    with (RUN / "comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    base.emit(RUN, {"stage": "complete", "responses": 630, "judgments": 1260})
    base.dump(RUN / "output_manifest.json", {str(p.relative_to(RUN)): base.sha(p)
                                            for p in RUN.rglob("*") if p.is_file()
                                            and p.suffix in (".json", ".jsonl", ".csv")
                                            and p.name not in ("output_manifest.json", "stage.json")})
    print(json.dumps(comparison), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "judge", "summarize"])
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    else:
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        base.RUN = RUN
        base.dump(RUN / "stage.json", {"stage": args.stage, "started_at": datetime.now(timezone.utc).isoformat()})
        if args.stage == "judge":
            base.judge()
        else:
            summarize()
