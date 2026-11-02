"""Compute a disjoint tail of Financial Advice inputs with the frozen boundary.

No responses are generated and no boundaries are fitted. The main runner
validates and reuses these standard optimization artifacts when it reaches them.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from attack_scripts.jailbound_financial_reproduction import (
    CATEGORY,
    DEPENDENCIES,
    ROOT,
    make_adapter,
)
from attack_scripts.jailbound_paper_protocol import sha
from attack_scripts.jailbound_table3_pilot import log, optimize, write
from multimodalmodels.loaders import load_torch_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--tail-count", type=int, default=50)
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    config = json.loads((args.run / "manifest.json").read_text())
    source = json.loads((args.run / "source_cases.json").read_text())
    integrity = json.loads((args.run / "boundary_integrity.json").read_text())
    if {name: sha(ROOT / name) for name in DEPENDENCIES} != config["implementation_sha256"]:
        raise ValueError("Main experiment implementation changed")
    if sha(args.run / "boundaries.pt") != integrity["sha256"]:
        raise ValueError("Frozen boundary changed")
    evaluation = source["evaluation"]
    if not 0 < args.tail_count < len(evaluation) or any(row["category"] != CATEGORY for row in evaluation):
        raise ValueError("Auxiliary work requires a proper Financial Advice tail")
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < 40 * 1024 ** 3:
        raise RuntimeError("Less than 40 GiB GPU memory is free; leave all optimization to the main runner")
    adapter = make_adapter(source["financial"], config)
    boundaries = load_torch_checkpoint(str(args.run / "boundaries.pt"), map_location="cpu", weights_only=True)
    positions = {row["id"]: index for index, row in enumerate(evaluation)}
    assigned = list(reversed(evaluation[-args.tail_count:]))
    state = {"script_sha256": sha(Path(__file__)), "main_implementation_sha256": config["implementation_sha256"],
             "boundary_sha256": integrity["sha256"], "assigned_ids_reverse_order": [row["id"] for row in assigned],
             "completed": [], "new_text_generations": 0, "new_boundary_fits": 0,
             "minimum_leading_sample_margin": 3, "status": "running"}
    for row in assigned:
        if sha(Path(row["image"])) != row["image_sha256"]:
            raise ValueError("Auxiliary input image changed")
        for mode in ("text", "image", "joint"):
            progress = json.loads((args.run / "progress.json").read_text())
            parent_position = positions.get(progress.get("last_id"), -1)
            if progress["phase"] != "optimization" or parent_position + 3 >= positions[row["id"]]:
                state["status"] = "stopped_before_main_work_front"
                write(args.run / "auxiliary_optimization.json", state)
                log("auxiliary optimization stopped before approaching main runner")
                return
            optimize(adapter, row, boundaries, mode, args.run)
            state["completed"].append({"id": row["id"], "mode": mode})
            write(args.run / "auxiliary_optimization.json", state)
            log(f"auxiliary completed {len(state['completed'])}/{3 * len(assigned)}")
    state["status"] = "completed"
    write(args.run / "auxiliary_optimization.json", state)
    log("auxiliary optimization completed")


if __name__ == "__main__":
    main()
