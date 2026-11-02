#!/usr/bin/env python
"""Verify archived Table 6 artifacts and run CLIP parity / save-path diagnostics.

No target responses are printed. This measures implementation agreement and
embedding distances, not attack success. The original loop is extracted from
the pinned upstream AST, without running its unrelated imports or overwriting
its released outputs. Existing experiment artifacts are read only.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import sys
from pathlib import Path

SIM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIM_ROOT))

from attack_scripts.jip_table6_protocol import validate_results


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parity_steps", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=25)
    args = parser.parse_args()
    if args.parity_steps <= 0 or args.rounds <= 0:
        parser.error("parity_steps and rounds must be positive")
    args.source = args.source.resolve()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    report = {"scope": "archived artifact integrity, real CLIP loop parity and image round trip",
              "source": str(args.source), "arms": {}, "paper_reproduction": False}
    source_hashes = {}
    for arm in ("released", "regenerated"):
        folder = args.source / arm
        files = {key: folder / name for key, name in {
            "cases": "test_cases.jsonl", "completions": "completions_llava_v1_13b_336px.jsonl",
            "results": "results_llava_v1_13b_336px.jsonl"}.items()}
        data = {key: read_rows(path) for key, path in files.items()}
        selected = [c for c in data["cases"] if c["attempt"] < args.rounds]
        expected = {f"jip_table6:{topic}:p{index + 1:02d}": set(range(args.rounds))
                    for topic in ("drugs", "bombs") for index in range(12)
                    if topic != "drugs" or index not in (9, 10)}
        actual = {}
        for case in selected:
            actual.setdefault(case["behavior_id"], set()).add(case["attempt"])
        if actual != expected:
            raise ValueError(f"{arm}: test cases do not cover all 22 cells x {args.rounds} repeats")
        coverage = validate_results(selected, data["completions"], data["results"], "llava_v1_13b_336px")
        source_hashes.update({str(p): digest(p) for p in files.values()})
        coverage["prepared"] = len(data["cases"])
        report["arms"][arm] = coverage

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    import yaml

    from multi_attacks.jailbreak_in_pieces.jailbreak_in_pieces import (
        JailbreakInPiecesAttack,
        embedding_distance,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no GPU diagnostic claimed")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = yaml.safe_load((SIM_ROOT / "configs/attacks/JailbreakInPieces.yaml").read_text())
    config.update(num_epochs=args.parity_steps, device="cuda")
    attack = JailbreakInPiecesAttack(config)
    model, processor = attack._load_surrogate()
    target = SIM_ROOT / "data/JailbreakInPieces/meth_pill.png"
    base = attack._pixels(attack.base_image)
    with torch.no_grad():
        target_embedding = model.get_image_features(attack._pixels(str(target)))

    upstream = SIM_ROOT.parent / "reference_repos/Jailbreak-In-Pieces/src/adv_image.py"
    if digest(upstream) != "fe205f8dfe10c728d7962e73139ccf472778c749f1b1fe7f91c3c3a1972a6248":
        raise ValueError("upstream source differs from the reviewed pinned commit")
    module = ast.parse(upstream.read_text())
    main_fn = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    # Execute the upstream loss, optimizer and complete loop exactly as written.
    selected_nodes = []
    for node in main_fn.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("pdist", "optimizer", "loss_list"):
                selected_nodes.append(node)
        if isinstance(node, ast.For):
            selected_nodes.append(node)
    if len(selected_nodes) != 4:
        raise ValueError("unexpected upstream AST; verify source manually")
    env = {"torch": torch, "optim": torch.optim, "model": model,
           "random_img": base.detach().clone().requires_grad_(True),
           "img_emb": target_embedding, "num_epochs": args.parity_steps}
    code = compile(ast.Module(body=selected_nodes, type_ignores=[]), str(upstream), "exec")
    with contextlib.redirect_stdout(io.StringIO()):
        exec(code, env)  # noqa: S102 - hash-verified, pinned upstream numeric loop
    artifact = attack._optimize(str(target), str(args.out / "parity_images"))
    port_losses = json.loads((SIM_ROOT / artifact["loss_history"]).read_text())
    delta = np.asarray(env["loss_list"]) - np.asarray(port_losses)
    report["loop_parity"] = {"steps": args.parity_steps, "bit_identical_loss_history": bool(np.all(delta == 0)),
                             "max_abs_loss_difference": float(np.max(np.abs(delta))),
                             "upstream_loop_sha256": digest(upstream), "port_artifact": artifact}
    print(json.dumps(report["loop_parity"], ensure_ascii=False), flush=True)

    report["saved_image_distances"] = {}
    for arm in ("released", "regenerated"):
        manifest = json.loads((args.source / arm / "run_manifest.json").read_text())
        report["saved_image_distances"][arm] = {}
        for topic, target_name in (("drugs", "meth_pill.png"), ("bombs", "grenade_bomb.png")):
            spec = manifest["images"][topic]
            image = Path(spec["image"])
            image = image if image.is_absolute() else SIM_ROOT / image
            if digest(image) != spec["image_sha256"]:
                raise ValueError(f"image hash mismatch: {image}")
            row = {"image_sha256": digest(image)}
            for normalized in (False, True):
                processor.do_normalize = normalized
                with torch.no_grad():
                    emb = model.get_image_features(attack._pixels(str(image)))
                    tgt = model.get_image_features(attack._pixels(str(SIM_ROOT / "data/JailbreakInPieces" / target_name)))
                    row["normalized_l2" if normalized else "upstream_unnormalized_l2"] = float(embedding_distance(emb, tgt).mean())
            if arm == "regenerated":
                row["recorded_pre_save_tensor_l2"] = spec["optimization"]["post_update_loss"]
                row["recorded_saved_jpeg_l2"] = spec["optimization"]["saved_image_loss"]
                row["saved_l2_absolute_difference_from_archive"] = abs(
                    row["upstream_unnormalized_l2"] - row["recorded_saved_jpeg_l2"])
            report["saved_image_distances"][arm][topic] = row
    processor.do_normalize = False
    report["source_sha256"] = source_hashes
    report["source_artifacts_unchanged"] = all(digest(path) == value for path, value in source_hashes.items())
    report["runtime"] = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
                         "transformers": importlib.metadata.version("transformers"),
                         "clip_revision": attack.model_revision, "deterministic_algorithms": True}
    report["checks_passed"] = (report["loop_parity"]["bit_identical_loss_history"]
                               and report["source_artifacts_unchanged"]
                               and all(row["complete"] for row in report["arms"].values()))
    (args.out / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"checks_passed": report["checks_passed"],
                      "saved_image_distances": report["saved_image_distances"]}), flush=True)
    if not report["checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
