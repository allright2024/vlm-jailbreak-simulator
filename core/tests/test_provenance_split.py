"""Tests for the audit fixes: config-aware fidelity, artifact lineage, and
train-exclusion-before-subset (+ a manifest integration test).

Model-free. Run:  python -m core.tests.test_provenance_split
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import yaml

import core.attacks  # noqa: F401
import core.datasets  # noqa: F401
from core import REPO_ROOT, get_attack, get_dataset
from core.pipeline import select_eligible

PYTHON = sys.executable


def _official(name):
    return yaml.safe_load(open(f"configs/attacks/{name}_official.yaml"))


def _pr(attack, cfg):
    return get_attack(attack)(cfg).provenance()["paper_reproduction"]


def main() -> None:
    # ---- #1: COMPLETE signature (every result-affecting field), not a subset ----
    force = get_attack("FORCE")(_official("FORCE"))
    ultra = get_attack("UltraBreak")(_official("UltraBreak"))
    assert force.effective_config() == force.official_signature
    assert ultra.effective_config() == ultra.official_signature
    assert force.provenance()["paper_reproduction"] is True
    assert ultra.provenance()["paper_reproduction"] is True
    reduced_signature = dict(force.official_signature)
    reduced_signature.pop("seed")
    force.official_signature = reduced_signature
    assert force.profile_verified() is False  # exact dict equality, not subset matching
    # reviewer counterexamples: a single off-signature field must break it
    assert _pr("FORCE", {**_official("FORCE"), "lambda_1": 999}) is False
    assert _pr("FORCE", {**_official("FORCE"), "num_noise": 1}) is False
    assert _pr("FORCE", {**_official("FORCE"), "seed": 7}) is False
    assert _pr("UltraBreak", {**_official("UltraBreak"), "lr": 99}) is False
    assert _pr("UltraBreak", {**_official("UltraBreak"), "tv_weight": 0}) is False
    assert _pr("UltraBreak", {**_official("UltraBreak"), "train_rows": 1}) is False
    # quick profile and model-substituting attacks never reproduce
    assert _pr("FORCE", {}) is False
    assert _pr("JPS", {"experiment_profile": "official"}) is False

    # ---- #2: artifact lineage must be verified for a reused artifact ----
    with tempfile.TemporaryDirectory() as td:
        art = os.path.join(td, "universal_adv.png")
        from PIL import Image
        Image.new("RGB", (16, 16), (1, 2, 3)).save(art)

        opt = get_attack("UltraBreak")(_official("UltraBreak"))
        opt.write_lineage(art)  # as if just optimized by this config
        assert os.path.isfile(art + ".lineage.json")

        # reuse with the SAME config + artifact -> verified -> reproduction
        reuse = get_attack("UltraBreak")({**_official("UltraBreak"), "adv_image": art})
        reuse.verify_lineage(art)
        p = reuse.provenance()
        assert p["artifact_reused"] and p["artifact_lineage_verified"] and p["paper_reproduction"]

        # reuse with a DIFFERENT config (epochs) -> lineage mismatch -> no reproduction
        tamper = get_attack("UltraBreak")({**_official("UltraBreak"), "adv_image": art, "epochs": 1})
        tamper.verify_lineage(art)
        assert tamper._lineage_verified is False
        assert tamper.provenance()["paper_reproduction"] is False

        # reuse an artifact with NO sidecar -> cannot verify
        art2 = os.path.join(td, "mystery.png")
        Image.new("RGB", (16, 16), (9, 9, 9)).save(art2)
        nolin = get_attack("UltraBreak")({**_official("UltraBreak"), "adv_image": art2})
        nolin.verify_lineage(art2)
        assert nolin._lineage_verified is False
        assert nolin.provenance()["paper_reproduction"] is False
        # even without calling verify_lineage, config with adv_image is flagged reused
        assert get_attack("UltraBreak")({**_official("UltraBreak"),
                                         "adv_image": art2}).provenance()["artifact_reused"] is True

        # VisualAdv lineage covers optimization settings and input file bytes.
        visual_art = os.path.join(td, "visual.png")
        visual_base = os.path.join(td, "base.png")
        visual_targets = os.path.join(td, "targets.csv")
        Image.new("RGB", (8, 8), (4, 5, 6)).save(visual_art)
        Image.new("RGB", (8, 8), (7, 8, 9)).save(visual_base)
        with open(visual_targets, "w", encoding="utf-8") as f:
            f.write("target one\ntarget two\n")
        visual_cfg = {
            "surrogate": "LLaVA_v1_5", "image_path": visual_base,
            "target_path": visual_targets, "n_iters": 10,
        }
        visual_opt = get_attack("VisualAdv")(visual_cfg)
        visual_opt.write_lineage(visual_art)
        visual_changed = get_attack("VisualAdv")({**visual_cfg, "n_iters": 11})
        assert visual_changed.verify_lineage(visual_art) is False
        with open(visual_targets, "a", encoding="utf-8") as f:
            f.write("target three\n")
        assert get_attack("VisualAdv")(visual_cfg).verify_lineage(visual_art) is False

        # JPS's full resolved config and steer bytes are both part of lineage.
        jps_cfg = yaml.safe_load(open("configs/attacks/JPS.yaml"))
        jps_art = os.path.join(td, "jps_adv.png")
        jps_steer = os.path.join(td, "jps_adv_steer.txt")
        Image.new("RGB", (8, 8), (10, 11, 12)).save(jps_art)
        with open(jps_steer, "w", encoding="utf-8") as f:
            f.write("original steer")
        jps_opt = get_attack("JPS")(jps_cfg)
        jps_opt.write_lineage(jps_art)
        verified = subprocess.run(
            [PYTHON, "verify_artifact_lineage.py", "--attack", "JPS",
             "--config", "configs/attacks/JPS.yaml", "--artifact", jps_art],
            cwd=REPO_ROOT, capture_output=True, text=True)
        assert verified.returncode == 0, (verified.stdout, verified.stderr)
        assert get_attack("JPS")({**jps_cfg, "attack_steps": 1}).verify_lineage(jps_art) is False
        with open(jps_steer, "w", encoding="utf-8") as f:
            f.write("tampered steer")
        assert get_attack("JPS")(jps_cfg).verify_lineage(jps_art) is False

        # Restore the authenticated steer and exercise a real JPS CLI manifest.
        with open(jps_steer, "w", encoding="utf-8") as f:
            f.write("original steer")
        reuse_cfg = os.path.join(td, "jps_reuse.yaml")
        with open(reuse_cfg, "w", encoding="utf-8") as f:
            yaml.safe_dump({**jps_cfg, "adv_image": jps_art, "steer_path": jps_steer}, f)
        jps_run = os.path.join(td, "jps_run")
        env = {**os.environ, "PYTHONNOUSERSITE": "1"}
        jr = subprocess.run(
            [PYTHON, "run_generate_test_cases.py", "--attack", "JPS",
             "--dataset", "AdvBench", "--limit", "3", "--config", reuse_cfg,
             "--save_dir", jps_run],
            cwd=REPO_ROOT, env=env, capture_output=True, text=True)
        assert jr.returncode == 0, (jr.stdout, jr.stderr)
        jman = json.load(open(os.path.join(jps_run, "run_manifest.json")))
        assert jman["n_test_cases"] == 3
        assert jman["n_excluded_behaviors"] == 50
        assert len(jman["excluded_behavior_ids"]) == 50
        assert set(jman["excluded_behaviors"].values()) <= {
            "train_split", "train_content_dup"
        }
        assert jman["artifact_lineage_verified"] is True
        assert jman["artifact_lineage"]["auxiliary_files"]["steer_prompt"]
        assert jman["artifact_lineage"]["implementation_sha256"]
        assert jman["lineage_sidecar_sha256"]

    # ---- #3 + ordering: exclusion before subset, reasons recorded ----
    jps = get_attack("JPS")({})
    pool = get_dataset("AdvBench")().load()
    assert len(jps.training_behavior_ids()) == 50
    elig, excl = select_eligible(jps, pool, limit=8)
    train_ids = set(jps.training_behavior_ids())
    assert len(elig) == 8 and all(b.behavior_id not in train_ids for b in elig)
    assert len(excl) == 50 and set(excl.values()) <= {"train_split", "train_content_dup"}
    sb = get_dataset("SafeBench")().load()
    assert len(select_eligible(jps, sb, limit=3, per_category=2)[0]) == 3
    assert len(select_eligible(jps, sb, per_category=2)[0]) == 2 * len({b.category for b in sb})

    # ---- manifest integration: real run_manifest.json from the CLI (FigStep, no model) ----
    env = {**os.environ, "PYTHONNOUSERSITE": "1"}
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run(
            [PYTHON, "run_generate_test_cases.py", "--attack", "FigStep",
             "--dataset", "AdvBench", "--limit", "3", "--save_dir", td],
            cwd=REPO_ROOT, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        man = json.load(open(os.path.join(td, "run_manifest.json")))
        for k in ("algorithm_fidelity", "experiment_profile", "effective_config", "profile_verified",
                  "paper_reproduction", "artifact_lineage", "excluded_behaviors", "excluded_behavior_ids",
                  "config_sha256", "n_test_cases"):
            assert k in man, k
        assert man["n_test_cases"] == 3 and man["paper_reproduction"] is False

        # empty run must hard-fail (exit 2) and write nothing
        r2 = subprocess.run(
            [PYTHON, "run_generate_test_cases.py", "--attack", "FigStep",
             "--dataset", "AdvBench", "--categories", "NO_SUCH_CAT", "--save_dir", td + "_empty"],
            cwd=REPO_ROOT, env=env, capture_output=True, text=True)
        assert r2.returncode == 2, (r2.returncode, r2.stderr)
        for name in ("test_cases.jsonl", "behaviors.jsonl", "run_manifest.json"):
            assert not os.path.exists(os.path.join(td + "_empty", name))

    print("provenance/split OK: complete config-aware signature, artifact lineage "
          "verify/tamper/missing, exclusion-before-limit, manifest integration, hard-fail")


if __name__ == "__main__":
    main()
