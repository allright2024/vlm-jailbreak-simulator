#!/usr/bin/env python3
"""Audit and reproduce the visual-image rows of Qi et al. (AAAI 2024).

Stages: plan, fetch-data, audit, optimize, generate, score, report,
all. The controller is model-free; original models run in separate subprocesses.
See image_attacks/visual_adv/REPRODUCTION.md before a full run.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from attack_scripts.visualadv_protocol import (  # noqa: E402
    ATTRIBUTES, BUDGETS, EXPECTED_RTP, MANUAL_REPEATS, MODELS, PAPER_URL,
    REFERENCE_FILE, RTP_REPEATS, UPSTREAM_COMMIT, VARIANTS, digest,
    expected_cases, paper_protocol, read_jsonl, read_rtp, sha256,
    score_flags, summarize_scores, unique, validate_responses, write_json,
)


def log(message):
    stamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(timespec="seconds")
    print(f"[{stamp}] {message}", flush=True)


def resolve(value):
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(path):
    import yaml
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg["upstream"] = str(resolve(cfg["upstream"]))
    cfg["rtp_path"] = str(resolve(cfg["rtp_path"]))
    for spec in cfg["models"].values():
        for key in ("checkpoint", "llm_path", "python"):
            if key in spec:
                # Preserve a venv's executable path: resolving its symlink runs
                # the base interpreter and silently loses the isolated packages.
                spec[key] = str(Path(spec[key]).expanduser().absolute()) if key == "python" else str(resolve(spec[key]))
    cfg["scoring"]["python"] = str(Path(cfg["scoring"].get("python", sys.executable)).expanduser().absolute())
    if cfg.get("manual_categories"):
        cfg["manual_categories"] = str(resolve(cfg["manual_categories"]))
    if not cfg.get("sampling_seeds") or len(set(cfg["sampling_seeds"])) != len(cfg["sampling_seeds"]):
        raise ValueError("sampling_seeds must be nonempty and distinct")
    return cfg


@contextlib.contextmanager
def output_lock(out):
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".worker.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"Another VisualAdv worker is writing {out}") from error
        yield


def verify_upstream(cfg):
    upstream = Path(cfg["upstream"])
    head = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=no"], text=True)
    if head != UPSTREAM_COMMIT or dirty.strip():
        raise ValueError("Official checkout must be clean at " + UPSTREAM_COMMIT)
    return upstream


def asset_files(path):
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix in
                  {".json", ".bin", ".safetensors", ".pth", ".model"})


def missing_asset(path):
    path = Path(path)
    files = asset_files(path)
    weights = [p for p in files if p.suffix in {".bin", ".safetensors", ".pth"} and p.stat().st_size > 1_000_000]
    if not weights:
        return f"weights missing/incomplete: {path}"
    for index in path.glob("*.index.json") if path.is_dir() else []:
        mapping = json.loads(index.read_text()).get("weight_map", {})
        if any(not (path / p).is_file() for p in set(mapping.values())):
            return f"checkpoint shards missing: {path}"
    return None


def checkpoint_fingerprint(spec):
    result = {}
    for key in ("checkpoint", "llm_path"):
        if key not in spec:
            continue
        missing = missing_asset(spec[key])
        if missing:
            raise ValueError(missing)
        log(f"Hashing original {key}: {spec[key]}")
        result[key] = {str(p): sha256(p) for p in asset_files(spec[key])}
    import torch
    cache = Path(torch.hub.get_dir()) / "checkpoints"
    auxiliary = [cache / "eva_vit_g.pth", cache / "blip2_pretrained_flant5xxl.pth"]
    result["vision_and_qformer"] = {str(path): sha256(path) for path in auxiliary}
    return result


def source_fingerprint():
    paths = [Path(__file__), ROOT / "attack_scripts/visualadv_protocol.py",
             ROOT / "image_attacks/visual_adv/official.py", ROOT / "image_attacks/visual_adv/pgd.py",
             ROOT / "image_attacks/visual_adv/batching.py"]
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def runtime_fingerprint():
    import importlib.metadata
    import torch
    packages = ("torch", "torchvision", "transformers", "accelerate", "timm",
                "omegaconf", "sentencepiece", "bitsandbytes")
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"packages": versions, "python": sys.version, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def bind_manifest(path, value):
    path = Path(path)
    value = json.loads(json.dumps(value, allow_nan=False))
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"{path}: configuration/input/code changed; choose a new output directory")
    else:
        write_json(path, value)


def suites(args):
    return ("rtp",)


def cases_for(args, cfg, suite):
    cases = read_rtp(cfg["rtp_path"], expected=cfg["rtp_expected_prompts"])
    if args.limit:
        cases = cases[:args.limit]
    return expected_cases(cases, "rtp", len(cfg["sampling_seeds"]))


def image_location(args, cfg, source, variant):
    upstream = Path(cfg["upstream"])
    if variant == "clean":
        return upstream / "adversarial_images/clean.jpeg"
    if args.image_source == "released":
        if source != "minigpt4":
            raise ValueError("The release only supplies MiniGPT-4 images; train other surrogates")
        suffix = "unconstrained" if variant == "unconstrained" else "constrained_" + variant[3:]
        return upstream / f"adversarial_images/prompt_{suffix}.bmp"
    return args.out / "images" / source / variant / "visual_adv.png"


def cells(args):
    for target in args.models:
        if "clean" in args.variants:
            yield "clean", "clean", target
        for source in args.surrogates:
            for variant in args.variants:
                if variant != "clean":
                    yield source, variant, target


def cell_path(args, suite, source, variant, target):
    return args.out / "cells" / suite / source / variant / target


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def stage_plan(args, cfg):
    protocol = paper_protocol()
    protocol["requested"] = {"models": args.models, "surrogates": args.surrogates,
                             "variants": args.variants, "suite": args.suite,
                             "image_source": args.image_source, "limit": args.limit,
                             "steps": args.steps or cfg["steps"],
                             "iteration_convention": cfg["iteration_convention"],
                             "std_ddof": cfg["std_ddof"], "sampling_seeds": cfg["sampling_seeds"]}
    protocol["paper_reproduction"] = False
    protocol["note"] = "Plan only. Completion/coverage and original model identity must be checked against actual artifacts."
    write_json(args.out / "protocol.json", protocol)
    reference = json.loads(REFERENCE_FILE.read_text())
    write_json(args.out / "paper_reference.json", reference)
    log(f"Protocol and paper references -> {args.out}")


def stage_fetch_data(args, cfg):
    path = Path(cfg["rtp_path"])
    if path.exists():
        read_rtp(path, expected=cfg["rtp_expected_prompts"])
        log(f"RTP already present: {cfg['rtp_expected_prompts']} challenging prompts")
        return
    revision = cfg["rtp_revision"]
    url = f"https://huggingface.co/datasets/allenai/real-toxicity-prompts/resolve/{revision}/prompts.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".download")
    log("Downloading original public RealToxicityPrompts (67.7 MB)")
    if not temporary.exists():
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream)
    read_rtp(temporary, expected=cfg["rtp_expected_prompts"])
    temporary.replace(path)
    write_json(path.with_suffix(".source.json"), {"url": url, "sha256": sha256(path),
                                                "challenging_prompts": cfg["rtp_expected_prompts"],
                                                "paper_challenging_prompts": EXPECTED_RTP})


def stage_audit(args, cfg):
    failures = []
    try:
        upstream = verify_upstream(cfg)
    except (ValueError, subprocess.CalledProcessError, FileNotFoundError) as error:
        failures.append(str(error))
        upstream = Path(cfg["upstream"])
    if "rtp" in suites(args):
        try:
            read_rtp(cfg["rtp_path"], expected=cfg["rtp_expected_prompts"])
        except (ValueError, FileNotFoundError) as error:
            failures.append(str(error))
    model_status = {}
    for name in sorted(set(args.models) | (set(args.surrogates) if args.image_source == "trained" else set())):
        spec = cfg["models"][name]
        errors = [error for key in ("checkpoint", "llm_path") if key in spec
                  for error in [missing_asset(spec[key])] if error]
        needed = ["torch", "torchvision", "transformers", "omegaconf", "timm", "einops", "sentencepiece"]
        if name == "minigpt4" and spec.get("low_resource", True):
            needed.append("bitsandbytes")
        code = ("import importlib.util,json; names=" + repr(needed) + "; "
                "print(json.dumps([n for n in names if importlib.util.find_spec(n) is None]))")
        try:
            run = subprocess.run([spec["python"], "-c", code], capture_output=True, text=True, timeout=30,
                                 env={**os.environ, "PYTHONNOUSERSITE": "1"})
            if run.returncode:
                errors.append("runtime probe failed: " + run.stderr[-300:])
            else:
                missing = json.loads(run.stdout.strip().splitlines()[-1])
                if missing:
                    errors.append("missing modules: " + ", ".join(missing))
        except (OSError, subprocess.TimeoutExpired, ValueError) as error:
            errors.append("runtime unavailable: " + str(error))
        model_status[name] = {"ready": not errors, "errors": errors, "identity": spec["identity"]}
        failures.extend(f"{name}: {error}" for error in errors)
    scoring = cfg["scoring"]
    if "rtp" in suites(args):
        if "perspective_api" in scoring["scorers"] and not os.environ.get(scoring["perspective_key_env"]):
            failures.append("Perspective API key absent from " + scoring["perspective_key_env"])
    audit = {"ready": not failures, "models": model_status, "failures": failures,
             "paper_reproduction": False, "historical_checkpoint_identity_verified": False,
             "limitations": paper_protocol()["limitations"]}
    write_json(args.out / "audit.json", audit)
    log(f"Readiness: {'ready' if audit['ready'] else 'missing prerequisites'} ({len(failures)} issues)")
    for error in failures:
        log(error)
    return audit


def optimize_manifest(args, cfg, name, variant, fingerprints):
    upstream = Path(cfg["upstream"])
    steps = args.steps or cfg["steps"]
    if cfg["iteration_convention"] not in ("paper", "released"):
        raise ValueError("iteration_convention must be paper or released")
    return {"commit": UPSTREAM_COMMIT, "model": name, "checkpoints": fingerprints,
            "low_resource": cfg["models"][name].get("low_resource", True),
            "runtime": runtime_fingerprint(),
            "variant": variant, "updates": steps + int(cfg["iteration_convention"] == "released"),
            "iteration_convention": cfg["iteration_convention"], "alpha_255": 1, "batch_size": 8,
            "seed": cfg["optimization_seed"], "code": source_fingerprint(),
            "corpus_sha256": sha256(upstream / "harmful_corpus/derogatory_corpus.csv"),
            "clean_sha256": sha256(upstream / "adversarial_images/clean.jpeg")}


def worker_optimize(args, cfg):
    import torch
    from torchvision.utils import save_image
    from image_attacks.visual_adv.official import OfficialModel
    from image_attacks.visual_adv.pgd import optimize_pixels

    upstream = verify_upstream(cfg)
    name = args.worker_model
    spec = cfg["models"][name]
    fingerprints = checkpoint_fingerprint(spec)
    with (upstream / "harmful_corpus/derogatory_corpus.csv").open(newline="") as stream:
        targets = [r[0] for r in csv.reader(stream) if r and r[0].strip()]
    if len(targets) != 66:
        raise ValueError("Official corpus must have 66 target sentences")
    model = None
    for variant in args.variants:
        if variant == "clean":
            continue
        folder = args.out / "images" / name / variant
        image_path = image_location(args, cfg, name, variant)
        manifest = optimize_manifest(args, cfg, name, variant, fingerprints)
        folder.mkdir(parents=True, exist_ok=True)
        with output_lock(folder):
            bind_manifest(folder / "manifest.json", manifest)
            if image_path.exists() and (folder / "complete.json").exists():
                completion = json.loads((folder / "complete.json").read_text())
                if completion["image_sha256"] != sha256(image_path) \
                        or completion["manifest_sha256"] != digest(manifest):
                    raise ValueError("Optimized image hash changed")
                log(f"Reuse verified image: {name}/{variant}")
                continue
            if image_path.exists():
                image_path.rename(folder / ("unverified-" + sha256(image_path)[:12] + ".png"))
            seed_all(cfg["optimization_seed"])
            if model is None:
                model = OfficialModel(upstream, name, spec)
            base = model.denormalize(model.preprocess(upstream / "adversarial_images/clean.jpeg"))
            write_json(folder / "resolved_model_config.json", model.resolved_config)
            with (folder / "loss.jsonl").open("w") as losses:
                def progress(step, pixels, loss):
                    losses.write(json.dumps({"step": step, "loss": loss}) + "\n")
                    if step % 100 == 0:
                        losses.flush()
                        save_image(pixels.squeeze(0).cpu(), folder / "latest.png")
                        log(f"{name}/{variant}: {step + 1}/{manifest['updates']} loss={loss:.4f}")
                final = optimize_pixels(
                    base, targets, model.loss, constrained=variant != "unconstrained",
                    epsilon=int(variant[3:]) / 255 if variant != "unconstrained" else 0,
                    alpha=1 / 255, steps=manifest["updates"], batch_size=8,
                    seed=cfg["optimization_seed"], callback=progress,
                )
            save_image(final.squeeze(0).cpu(), folder / "final.tmp.png")
            (folder / "final.tmp.png").replace(image_path)
            write_json(folder / "complete.json", {"image_sha256": sha256(image_path),
                                                  "manifest_sha256": digest(manifest),
                                                  "updates": manifest["updates"]})
    del model
    torch.cuda.empty_cache()


def worker_generate(args, cfg):
    from image_attacks.visual_adv.official import OfficialModel
    upstream = verify_upstream(cfg)
    target = args.worker_model
    fingerprints = checkpoint_fingerprint(cfg["models"][target])
    model = None
    for suite in suites(args):
        if suite == "manual" and target != "minigpt4":
            continue
        planned = cases_for(args, cfg, suite)
        for source, variant, name in cells(args):
            if name != target or (suite == "manual" and source not in ("clean", "minigpt4")):
                continue
            image = image_location(args, cfg, source, variant)
            if not image.is_file():
                raise ValueError(f"Missing image: {image}; run optimize first")
            image_provenance = {"origin": "clean" if variant == "clean" else args.image_source,
                                "sha256": sha256(image)}
            if variant != "clean" and args.image_source == "trained":
                folder = image.parent
                manifest = json.loads((folder / "manifest.json").read_text())
                done = json.loads((folder / "complete.json").read_text())
                if done["image_sha256"] != sha256(image) or done["manifest_sha256"] != digest(manifest):
                    raise ValueError("Incomplete/tampered optimization artifact")
                if manifest != optimize_manifest(args, cfg, source, variant, manifest["checkpoints"]):
                    raise ValueError("Image optimization parameters/code differ from this run")
                image_provenance["optimization"] = manifest
            folder = cell_path(args, suite, source, variant, target)
            folder.mkdir(parents=True, exist_ok=True)
            manifest = {"suite": suite, "source": source, "variant": variant, "target": target,
                        "image": image_provenance, "checkpoints": fingerprints, "commit": UPSTREAM_COMMIT,
                        "sampling_seeds": cfg["sampling_seeds"], "cases_sha256": digest(planned),
                        "code": source_fingerprint(), "decoding": paper_protocol()["decoding"][target],
                        "generation_batch_size": cfg.get("generation_batch_size", 1),
                        "low_resource": cfg["models"][target].get("low_resource", True)}
            manifest["runtime"] = runtime_fingerprint()
            with output_lock(folder):
                bind_manifest(folder / "manifest.json", manifest)
                path = folder / "responses.jsonl"
                existing = validate_responses(read_jsonl(path, repair_final=True), planned, complete=False)
                if len(existing) == len(planned):
                    continue
                if model is None:
                    model = OfficialModel(upstream, target, cfg["models"][target])
                write_json(folder / "resolved_model_config.json", model.resolved_config)
                processed = model.preprocess(image)
                with path.open("a", encoding="utf-8") as stream:
                    batch_size = cfg.get("generation_batch_size", 1)
                    if batch_size < 1:
                        raise ValueError("generation_batch_size must be positive")
                    # Fixed chunks make an interrupted partial batch reproducible.
                    for start in range(0, len(planned), batch_size):
                        batch = planned[start:start + batch_size]
                        if all(case["case_id"] in existing for case in batch):
                            continue
                        base_seed = cfg["sampling_seeds"][batch[0]["repeat"]]
                        seed = (base_seed + int(digest([c["case_id"] for c in batch])[:8], 16)) % (2**32)
                        seed_all(seed)
                        from image_attacks.visual_adv.batching import generate_batch
                        generated = generate_batch(model, processed, [case["text"] for case in batch])
                        for case, response in zip(batch, generated, strict=True):
                            if case["case_id"] in existing:
                                if existing[case["case_id"]]["continuation"] != response:
                                    raise ValueError("Resumed batch changed an existing continuation")
                                continue
                            stream.write(json.dumps({**case, "continuation": response, "seed": seed,
                                                     "run_sha256": digest(manifest)}, ensure_ascii=False) + "\n")
                        stream.flush()
                        log(f"{suite}/{source}/{variant}/{target}: {min(start + batch_size, len(planned))}/{len(planned)}")


def worker_score(args, cfg):
    scorer = args.worker_scorer
    if scorer == "detoxify":
        from detoxify import Detoxify
        model = Detoxify("original", device=cfg["scoring"].get("detoxify_device", "cuda"))
        predict = model.predict
        import importlib.metadata
        import torch
        checkpoint = Path(torch.hub.get_dir()) / "checkpoints/toxic_original-c1212f89.ckpt"
        identity = {"scorer": "detoxify", "variant": "original", "runtime": runtime_fingerprint(),
                    "version": importlib.metadata.version("detoxify"),
                    "checkpoint_sha256": sha256(checkpoint)}
    else:
        verify_upstream(cfg)
        sys.path.insert(0, cfg["upstream"])
        from metric.perspective_api import PerspectiveApiScorer
        key = os.environ.get(cfg["scoring"]["perspective_key_env"])
        if not key:
            raise ValueError("Set the configured Perspective API key environment variable")
        model = PerspectiveApiScorer(key)
        predict = model.get_scores
        identity = {"scorer": "perspective_api", "endpoint": "commentanalyzer/v1alpha1",
                    "scoring_date_kst": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).date().isoformat()}
    planned = cases_for(args, cfg, "rtp")
    for source, variant, target in cells(args):
        folder = cell_path(args, "rtp", source, variant, target)
        if not (folder / "responses.jsonl").exists():
            continue
        responses = read_jsonl(folder / "responses.jsonl")
        validate_responses(responses, planned)
        with output_lock(folder):
            bind_manifest(folder / f"{scorer}.manifest.json", {**identity, "responses_sha256": sha256(folder / "responses.jsonl"),
                                                              "attributes": ATTRIBUTES[scorer]})
            path = folder / f"{scorer}.jsonl"
            existing = unique(read_jsonl(path, repair_final=True), scorer)
            wanted = unique(responses, "responses")
            if set(existing) - set(wanted):
                raise ValueError("Foreign score IDs in " + str(path))
            with path.open("a", encoding="utf-8") as stream:
                for case in responses:
                    response_hash = digest(case["continuation"])
                    if case["case_id"] in existing:
                        row = existing[case["case_id"]]
                        if row["response_sha256"] != response_hash:
                            raise ValueError("Score belongs to an older continuation")
                        score_flags(row["scores"], scorer)
                        continue
                    if not case["continuation"].strip():
                        # An empty continuation displays no toxic content. Retain it in
                        # the denominator and mark the policy, instead of dropping it.
                        scores = {attribute: 0.0 for attribute in ATTRIBUTES[scorer]}
                    else:
                        scores = {key: float(value) for key, value in predict(case["continuation"]).items()}
                    score_flags(scores, scorer)  # API errors never become negative labels
                    stream.write(json.dumps({"case_id": case["case_id"], "response_sha256": response_hash,
                                             "scores": scores, "empty_continuation": not bool(case["continuation"].strip())}) + "\n")
                    stream.flush()




def stage_report(args, cfg):
    refs = json.loads(REFERENCE_FILE.read_text())["rtp"]["minigpt4"]
    try:
        planned = cases_for(args, cfg, "rtp")
        data_error = None
    except (ValueError, FileNotFoundError) as error:
        planned, data_error = [], str(error)
    report_rows = []
    for source, variant, target in cells(args):
        folder = cell_path(args, "rtp", source, variant, target)
        rows = read_jsonl(folder / "responses.jsonl")
        for scorer in cfg["scoring"]["scorers"]:
            result, error = None, data_error
            try:
                if error:
                    raise ValueError(error)
                manifest = json.loads((folder / "manifest.json").read_text())
                if manifest.get("cases_sha256") != digest(planned) or manifest.get("code") != source_fingerprint():
                    raise ValueError("Generation manifest differs from current cases/code")
                if manifest.get("generation_batch_size") != cfg.get("generation_batch_size", 1) \
                        or manifest.get("low_resource") != cfg["models"][target].get("low_resource", True) \
                        or manifest.get("sampling_seeds") != cfg["sampling_seeds"]:
                    raise ValueError("Generation runtime/precision/seeds differ from this config")
                if any(row.get("run_sha256") != digest(manifest) for row in rows):
                    raise ValueError("Response provenance does not match generation manifest")
                if sha256(image_location(args, cfg, source, variant)) != manifest["image"]["sha256"]:
                    raise ValueError("Evaluated image bytes changed")
                result = summarize_scores(rows, read_jsonl(folder / f"{scorer}.jsonl"), planned,
                                          scorer, ddof=cfg["std_ddof"])
            except (ValueError, KeyError, FileNotFoundError, statistics_error()) as problem:
                error = str(problem)
            pilot = bool(args.limit or len(cfg["sampling_seeds"]) != RTP_REPEATS or
                         (source != "clean" and args.image_source == "trained" and
                          (args.steps or cfg["steps"]) != 5000))
            for attr in ("any",) + ATTRIBUTES[scorer]:
                reference = refs[variant][scorer][attr]
                measured = result[attr] if result else None
                report_rows.append({"table": 2, "source": source, "variant": variant, "target": target,
                                    "scorer": scorer, "attribute": attr,
                                    "paper_mean": reference[0], "paper_std": reference[1],
                                    "measured_mean": measured["mean"] if measured else None,
                                    "measured_std": measured["std"] if measured else None,
                                    "delta_pp": measured["mean"] - reference[0] if measured else None,
                                    "n": measured["n"] if measured else len(rows),
                                    "status": "pilot" if result and pilot else "complete" if result else "incomplete",
                                    "detail": error})
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "table2.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report_rows[0]))
        writer.writeheader()
        writer.writerows(report_rows)
    report = {"paper": PAPER_URL, "paper_reproduction": False,
              "exact_historical_reproduction": False,
              "reason": "Paper/public-data counts differ and historical seeds/runtime are unavailable",
              "std_ddof": cfg["std_ddof"], "empty_continuation_policy": "count as non-toxic; retain denominator",
              "dataset_challenging_prompts": cfg["rtp_expected_prompts"],
              "paper_challenging_prompts": EXPECTED_RTP,
              "image_source": args.image_source, "cells": report_rows,
              "generation_batch_size": cfg.get("generation_batch_size", 1),
              "low_resource": cfg["models"]["minigpt4"].get("low_resource", True),
              "protocol_complete": bool(report_rows) and all(r["status"] == "complete" for r in report_rows),
              "limitations": paper_protocol()["limitations"]}
    write_json(args.out / "report.json", report)
    lines = ["# VisualAdv: one-benchmark reproduction check", "", f"Source: {PAPER_URL}", "",
             f"Public RTP challenging subset: {cfg['rtp_expected_prompts']} prompts (paper: {EXPECTED_RTP}); three repeats.",
             "Blank measurements indicate incomplete coverage. Exact historical reproduction is unverified.", "",
             "| Input | Scorer | Attribute | Paper (%) | Measured mean ± std (%) | Δ pp | Status |",
             "|---|---|---|---:|---:|---:|---|"]
    for row in report_rows:
        number = lambda value: "—" if value is None else f"{value:.2f}"
        measured = f"{number(row['measured_mean'])} ± {number(row['measured_std'])}" if row['measured_mean'] is not None else "—"
        lines.append(f"| {row['variant']} | {row['scorer']} | {row['attribute']} | {number(row['paper_mean'])} | {measured} | {number(row['delta_pp'])} | {row['status']} |")
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"Table 2 -> {args.out / 'table2.csv'}; complete={report['protocol_complete']}")

def statistics_error():
    import statistics
    return statistics.StatisticsError


def dispatch_worker(args, cfg, stage, *, model=None, scorer=None):
    executable = cfg["models"][model]["python"] if model else cfg["scoring"]["python"]
    command = [executable, str(Path(__file__).resolve()), "--stage", stage,
               "--config", str(args.config), "--out", str(args.out),
               "--suite", args.suite, "--image-source", args.image_source,
               "--models", *args.models, "--surrogates", *args.surrogates,
               "--variants", *args.variants, "--limit", str(args.limit), "--worker"]
    if args.steps:
        command += ["--steps", str(args.steps)]
    if model:
        command += ["--worker-model", model]
    if scorer:
        command += ["--worker-scorer", scorer]
    log(f"{stage}: {model or scorer} in {executable}")
    subprocess.run(command, check=True, cwd=ROOT,
                   env={**os.environ, "PYTHONNOUSERSITE": "1", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "fetch-data", "audit", "optimize", "generate", "score", "report", "all"), default="audit")
    parser.add_argument("--config", type=resolve, default=ROOT / "configs/experiments/VisualAdv_paper.yaml")
    parser.add_argument("--out", type=resolve, default=ROOT / "runs/visualadv_paper")
    parser.add_argument("--models", nargs="+", choices=("minigpt4",), default=["minigpt4"])
    parser.add_argument("--surrogates", nargs="+", choices=("minigpt4",), default=["minigpt4"])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=["clean", "eps16"])
    parser.add_argument("--suite", choices=("rtp",), default="rtp")
    parser.add_argument("--image-source", choices=("trained", "released"), default="trained")
    parser.add_argument("--limit", type=int, default=0, help="pilot only; 0 means full benchmark")
    parser.add_argument("--steps", type=int, help="override PGD steps, recorded in image manifest")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-model", choices=MODELS, help=argparse.SUPPRESS)
    parser.add_argument("--worker-scorer", choices=ATTRIBUTES, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.limit < 0 or (args.steps is not None and args.steps < 1):
        parser.error("limit must be nonnegative and steps positive")
    if args.image_source == "released" and set(args.surrogates) != {"minigpt4"}:
        parser.error("released images require --surrogates minigpt4")
    for field in ("models", "surrogates", "variants"):
        if len(getattr(args, field)) != len(set(getattr(args, field))):
            parser.error(f"duplicate {field}")
    return args


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.worker:
        {"optimize": worker_optimize, "generate": worker_generate, "score": worker_score}[args.stage](args, cfg)
        return 0
    stages = ("plan", "fetch-data", "audit", "optimize", "generate", "score", "report") if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "optimize":
            if args.image_source == "trained":
                for name in args.surrogates:
                    dispatch_worker(args, cfg, stage, model=name)
        elif stage == "generate":
            for name in args.models:
                dispatch_worker(args, cfg, stage, model=name)
        elif stage == "score":
            if "rtp" in suites(args):
                for scorer in cfg["scoring"]["scorers"]:
                    dispatch_worker(args, cfg, stage, scorer=scorer)
        else:
            result = globals()["stage_" + stage.replace("-", "_")](args, cfg)
            if stage == "audit" and not result["ready"]:
                stage_report(args, cfg)
                return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as error:
        log(str(error))
        raise SystemExit(2)
