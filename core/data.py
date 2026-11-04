"""External benchmark-data contract and preflight validation."""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

from .paths import DATA_ROOT, DATA_ROOT_ENV, resolve_data_path, resolve_path

# Minimal layout needed to load each dataset. Deep validation additionally
# loads every row and checks all referenced images.
DATASET_LAYOUT: dict[str, tuple[str, ...]] = {
    "AdvBench": ("benchmarks/advbench/harmful_behaviors.csv",),
    "AdvBenchM": (
        "benchmarks/AdvBenchM/instructions.jsonl",
        "benchmarks/AdvBenchM/images.jsonl",
        "benchmarks/AdvBenchM/dataset/advimage",
    ),
    "JOOD": (
        "method_assets/JOOD/prompts/eval_all_instructions",
        "method_assets/JOOD/prompts/all_instructions",
        "method_assets/JOOD/prompts/all_instructions_harmful_annotated",
        "method_assets/JOOD/images/harmful",
        "method_assets/JOOD/images/harmless",
    ),
    "CS-DJ": ("method_assets/CS-DJ/instructions",),
    "HADES": ("benchmarks/HADES/hades.csv",),
    "HarmBench": ("benchmarks/HarmBench/harmbench_standard.json",),
    "MM-SafetyBench": ("benchmarks/MM-SafetyBench",),
    "SafeBench": ("benchmarks/SafeBench/safebench.csv",),
}

EXPECTED_COUNTS = {
    "AdvBench": 520,
    "AdvBenchM": 246,
    "JOOD": 170,
    "CS-DJ": 750,
    "HADES": 750,
    "HarmBench": 200,
    "MM-SafetyBench": 1680,
    "SafeBench": 500,
}


def _data_path(relative: str) -> str:
    return resolve_data_path(relative)


def attack_requirements(name: str, config: dict[str, Any] | None = None) -> list[str]:
    """Concrete files/directories needed before an attack can build cases."""
    c = config or {}
    if name == "VisualAdv":
        return [
            resolve_path(c.get("image_path", "data/method_assets/FORCE/clean.jpeg")),
            resolve_path(c.get("target_path", "data/method_assets/VisualAdv/derogatory_corpus.csv")),
        ]
    if name == "JPS":
        paths = [resolve_path(c.get("base_image", "data/method_assets/JPS/clean.jpeg"))]
        train_csv = c.get("train_csv")
        if train_csv:
            paths.append(resolve_path(train_csv))
        else:
            train_name = str(c.get("train_dataset", "AdvBench"))
            paths.extend(_data_path(p) for p in DATASET_LAYOUT.get(train_name, ()))
        return paths
    if name == "CS-DJ":
        paths = [resolve_path(c.get("image_dir", "data/method_assets/CS-DJ/llava_images"))]
        if c.get("strict_font") and c.get("font_path"):
            paths.append(resolve_path(c["font_path"]))
        return paths
    if name == "MML" and str(c.get("input_mode", "generated")).lower() == "precomputed":
        paths = [resolve_path(c.get("precomputed_root", "data/method_assets/MML"))]
        if str(c.get("prompt_source", "clean_room")).lower() == "upstream":
            paths.append(resolve_path(c.get(
                "upstream_const_path", "data/method_assets/MML/upstream/const.py"
            )))
        return paths
    if name == "DeltaJP":
        train_name = str(c.get("train_dataset", "AdvBenchM"))
        return [_data_path(p) for p in DATASET_LAYOUT.get(train_name, ())]
    if name == "HADES" and str(c.get("mode", "white_box")).lower() == "white_box":
        train_name = str(c.get("train_dataset", "HADES"))
        return [_data_path(p) for p in DATASET_LAYOUT.get(train_name, ())]
    if name == "JOOD":
        return [resolve_path(c.get("harmless_dir", "data/method_assets/JOOD/images/harmless"))]
    return []


def missing_data(
    datasets: Iterable[str] = (),
    attacks: Iterable[str] = (),
    attack_configs: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    """Return missing required paths without reading model weights or APIs."""
    required: list[str] = []
    for name in datasets:
        required.extend(_data_path(p) for p in DATASET_LAYOUT.get(name, ()))
    configs = attack_configs or {}
    for name in attacks:
        required.extend(attack_requirements(name, configs.get(name)))
    return sorted({path for path in required if not os.path.exists(path)})


def require_data(
    datasets: Iterable[str] = (),
    attacks: Iterable[str] = (),
    attack_configs: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Fail early with an actionable message when external data is unavailable."""
    missing = missing_data(datasets, attacks, attack_configs)
    if not missing:
        return
    shown = "\n".join(f"  - {path}" for path in missing)
    raise FileNotFoundError(
        "Required benchmark data is missing. Data is not distributed in this Git "
        f"repository. Set {DATA_ROOT_ENV} to the downloaded Drive directory "
        f"(current: {DATA_ROOT}) and run `python -m core.validation.data`.\n{shown}"
    )


def all_data_available() -> bool:
    return not missing_data(DATASET_LAYOUT, ("VisualAdv", "JPS", "DeltaJP", "JOOD"))
