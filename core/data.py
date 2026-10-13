"""External benchmark-data contract and preflight validation."""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

from .paths import DATA_ROOT, DATA_ROOT_ENV, resolve_path

# Minimal layout needed to load each dataset. Deep validation additionally
# loads every row and checks all referenced images.
DATASET_LAYOUT: dict[str, tuple[str, ...]] = {
    "AdvBench": ("advbench/harmful_behaviors.csv",),
    "AdvBenchM": (
        "AdvBenchM/prompts/eval_all_instructions",
        "AdvBenchM/prompts/all_instructions",
        "AdvBenchM/prompts/all_instructions_harmful_annotated",
        "AdvBenchM/images/harmful",
        "AdvBenchM/images/harmless",
    ),
    "HADES": ("HADES/hades.csv",),
    "MM-SafetyBench": ("MM-SafetyBench",),
    "SafeBench": ("SafeBench/safebench.csv",),
}

EXPECTED_COUNTS = {
    "AdvBench": 520,
    "AdvBenchM": 170,
    "HADES": 750,
    "MM-SafetyBench": 1680,
    "SafeBench": 500,
}


def _data_path(relative: str) -> str:
    return os.path.join(DATA_ROOT, relative)


def attack_requirements(name: str, config: dict[str, Any] | None = None) -> list[str]:
    """Concrete files/directories needed before an attack can build cases."""
    c = config or {}
    if name == "FORCE":
        return [resolve_path(c.get("image_path", "data/FORCE/clean.jpeg"))]
    if name == "VisualAdv":
        return [
            resolve_path(c.get("image_path", "data/FORCE/clean.jpeg")),
            resolve_path(c.get("target_path", "data/legacy/derogatory_corpus.csv")),
        ]
    if name == "UltraBreak":
        return [
            resolve_path(c.get(
                "train_config",
                "data/UltraBreak/train_configs/safebench_white_force_jailbroken_mode.csv",
            )),
            resolve_path(c.get("base_image", "data/UltraBreak/images/white.jpeg")),
        ]
    if name == "JPS":
        paths = [resolve_path(c.get("base_image", "data/FORCE/clean.jpeg"))]
        train_csv = c.get("train_csv")
        if train_csv:
            paths.append(resolve_path(train_csv))
        else:
            train_name = str(c.get("train_dataset", "AdvBench"))
            paths.extend(_data_path(p) for p in DATASET_LAYOUT.get(train_name, ()))
        return paths
    if name == "DeltaJP":
        train_name = str(c.get("train_dataset", "AdvBenchM"))
        return [_data_path(p) for p in DATASET_LAYOUT.get(train_name, ())]
    if name == "JOOD":
        return [resolve_path(c.get("harmless_dir", "data/AdvBenchM/images/harmless"))]
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
        f"(current: {DATA_ROOT}) and run `python validate_data.py`.\n{shown}"
    )


def all_data_available() -> bool:
    return not missing_data(DATASET_LAYOUT, ("FORCE", "VisualAdv", "UltraBreak", "JPS", "DeltaJP", "JOOD"))
