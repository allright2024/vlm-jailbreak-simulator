"""Attack config loading.

An attack's hyperparameters live in ``configs/attacks/<Attack>.yaml`` (new
unified location). ``load_attack_config`` returns a plain dict, or {} when no
config file exists, so attacks with no hyperparameters need no file.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from .dataset import REPO_ROOT

ATTACK_CONFIG_DIR = os.path.join(REPO_ROOT, "configs", "attacks")


def load_attack_config(attack_name: str, path: Optional[str] = None) -> dict[str, Any]:
    cfg_path = path or os.path.join(ATTACK_CONFIG_DIR, f"{attack_name}.yaml")
    if not os.path.isfile(cfg_path):
        return {}
    import yaml

    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
