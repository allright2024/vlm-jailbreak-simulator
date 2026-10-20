import os
import yaml

from .adashield import AdaShieldDefense
from .asd import AsDDefense
from .ecso import ECSODefense

DEFENSE_REGISTRY = {
    "AdaShield": AdaShieldDefense,
    "AsD": AsDDefense,
    "ECSO": ECSODefense,
}

try:
    from .jailguard import JailGuardDefense
    DEFENSE_REGISTRY["JailGuard"] = JailGuardDefense
except Exception as _e:  # noqa
    pass

try:
    from .hiddendetect import HiddenDetectDefense
    DEFENSE_REGISTRY["HiddenDetect"] = HiddenDetectDefense
except Exception as _e:  # noqa
    pass

try:
    from .cider import CIDERDefense
    DEFENSE_REGISTRY["CIDER"] = CIDERDefense
except Exception as _e:  # noqa
    pass


def load_defense(
    defense_name: str,
    config_dir: str = "configs/defense_configs",
    config_path: str = None,
):
    if defense_name not in DEFENSE_REGISTRY:
        raise ValueError(
            f"Unknown defense_name '{defense_name}'. "
            f"Available: {list(DEFENSE_REGISTRY.keys())}"
        )

    config_path = config_path or os.path.join(config_dir, f"{defense_name}.yaml")
    config_path = os.path.abspath(os.path.expanduser(config_path))
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Defense config must be a YAML mapping: {config_path}")
    config["_config_path"] = config_path

    return DEFENSE_REGISTRY[defense_name](config)
