"""Validate a cached universal attack artifact against its lineage sidecar.

Used by the full-run shell scripts before accepting a cache hit. The supplied
config is the optimization config (without ``adv_image``); attack-specific
hooks locate auxiliary files such as JPS's steer prompt next to the artifact.
"""

from __future__ import annotations

import argparse
import os

import core.attacks  # noqa: F401  register attacks
from core import get_attack
from core.config import load_attack_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify an attack artifact lineage sidecar.")
    parser.add_argument("--attack", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifact", required=True)
    args = parser.parse_args()

    attack_cls = get_attack(args.attack)
    attack = attack_cls(load_attack_config(attack_cls.name, args.config))
    artifact = os.path.abspath(args.artifact)
    if attack.verify_lineage(artifact):
        print(f"lineage verified: {artifact}")
        return 0
    print(f"lineage mismatch or missing sidecar: {artifact}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
