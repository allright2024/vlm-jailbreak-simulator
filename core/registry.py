"""Name -> class registries for attacks and datasets.

Attacks and dataset loaders register themselves with a decorator, so the
pipeline can look them up by the string a user passes on the command line
without importing every module eagerly or maintaining a hand-written mapping
(the old ``attacks/__init__.py`` approach, which also deleted modules from
``sys.modules`` on every lookup).
"""

from __future__ import annotations

from typing import Callable, TypeVar

_ATTACKS: dict[str, type] = {}
_DATASETS: dict[str, type] = {}

T = TypeVar("T")


def register_attack(name: str) -> Callable[[type[T]], type[T]]:
    """Class decorator: register an Attack subclass under ``name``."""

    def deco(cls: type[T]) -> type[T]:
        key = name.lower()
        if key in _ATTACKS and _ATTACKS[key] is not cls:
            raise ValueError(f"Attack name already registered: {name!r}")
        _ATTACKS[key] = cls
        cls.name = name  # type: ignore[attr-defined]
        return cls

    return deco


def register_dataset(name: str) -> Callable[[type[T]], type[T]]:
    """Class decorator: register a Dataset subclass under ``name``."""

    def deco(cls: type[T]) -> type[T]:
        key = name.lower()
        if key in _DATASETS and _DATASETS[key] is not cls:
            raise ValueError(f"Dataset name already registered: {name!r}")
        _DATASETS[key] = cls
        cls.name = name  # type: ignore[attr-defined]
        return cls

    return deco


def get_attack(name: str) -> type:
    try:
        return _ATTACKS[name.lower()]
    except KeyError:
        raise KeyError(
            f"Unknown attack {name!r}. Registered: {sorted(_ATTACKS)}"
        ) from None


def get_dataset(name: str) -> type:
    try:
        return _DATASETS[name.lower()]
    except KeyError:
        raise KeyError(
            f"Unknown dataset {name!r}. Registered: {sorted(_DATASETS)}"
        ) from None


def list_attacks() -> list[str]:
    return sorted(c.name for c in set(_ATTACKS.values()))  # type: ignore[attr-defined]


def list_datasets() -> list[str]:
    return sorted(c.name for c in set(_DATASETS.values()))  # type: ignore[attr-defined]
