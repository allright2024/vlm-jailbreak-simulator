"""Dataset-selection policies for jailbreak generation.

The source benchmarks contain evaluation categories that are useful for other
experiments (professional advice, subjective opinion, copyright), but they are
not harmful-instruction jailbreak positives.  Keeping this policy in one place
prevents individual runners from silently mixing those records into a
jailbreak dataset.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from .schema import Behavior, EvalType


JAILBREAK_EVAL_TYPES = frozenset({EvalType.HARMFUL})


def parse_eval_type_spec(value: str) -> tuple[EvalType, ...]:
    """Parse ``harmful,advice`` or ``all`` without importing a CLI module."""

    names = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not names:
        raise ValueError("eval-type specification must not be empty")
    if names == ["all"]:
        return tuple(EvalType)
    if "all" in names:
        raise ValueError("use 'all' alone, not with other eval types")
    try:
        return tuple(dict.fromkeys(EvalType(name) for name in names))
    except ValueError as exc:
        choices = ", ".join(item.value for item in EvalType)
        raise ValueError(
            f"unknown eval type in {value!r}; choose from {choices}, or all"
        ) from exc


def select_eval_types(
    behaviors: Iterable[Behavior],
    allowed: Iterable[EvalType] = JAILBREAK_EVAL_TYPES,
) -> tuple[list[Behavior], dict[str, str]]:
    """Return allowed behaviors and auditable exclusion reasons.

    The default is intentionally harmful-only.  Callers running a separate,
    non-jailbreak benchmark may pass a different explicit set.
    """

    allowed_set = frozenset(allowed)
    if not allowed_set:
        raise ValueError("allowed eval types must not be empty")

    selected: list[Behavior] = []
    excluded: dict[str, str] = {}
    for behavior in behaviors:
        if behavior.eval_type in allowed_set:
            selected.append(behavior)
        else:
            excluded[behavior.behavior_id] = (
                f"policy_eval_type:{behavior.eval_type.value}"
            )
    return selected, excluded


def require_harmful_only(behaviors: Iterable[Behavior]) -> None:
    """Fail closed when any non-harmful behavior reaches jailbreak output."""

    invalid = [
        (behavior.behavior_id, behavior.eval_type.value)
        for behavior in behaviors
        if behavior.eval_type is not EvalType.HARMFUL
    ]
    if invalid:
        preview = ", ".join(f"{bid}={kind}" for bid, kind in invalid[:5])
        suffix = "" if len(invalid) <= 5 else f" (+{len(invalid) - 5} more)"
        raise ValueError(
            "jailbreak datasets require eval_type=harmful; found "
            f"{preview}{suffix}"
        )


def eval_type_counts(behaviors: Iterable[Behavior]) -> dict[str, int]:
    """Stable, JSON-ready counts for manifests and audits."""

    counts = Counter(behavior.eval_type.value for behavior in behaviors)
    return {name: counts[name] for name in sorted(counts)}
