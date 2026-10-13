"""Integrity check for all dataset loaders against the real files under data/.

Verifies, per dataset: behavior count, unique ids, non-empty instruction,
which optional fields are populated, image count per behavior, that every image
path resolves, and the eval_type split. Also checks ids are unique across
datasets. Run with:

    python -m core.tests.test_datasets
"""

from __future__ import annotations

import os
from collections import Counter

import core.datasets  # noqa: F401  (registers loaders)
from core import REPO_ROOT, EvalType, get_dataset, list_datasets

# Expected shape of each dataset: count, images per behavior, and which optional
# fields must be filled for every behavior.
EXPECTED = {
    "AdvBench": dict(n=520, images=0, fields={"target"}),
    "AdvBenchM": dict(n=170, images=1, fields={"keyword"}),
    "HADES": dict(n=750, images=6, fields={"keyword", "keyword_type"}),
    "MM-SafetyBench": dict(n=1680, images=3, fields={"keyword", "keyword_type"}),
    "SafeBench": dict(n=500, images=0, fields={"statement"}),
}
OPTIONAL = ("keyword", "keyword_type", "statement", "target")


def check(name: str) -> list:
    exp = EXPECTED[name]
    bs = get_dataset(name)().load()
    assert len(bs) == exp["n"], f"{name}: {len(bs)} behaviors, expected {exp['n']}"
    assert len({b.behavior_id for b in bs}) == len(bs), f"{name}: duplicate behavior_id"
    assert all(b.dataset == name for b in bs), f"{name}: wrong dataset field"
    assert all(b.instruction.strip() for b in bs), f"{name}: empty instruction"

    for f in OPTIONAL:
        filled = sum(bool(getattr(b, f)) for b in bs)
        if f in exp["fields"]:
            assert filled == len(bs), f"{name}: {f} filled for {filled}/{len(bs)}"
        else:
            assert filled == 0, f"{name}: unexpected {f} on {filled} behaviors"

    assert all(len(b.images) == exp["images"] for b in bs), f"{name}: image count"
    missing = [p for b in bs for p in b.images if not os.path.isfile(os.path.join(REPO_ROOT, p))]
    assert not missing, f"{name}: {len(missing)} missing images, e.g. {missing[:3]}"

    et = Counter(b.eval_type.value for b in bs)
    cats = len({b.category for b in bs if b.category})
    print(f"  {name:15s} n={len(bs):5d}  categories={cats:2d}  "
          f"images/behavior={exp['images']}  fields={sorted(exp['fields'])}  eval_type={dict(et)}")
    return bs


def main() -> None:
    # The real loaders must all be registered; other tests may register extra
    # stand-in datasets in the same process, so require a superset, not equality.
    assert set(EXPECTED) <= set(list_datasets()), list_datasets()
    print("dataset loaders:")
    all_ids = []
    for name in sorted(EXPECTED):
        all_ids += [b.behavior_id for b in check(name)]
    assert len(all_ids) == len(set(all_ids)), "behavior_id collision across datasets"
    print(f"OK: {len(all_ids)} behaviors across {len(EXPECTED)} datasets, all ids globally unique")


if __name__ == "__main__":
    main()
