"""FORCE attack test: declaration, config, and frequency-mask helper (no model).

The full attack needs a LLaVA surrogate with gradients, so the end-to-end PGD is
exercised by a real run, not here. This checks the pieces that don't need the
model: registration, requirement flags, config parsing, and the vendored
frequency-mask / CE-loss utilities. Run with:

    python -m core.tests.test_force
"""

from __future__ import annotations

import core.attacks  # noqa: F401
from core import get_attack


def main() -> None:
    cls = get_attack("FORCE")
    atk = cls({"epsilon": 16, "num_band": 6, "max_iter": 7, "max_query": 1})
    assert atk.requires_surrogate and not atk.requires_images
    assert atk.epsilon == 16 and atk.num_band == 6 and atk.max_iter == 7
    assert atk.target_text == "Sure, here is"

    # FORCE needs no dataset image/keyword/target, only the instruction + surrogate
    from core.schema import Behavior, EvalType
    b = Behavior("x", "AdvBench", "do bad things", EvalType.HARMFUL)
    assert atk.missing_requirements(b) == []  # surrogate requirement isn't a behavior field

    # vendored utils are importable and shaped right (CPU, no model)
    import torch
    from multi_attacks.force import force_utils as fu

    masks = fu.generate_frequency_masks(8, 8, K=6, device="cpu")
    assert len(masks) == 6
    stacked = torch.stack([m for m in masks]).int().sum(0)
    assert (stacked <= 1).all(), "frequency bands must be disjoint"
    # bands partition most of the plane (corners beyond r_max bound can be unassigned)
    assert stacked.sum() >= 0.6 * 8 * 8

    logits = torch.randn(2, 5, 10)
    labels = torch.full((2, 5), -100)
    labels[:, 3:] = torch.tensor([[1, 2], [3, 4]])
    loss = fu.per_sample_ce_loss(logits, labels)
    assert loss.shape == (2,) and torch.isfinite(loss).all()

    print("FORCE OK: registered, requirement flags, config, frequency masks disjoint, CE loss")


if __name__ == "__main__":
    main()
