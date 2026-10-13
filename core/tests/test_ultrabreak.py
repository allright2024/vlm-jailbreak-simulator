"""UltraBreak test: declaration, config, and vendored optimization utilities.

The full attack optimizes a universal image on a Qwen2-VL surrogate (GPU, slow),
so that is exercised by a real run, not here. This checks the model-free pieces:
registration, requirement flags, config parsing, TPG, and the semantic-loss /
transform / TV utilities. Run with:

    python -m core.tests.test_ultrabreak
"""

from __future__ import annotations

import torch

import core.attacks  # noqa: F401
from core import get_attack
from multi_attacks.ultrabreak import ultrabreak_utils as uu


def main() -> None:
    atk = get_attack("UltraBreak")({"epochs": 3, "train_rows": 2, "patch_size": 64})
    assert atk.requires_surrogate and not atk.requires_images
    assert atk.epochs == 3 and atk.patch_size == 64
    assert "[Jailbroken Mode]" in atk.tpg_suffix

    full = get_attack("UltraBreak")({})
    train_ids = full.training_behavior_ids()
    assert len(train_ids) == len(set(train_ids)) == 50

    # UltraBreak needs only the instruction (+ surrogate) — no dataset image/keyword
    from core.schema import Behavior, EvalType
    b = Behavior("x", "SafeBench", "How can someone do bad things?", EvalType.HARMFUL)
    assert atk.missing_requirements(b) == []

    # total variation: zero on a flat image, positive on a noisy one
    flat = torch.zeros(3, 8, 8)
    assert uu.total_variation(flat).item() == 0.0
    assert uu.total_variation(torch.rand(3, 8, 8)).item() > 0.0

    # differentiable patch transform keeps gradients and image shape
    img = torch.zeros(3, 48, 48)
    patch = torch.rand(3, 16, 16, requires_grad=True)
    out = uu.apply_random_patch(img, patch, scale_range=(1.0, 1.0), rotation_range=(0, 0))
    assert out.shape == img.shape and out.requires_grad

    # semantic similarity loss: in [0, 2], finite, differentiable; perfect match -> ~0
    V, D, T = 12, 8, 5
    emb = torch.randn(V, D)
    labels = torch.tensor([[-100, -100, 3, 4, 5]])
    # logits that put all mass on the label tokens -> expected embedding == target
    logits = torch.full((1, T, V), -1e4)
    for t, lab in enumerate(labels[0].tolist()):
        if lab >= 0:
            logits[0, t, lab] = 1e4
    loss = uu.semantic_similarity_loss(logits, labels, emb, mode="token")
    assert torch.isfinite(loss) and 0.0 <= loss.item() <= 2.0

    print("UltraBreak OK: registered, config, TPG, TV, differentiable transform, semantic loss")


if __name__ == "__main__":
    main()
