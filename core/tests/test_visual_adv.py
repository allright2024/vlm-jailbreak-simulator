"""VisualAdv test: declaration, config, corpus loading, normalize round-trip (no model).

The PGD optimization needs the LLaVA surrogate (GPU) and is exercised by a real
run. Run with:
    python -m core.tests.test_visual_adv
"""

from __future__ import annotations

import os

import torch

import core.attacks  # noqa: F401
from core import REPO_ROOT, get_attack


def main() -> None:
    atk = get_attack("VisualAdv")({"epsilon": 32, "alpha": 2, "n_iters": 10,
                                   "batch_size": 4, "constrained": True})
    assert atk.requires_surrogate and not atk.requires_images
    assert abs(atk.epsilon - 32 / 255) < 1e-9 and atk.batch_size == 4
    # default backend is white-box LLaVA; transfer selects the Qwen2-VL surrogate
    assert atk.surrogate_backend == "llava"
    tr = get_attack("VisualAdv")({"surrogate_backend": "qwen2"})
    assert tr.surrogate_backend == "qwen2"

    # VisualAdv needs only the instruction (+ surrogate)
    from core.schema import Behavior, EvalType
    b = Behavior("x", "AdvBench", "do something harmful", EvalType.HARMFUL)
    assert atk.missing_requirements(b) == []

    # corpus loads and is non-empty
    targets = atk._load_targets()
    assert len(targets) > 0 and all(isinstance(t, str) and t for t in targets)

    # base image exists
    assert os.path.isfile(atk._rel(atk.image_path)), atk.image_path

    # the LLaVA backend's normalize/denormalize round-trips in [0,1] space
    from image_attacks.ce_surrogate import _CLIP_MEAN, _CLIP_STD
    x = torch.rand(1, 3, 8, 8)
    mean = torch.tensor(_CLIP_MEAN)[None, :, None, None]
    std = torch.tensor(_CLIP_STD)[None, :, None, None]
    rt = ((x - mean) / std) * std + mean
    assert torch.allclose(x, rt, atol=1e-5)

    print(f"VisualAdv OK: registered, config (llava/qwen2 backends), "
          f"corpus ({len(targets)} targets), norm round-trip")


if __name__ == "__main__":
    main()
