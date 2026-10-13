"""Single source of truth for every model this repo loads.

Each entry pins a HuggingFace repo id to an exact commit (``revision``) so the
eval target, the judge, attack surrogates and defense encoders are reproducible
and auditable in one table. The pinned revisions were taken from, and are
checked against, the local HF cache snapshots (see
``core/tests/test_model_loading.py``). An entry whose ``revision`` is ``None``
is not pinned yet (it was not in the local cache) and resolves to the hub's
default branch; pin it before relying on it for a reproducible run.

This module is pure data (no torch / transformers import), so attack modules
can read pinned revisions without pulling in model code. The loading itself
lives in ``multimodalmodels/loaders.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelSpec:
    hf_id: str
    revision: Optional[str]   # pinned commit; None = not pinned yet
    kind: str                 # loader family in multimodalmodels/loaders.py
    used_by: str              # where the repo uses it (for auditing)


LLAVA_15_HF_ID = "llava-hf/llava-1.5-7b-hf"
QWEN2_VL_HF_ID = "Qwen/Qwen2-VL-7B-Instruct"
QWEN25_VL_HF_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
HARMBENCH_CLS_HF_ID = "cais/HarmBench-Llama-2-13b-cls"
VIT_BASE_HF_ID = "google/vit-base-patch16-224"

LLAVA_15_REVISION = "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
QWEN2_VL_REVISION = "eed13092ef92e448dd6875b2a00151bd3f7db0ac"
HARMBENCH_CLS_REVISION = "bda705349d1144fa618770bea64d99ce54e3835b"

MODEL_SPECS: dict[str, ModelSpec] = {
    spec.hf_id: spec
    for spec in (
        ModelSpec(LLAVA_15_HF_ID, LLAVA_15_REVISION, "llava",
                  "eval target; white-box surrogate (FORCE, VisualAdv, DeltaJP); "
                  "CIDER encoder"),
        ModelSpec(QWEN2_VL_HF_ID, QWEN2_VL_REVISION, "qwen2-vl",
                  "transfer surrogate (UltraBreak, JPS, VisualAdv/DeltaJP qwen2)"),
        ModelSpec(QWEN25_VL_HF_ID, None, "qwen2-vl",
                  "optional Qwen2.5-VL surrogate (not pinned: not in local cache)"),
        ModelSpec(HARMBENCH_CLS_HF_ID, HARMBENCH_CLS_REVISION, "causal-lm-classifier",
                  "HarmBench judge"),
        ModelSpec(VIT_BASE_HF_ID, None, "image-classifier",
                  "CIDER denoiser classifier fallback (not pinned: not in local cache)"),
    )
}


def get_spec(hf_id: str) -> Optional[ModelSpec]:
    """The registered spec for ``hf_id`` (None for local paths / unknown ids)."""
    return MODEL_SPECS.get(hf_id)


def default_revision(hf_id: str) -> Optional[str]:
    """The pinned revision for ``hf_id``, or None if unknown or not pinned."""
    spec = MODEL_SPECS.get(hf_id)
    return spec.revision if spec else None
