"""Centralized model loading: every model this repo uses is loaded here.

The eval target, the judge, attack surrogates (FORCE, UltraBreak, JPS,
VisualAdv, DeltaJP) and defense models (CIDER, JailGuard) all call these
helpers instead of ``*.from_pretrained`` / ``torch.load`` / ``spacy.load``
directly, so model ids, pinned revisions, dtype and device placement live in
one place. ``core/tests/test_model_loading.py`` enforces that no code outside
``multimodalmodels/`` loads a model itself.

When ``revision`` is None, the pinned revision from ``registry.py`` is used for
registered model ids (local paths and unregistered ids load as given).
Attack-specific machinery (FORCE's layer hooks, UltraBreak's patchify and
semantic loss) stays in the attacks; only the loading lives here.

Heavy imports (torch, transformers, spacy) are deferred to call time so that
importing this module is cheap.
"""

from __future__ import annotations

from typing import Any, Optional

from .registry import (HARMBENCH_CLS_HF_ID, LLAVA_15_HF_ID, QWEN25_VL_HF_ID,
                       QWEN2_VL_HF_ID, default_revision)


def _revision(model_id: str, revision: Optional[str]) -> Optional[str]:
    """An explicit revision wins; otherwise use the registry pin (if any)."""
    return revision if revision is not None else default_revision(model_id)


def _dtype(torch_dtype: Any):
    """Accept "float16" / "bfloat16" / "float32" / "auto" or a torch.dtype."""
    if not isinstance(torch_dtype, str) or torch_dtype == "auto":
        return torch_dtype
    import torch
    return getattr(torch, torch_dtype)


def load_llava(
    model_id: str = LLAVA_15_HF_ID,
    revision: Optional[str] = None,
    torch_dtype: Any = "float16",
    device_map: Optional[str] = "auto",
    to_device=None,
    eval_mode: bool = False,
    processor_kwargs: Optional[dict] = None,
):
    """LLaVA-1.5 model + processor.

    ``device_map="auto"`` shards across visible GPUs (inference: the eval target
    and the CE surrogate). Pass ``device_map=None`` with ``to_device`` for a
    single-device white-box attack that installs forward hooks (FORCE).
    """
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    revision = _revision(model_id, revision)
    kwargs = {"revision": revision, "torch_dtype": _dtype(torch_dtype)}
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = LlavaForConditionalGeneration.from_pretrained(model_id, **kwargs)
    if to_device is not None:
        model = model.to(to_device)
    if eval_mode:
        model = model.eval()
    processor = AutoProcessor.from_pretrained(model_id, revision=revision, **(processor_kwargs or {}))
    return model, processor


def load_qwen2vl(
    model_id: str = QWEN2_VL_HF_ID,
    revision: Optional[str] = None,
    torch_dtype: Any = "auto",
    device_map: Optional[str] = "auto",
):
    """Qwen2-VL (or Qwen2.5-VL) model + processor."""
    from transformers import (AutoProcessor, Qwen2_5_VLForConditionalGeneration,
                              Qwen2VLForConditionalGeneration)

    revision = _revision(model_id, revision)
    processor = AutoProcessor.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    cls = (Qwen2_5_VLForConditionalGeneration if model_id == QWEN25_VL_HF_ID
           else Qwen2VLForConditionalGeneration)
    model = cls.from_pretrained(model_id, revision=revision,
                                torch_dtype=_dtype(torch_dtype), device_map=device_map)
    return model, processor


def load_causal_lm_classifier(
    model_id: str = HARMBENCH_CLS_HF_ID,
    revision: Optional[str] = None,
    device_map: Optional[str] = "auto",
    load_in_8bit: bool = True,
    tokenizer_kwargs: Optional[dict] = None,
):
    """A causal-LM classifier (the HarmBench judge) + its tokenizer.

    8-bit loading uses ``BitsAndBytesConfig``, which is what transformers
    builds internally from the older ``load_in_8bit=True`` kwarg.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    revision = _revision(model_id, revision)
    kwargs: dict[str, Any] = {"revision": revision}
    if device_map is not None:
        kwargs["device_map"] = device_map
    if load_in_8bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, **(tokenizer_kwargs or {}))
    return model, tokenizer


def load_pretraining_encoder(
    model_path: str,
    revision: Optional[str] = None,
    torch_dtype: Any = "float16",
    low_cpu_mem_usage: bool = True,
    to_device=None,
):
    """A vision-language model used as an embedding encoder (CIDER) + its
    tokenizer and image processor."""
    from transformers import AutoImageProcessor, AutoModelForPreTraining, AutoTokenizer

    revision = _revision(model_path, revision)
    model = AutoModelForPreTraining.from_pretrained(
        model_path, revision=revision, torch_dtype=_dtype(torch_dtype),
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    if to_device is not None:
        model = model.to(to_device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision)
    image_processor = AutoImageProcessor.from_pretrained(model_path, revision=revision)
    return model, tokenizer, image_processor


def load_image_classifier(model_path: str, revision: Optional[str] = None):
    """A ViT image classifier (CIDER's denoiser classifier)."""
    from transformers import ViTForImageClassification

    return ViTForImageClassification.from_pretrained(model_path, revision=_revision(model_path, revision))


def load_torch_checkpoint(path_or_file, map_location=None, **kwargs):
    """Weights / state dict from a ``.pt`` checkpoint (path or file-like), e.g.
    CIDER's diffusion denoiser.

    Extra kwargs go straight to ``torch.load`` so each caller keeps its own
    semantics: CIDER passes ``weights_only=False`` (upstream behaviour -- only
    load trusted checkpoints), while callers that pass nothing get torch's
    default (``weights_only=True`` since torch 2.6).
    """
    import torch

    return torch.load(path_or_file, map_location=map_location, **kwargs)


def load_spacy(name: str):
    """A spaCy pipeline (JailGuard's similarity metric). Its version is fixed by
    the installed spaCy model package rather than an HF revision."""
    import spacy

    return spacy.load(name)
