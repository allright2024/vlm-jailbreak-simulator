"""multimodalmodels: the one place where every model in this repo is loaded.

* ``registry``  -- pinned model ids/revisions (pure data, imported eagerly).
* ``loaders``   -- the loading helpers every caller goes through.
* wrappers      -- LLaVA_v1_5, HarmBenchMultiModalModel, ... (model + I/O logic).

Wrappers and loaders are resolved lazily on first attribute access (PEP 562),
so importing this package -- or reading a pinned revision from ``registry`` --
does not pull in transformers or the optional wrappers' heavy dependencies.
``from multimodalmodels import LLaVA_v1_5``, ``getattr(multimodalmodels, name)``
and ``dir(multimodalmodels)`` behave as before; an optional wrapper whose
dependencies are missing is simply absent, as it was with the old eager
``try: from .x import * except ImportError: pass``.
"""

from __future__ import annotations

import importlib

from .registry import (HARMBENCH_CLS_HF_ID, HARMBENCH_CLS_REVISION,  # noqa: F401
                       LLAVA_15_HF_ID, LLAVA_15_REVISION, MODEL_SPECS,
                       QWEN25_VL_HF_ID, QWEN2_VL_HF_ID, QWEN2_VL_REVISION,
                       VIT_BASE_HF_ID, ModelSpec, default_revision, get_spec)

# name -> (module, attribute, optional). Optional wrappers need heavy or
# environment-specific dependencies (LAVIS-style InstructBLIP, Qwen-VL custom
# code, omegaconf for MiniGPT-4).
_LAZY: dict[str, tuple[str, str, bool]] = {
    "LLaVA_v1_5": (".llava", "LLaVA_v1_5", False),
    "LLaVA_v1_6": (".llava", "LLaVA_v1_6", False),
    "MultiModalModel": (".multimodalmodel", "MultiModalModel", False),
    "HarmBenchMultiModalModel": (".multimodalmodel", "HarmBenchMultiModalModel", False),
    "InstructBLIP": (".instructblip", "InstructBLIP", True),
    "Qwen_VL_Chat": (".qwen", "Qwen_VL_Chat", True),
    "MiniGPT4": (".minigpt4.minigpt4_model", "MiniGPT4", True),
    "load_llava": (".loaders", "load_llava", False),
    "load_qwen2vl": (".loaders", "load_qwen2vl", False),
    "load_causal_lm_classifier": (".loaders", "load_causal_lm_classifier", False),
    "load_pretraining_encoder": (".loaders", "load_pretraining_encoder", False),
    "load_image_classifier": (".loaders", "load_image_classifier", False),
    "load_torch_checkpoint": (".loaders", "load_torch_checkpoint", False),
    "load_spacy": (".loaders", "load_spacy", False),
}


def __getattr__(name: str):
    try:
        module_name, attr, optional = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    try:
        value = getattr(importlib.import_module(module_name, __name__), attr)
    except ImportError as e:
        if optional:
            raise AttributeError(f"{name} is unavailable (missing dependency: {e})") from e
        raise
    globals()[name] = value  # cache: later lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    names = set(globals())
    for name, (_, _, optional) in _LAZY.items():
        if not optional:
            names.add(name)
            continue
        try:
            __getattr__(name)
        except AttributeError:
            continue
        names.add(name)
    return sorted(names)
