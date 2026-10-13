"""Model loading is centralized in multimodalmodels/ (and stays that way).

Checks, without downloading or loading any real weights:
  1. No code outside multimodalmodels/ loads a model itself (AST scan for
     ``*.from_pretrained`` / ``torch.load`` / ``spacy.load`` /
     ``timm.create_model``); the in-use wrappers go through the loaders too.
  2. Every rerouted call site passes the same arguments as before (pinned
     revision, dtype, device placement, processor/tokenizer kwargs), using fake
     ``from_pretrained`` methods.
  3. ``import multimodalmodels`` / ``import core.attacks`` stay lazy (no
     transformers import) while ``from multimodalmodels import X`` still works.
  4. Pinned revisions in registry.py match the local HF cache snapshots (a
     model that is not cached here is skipped).

Run with:  python -m core.tests.test_model_loading
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import warnings
from contextlib import contextmanager
from types import SimpleNamespace

from core import REPO_ROOT

# Archived pre-refactor code that nothing imports; everything else is scanned.
_SKIP_DIRS = {".git", "__pycache__", "multimodalmodels", "legacy_attacks", "runs", "data"}


def _model_load_calls(tree):
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        attr, value = node.func.attr, node.func.value
        base = value.id if isinstance(value, ast.Name) else None
        if (attr == "from_pretrained"
                or (attr == "load" and base in {"torch", "th", "spacy"})
                or (attr == "create_model" and base == "timm")):
            yield node.lineno, ast.unparse(node.func)


def _scan_outside_hub():
    hits = []
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            with open(path, encoding="utf-8") as f, warnings.catch_warnings():
                # vendored files carry old invalid-escape strings; parsing them
                # here should not emit their warnings
                warnings.simplefilter("ignore")
                tree = ast.parse(f.read(), path)
            for lineno, call in _model_load_calls(tree):
                hits.append(f"{os.path.relpath(path, REPO_ROOT)}:{lineno} {call}")
    return hits


def _class_load_calls(path, class_name):
    with open(os.path.join(REPO_ROOT, path), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == class_name)
    return list(_model_load_calls(cls))


class _Recorder:
    """Stands in for a model/processor: records .to()/.eval() and returns self."""

    def __init__(self, name, args, kwargs):
        self.name, self.args, self.kwargs, self.moved_to, self.evaluated = name, args, kwargs, None, False

    def to(self, device):
        self.moved_to = device
        return self

    def eval(self):
        self.evaluated = True
        return self


@contextmanager
def _fake_from_pretrained(*classes):
    """Replace ``from_pretrained`` on each class with a recorder (restored after)."""
    saved = {}
    for cls in classes:
        saved[cls] = cls.__dict__.get("from_pretrained")
        setattr(cls, "from_pretrained",
                (lambda n: lambda *a, **k: _Recorder(n, a, k))(cls.__name__))
    try:
        yield
    finally:
        for cls, original in saved.items():
            if original is None:
                delattr(cls, "from_pretrained")
            else:
                setattr(cls, "from_pretrained", original)


@contextmanager
def _patched(module, name, value):
    original = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, original)


def check_centralized() -> None:
    hits = _scan_outside_hub()
    assert not hits, ("model loaded outside multimodalmodels/ -- route it through "
                      "multimodalmodels.loaders:\n  " + "\n  ".join(hits))
    # the wrappers the pipeline actually uses also go through the loaders
    assert not _class_load_calls("multimodalmodels/llava/llava_model.py", "LLaVA_v1_5")
    assert not _class_load_calls("multimodalmodels/multimodalmodel.py", "HarmBenchMultiModalModel")


def check_loader_arguments() -> None:
    import torch
    import transformers as tf

    import multimodalmodels.loaders as L
    from multimodalmodels.registry import (HARMBENCH_CLS_REVISION, LLAVA_15_REVISION,
                                           QWEN2_VL_REVISION)

    classes = (tf.LlavaForConditionalGeneration, tf.AutoProcessor,
               tf.Qwen2VLForConditionalGeneration, tf.Qwen2_5_VLForConditionalGeneration,
               tf.AutoModelForCausalLM, tf.AutoTokenizer)
    with _fake_from_pretrained(*classes):
        # LLaVA: pinned revision by default; eval-target settings
        model, proc = L.load_llava()
        assert model.name == "LlavaForConditionalGeneration"
        assert model.args == ("llava-hf/llava-1.5-7b-hf",)
        assert model.kwargs == {"revision": LLAVA_15_REVISION, "torch_dtype": torch.float16,
                                "device_map": "auto"}
        assert proc.kwargs == {"revision": LLAVA_15_REVISION}

        # LLaVA: FORCE's single-device + eval settings, processor kwargs passthrough
        model, proc = L.load_llava(device_map=None, to_device="cpu", eval_mode=True,
                                   processor_kwargs={"num_additional_image_tokens": 1})
        assert "device_map" not in model.kwargs and model.moved_to == "cpu" and model.evaluated
        assert proc.kwargs == {"revision": LLAVA_15_REVISION, "num_additional_image_tokens": 1}

        # an explicit revision wins; a local path is not pinned
        assert L.load_llava(revision="abc")[0].kwargs["revision"] == "abc"
        assert L.load_llava("/models/local-llava")[0].kwargs["revision"] is None

        # Qwen2-VL / Qwen2.5-VL
        model, proc = L.load_qwen2vl()
        assert model.name == "Qwen2VLForConditionalGeneration"
        assert model.kwargs == {"revision": QWEN2_VL_REVISION, "torch_dtype": "auto",
                                "device_map": "auto"}
        assert proc.kwargs == {"revision": QWEN2_VL_REVISION, "trust_remote_code": True}
        model, _ = L.load_qwen2vl("Qwen/Qwen2.5-VL-7B-Instruct")
        assert model.name == "Qwen2_5_VLForConditionalGeneration"
        assert model.kwargs["revision"] is None  # registered but not pinned yet

        # HarmBench judge: pinned, 8-bit via BitsAndBytesConfig, tokenizer kwargs
        model, tok = L.load_causal_lm_classifier(
            tokenizer_kwargs={"use_fast": False, "padding_side": "left"})
        assert model.kwargs["revision"] == HARMBENCH_CLS_REVISION
        assert model.kwargs["device_map"] == "auto"
        assert model.kwargs["quantization_config"].load_in_8bit is True
        assert tok.kwargs == {"revision": HARMBENCH_CLS_REVISION, "use_fast": False,
                              "padding_side": "left"}

    # call sites forward the same settings they used before the refactor
    calls = []

    def fake_load_llava(*a, **k):
        calls.append(("load_llava", a, k))
        fake_model = SimpleNamespace(gradient_checkpointing_enable=lambda: None,
                                     config=SimpleNamespace(use_cache=True), device="cpu")
        fake_proc = SimpleNamespace(image_processor=SimpleNamespace(
            image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5]))
        return fake_model, fake_proc

    import multimodalmodels.llava.llava_model as llava_model
    with _patched(llava_model, "load_llava", fake_load_llava):
        wrapper = llava_model.LLaVA_v1_5()
    assert calls[-1] == ("load_llava", ("llava-hf/llava-1.5-7b-hf",), {
        "revision": LLAVA_15_REVISION, "torch_dtype": "float16", "device_map": "auto",
        "processor_kwargs": {"num_additional_image_tokens": 1}})
    assert wrapper.model.config.use_cache is False  # gradient-checkpointing setup kept

    import core.attacks  # noqa: F401
    from core import get_attack
    with _patched(L, "load_llava", fake_load_llava):
        force = get_attack("FORCE")({})
        force._load_surrogate()
    name, args, kwargs = calls[-1]
    assert args == ("llava-hf/llava-1.5-7b-hf",)
    assert kwargs["revision"] == LLAVA_15_REVISION and kwargs["torch_dtype"] == torch.float16
    assert kwargs["device_map"] is None and kwargs["eval_mode"] is True  # hooks need one device

    judge_calls = []
    with _patched(L, "load_causal_lm_classifier",
                  lambda *a, **k: judge_calls.append((a, k)) or (None, None)):
        from multimodalmodels.multimodalmodel import HarmBenchMultiModalModel
        HarmBenchMultiModalModel("cais/HarmBench-Llama-2-13b-cls")
    assert judge_calls == [(("cais/HarmBench-Llama-2-13b-cls",), {
        "device_map": "auto", "load_in_8bit": True,
        "tokenizer_kwargs": {"use_fast": False, "truncation_side": "left",
                             "padding_side": "left"}})]

    qwen_calls = []
    with _patched(L, "load_qwen2vl",
                  lambda *a, **k: qwen_calls.append((a, k)) or (None, None)):
        from multi_attacks.ultrabreak.ultrabreak_adapter import Qwen2Adapter
        Qwen2Adapter("Qwen/Qwen2-VL-7B-Instruct", revision=QWEN2_VL_REVISION)
    assert qwen_calls == [(("Qwen/Qwen2-VL-7B-Instruct",), {"revision": QWEN2_VL_REVISION})]


def check_lazy_package() -> None:
    probe = (
        "import sys\n"
        "import multimodalmodels, core.attacks\n"
        "assert 'transformers' not in sys.modules, 'package import pulled in transformers'\n"
        "from multimodalmodels import LLAVA_15_REVISION, get_spec\n"
        "assert get_spec('llava-hf/llava-1.5-7b-hf').revision == LLAVA_15_REVISION\n"
        "from multimodalmodels import LLaVA_v1_5, HarmBenchMultiModalModel, load_llava\n"
        "assert isinstance(LLaVA_v1_5, type) and callable(load_llava)\n"
        "assert getattr(multimodalmodels, 'LLaVA_v1_5') is LLaVA_v1_5\n"
        "assert {'LLaVA_v1_5', 'HarmBenchMultiModalModel', 'load_llava'} <= set(dir(multimodalmodels))\n"
        "try:\n"
        "    multimodalmodels.NoSuchModel\n"
        "except AttributeError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('unknown name resolved')\n"
    )
    env = {**os.environ, "PYTHONNOUSERSITE": "1"}
    r = subprocess.run([sys.executable, "-c", probe], cwd=REPO_ROOT, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]

    # an optional wrapper whose dependency is missing is absent (AttributeError),
    # a required one fails loudly (ImportError)
    import multimodalmodels as mm
    mm._LAZY["_OptionalMissing"] = (".does_not_exist", "X", True)
    mm._LAZY["_RequiredMissing"] = (".does_not_exist", "X", False)
    try:
        try:
            mm._OptionalMissing
        except AttributeError:
            pass
        else:
            raise AssertionError("missing optional wrapper should be absent")
        try:
            mm._RequiredMissing
        except ImportError:
            pass
        else:
            raise AssertionError("missing required module should raise ImportError")
    finally:
        del mm._LAZY["_OptionalMissing"], mm._LAZY["_RequiredMissing"]


def check_pins_match_local_cache() -> list[str]:
    from multimodalmodels.registry import MODEL_SPECS

    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    checked = []
    for spec in MODEL_SPECS.values():
        if spec.revision is None:
            continue
        repo_dir = os.path.join(hub, "models--" + spec.hf_id.replace("/", "--"))
        if not os.path.isdir(repo_dir):
            continue  # not cached on this machine
        snap = os.path.join(repo_dir, "snapshots", spec.revision)
        assert os.path.isdir(snap), (f"{spec.hf_id}: pinned revision {spec.revision} is not "
                                     f"in the local cache ({os.listdir(os.path.join(repo_dir, 'snapshots'))})")
        checked.append(spec.hf_id)
    return checked


def main() -> None:
    check_centralized()
    check_loader_arguments()
    check_lazy_package()
    cached = check_pins_match_local_cache()
    print("model loading OK: no loads outside multimodalmodels/, call sites keep their "
          f"settings, package import is lazy, pins match cache for {cached}")


if __name__ == "__main__":
    main()
