"""Pytest entry point that collects the (model-free) main()-based tests.

Each ``core/tests/test_<name>.py`` exposes a ``main()`` that asserts and prints.
This wraps them as parametrized pytest cases so ``pytest`` (and CI) can collect
and report them, without rewriting each file. A gated GPU *smoke* case exercises real Qwen2-VL loading + one CE gradient step
(the transfer-surrogate path) when ``RUN_GPU_TESTS=1``; it is not a full
attack/pipeline end-to-end.

Run:
    PYTHONNOUSERSITE=1 pytest core/tests/test_suite.py
    RUN_GPU_TESTS=1 PYTHONNOUSERSITE=1 pytest core/tests/test_suite.py -m gpu
"""

from __future__ import annotations

import importlib
import os

import pytest

# Code-only tests require neither benchmark data nor model weights.
CODE_ONLY = [
    "test_core_contract",
    "test_pipeline",
    "test_figstep",
    "test_force",
    "test_best_of_n",
    "test_jps",
    "test_model_loading",
    "test_external_data",
]

# These use no model, but validate the separately distributed benchmark tree.
# They run automatically when the complete external data root is present.
EXTERNAL_DATA = [
    "test_datasets",
    "test_jood",
    "test_hades",
    "test_ultrabreak",
    "test_visual_adv",
    "test_delta_jp",
    "test_provenance_split",
]

from core.data import all_data_available

DATA_READY = all_data_available()


@pytest.mark.parametrize("module_name", CODE_ONLY)
def test_code_only(module_name: str) -> None:
    mod = importlib.import_module(f"core.tests.{module_name}")
    mod.main()


@pytest.mark.data
@pytest.mark.skipif(not DATA_READY,
                    reason="external benchmark data is not configured; run validate_data.py")
@pytest.mark.parametrize("module_name", EXTERNAL_DATA)
def test_external_data_suite(module_name: str) -> None:
    mod = importlib.import_module(f"core.tests.{module_name}")
    mod.main()


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("RUN_GPU_TESTS") != "1",
                    reason="set RUN_GPU_TESTS=1 to run GPU smoke tests")
def test_qwen2_ce_transfer_gradient_smoke() -> None:
    """Smoke test: the Qwen2-VL CE surrogate loads and backprops a gradient to
    the image (the transfer path used by VisualAdv/DeltaJP). Not a full E2E."""
    import torch
    from image_attacks.ce_surrogate import load_ce_surrogate

    s = load_ce_surrogate("qwen2", "Qwen/Qwen2-VL-7B-Instruct")
    img = torch.rand(1, 3, 336, 336, device=s.device, requires_grad=True)
    loss = s.compute_loss_batch(img, ["Sure, here is how to do it"], None)
    loss.backward()
    assert img.grad is not None and float(img.grad.abs().sum()) > 0


@pytest.mark.gpu
@pytest.mark.skipif(os.environ.get("RUN_GPU_TESTS") != "1",
                    reason="set RUN_GPU_TESTS=1 to run GPU smoke tests")
def test_centralized_loaders_real_weights_smoke() -> None:
    """Smoke test: the real weights load through multimodalmodels with the
    expected placement -- the eval target (fp16, sharded), FORCE's surrogate
    (fp16, one device, eval) and the HarmBench judge (8-bit). Models are loaded
    one at a time and freed, to keep peak GPU memory to a single model."""
    import gc

    import torch

    import multimodalmodels.loaders as L
    from multimodalmodels import HarmBenchMultiModalModel, LLaVA_v1_5

    def release():  # call after `del` of the last reference
        gc.collect()
        torch.cuda.empty_cache()

    target = LLaVA_v1_5()
    assert target.model.dtype == torch.float16 and target.model.device.type == "cuda"
    del target
    release()

    model, _ = L.load_llava(device_map=None, to_device="cuda", eval_mode=True)
    assert model.dtype == torch.float16 and not model.training
    assert {p.device.type for p in model.parameters()} == {"cuda"}
    del model, _
    release()

    judge = HarmBenchMultiModalModel("cais/HarmBench-Llama-2-13b-cls")
    assert getattr(judge.model, "is_loaded_in_8bit", False)
    del judge
    release()
