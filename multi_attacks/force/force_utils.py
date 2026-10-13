"""White-box optimization helpers for the FORCE attack.

Ported from the official FORCE implementation (tmllab/2026_CVPR_FORCE, utils.py),
MIT License. Kept behavior-identical; only the unused eval/npy helpers are
dropped. These operate on a LLaVA surrogate with gradients and forward hooks, so
unlike the rest of core they do import torch.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import amp
from torchvision.transforms.functional import to_pil_image, to_tensor


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pil_to_tensor(image):
    return to_tensor(image)


def tensor_to_pil(tensor):
    tensor = tensor.detach().cpu().clamp(0, 1)
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return to_pil_image(tensor)


def save_tensor_as_image(tensor, save_path):
    import os

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    to_pil_image(tensor.detach().cpu().float().clamp(0, 1)).save(save_path)


def generate_frequency_masks(H, W, K=5, device="cpu"):
    y = torch.arange(-H // 2, H // 2, device=device).reshape(-1, 1).expand(H, W)
    x = torch.arange(-W // 2, W // 2, device=device).reshape(1, -1).expand(H, W)
    r = torch.sqrt(x**2 + y**2)
    r_max = r.max()
    bounds = torch.linspace(0, r_max, K + 1, device=device)
    return [(r >= bounds[i]) & (r < bounds[i + 1]) for i in range(K)]


def per_sample_ce_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    B, Tm1, V = shift_logits.shape
    loss_flat = F.cross_entropy(
        shift_logits.view(-1, V), shift_labels.view(-1),
        reduction="none", ignore_index=ignore_index,
    ).view(B, Tm1)
    valid = (shift_labels != ignore_index)
    denom = valid.sum(dim=1).clamp_min(1)
    return (loss_flat * valid).sum(dim=1) / denom


def normalize_pixel_values_on_gpu(x: torch.Tensor, img_proc, out_dtype=torch.float16):
    """Normalize x ([B,3,H,W] in [0,1]) by the image processor mean/std; no resize/crop."""
    mean = torch.tensor(getattr(img_proc, "image_mean", [0.5, 0.5, 0.5]), device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(getattr(img_proc, "image_std", [0.5, 0.5, 0.5]), device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return ((x - mean) / std).to(out_dtype)


def register_llm_hidden_hooks(model, require_grad=False, move_to_cpu=False, check_finite=False):
    """Hook every language-model layer, capturing its hidden state into a dict."""
    features: dict[int, torch.Tensor] = {}
    handles = []

    def hook_fn(module, inp, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if check_finite and not torch.isfinite(hidden).all():
            return
        if not require_grad:
            hidden = hidden.detach()
        if move_to_cpu:
            hidden = hidden.to("cpu", non_blocking=True)
        features[module.layer_idx] = hidden

    if hasattr(model.language_model, "model"):
        layers = model.language_model.model.layers
    elif hasattr(model.language_model, "layers"):
        layers = model.language_model.layers
    else:
        raise AttributeError("Cannot locate transformer layers in model.language_model")

    for i, layer in enumerate(layers):
        layer.layer_idx = i
        handles.append(layer.register_forward_hook(hook_fn))
    return handles, features


def generate_response(model, processor, image_tensor, query):
    conversation = [{"role": "user", "content": [{"type": "text", "text": query}, {"type": "image"}]}]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    if image_tensor.ndim == 4 and image_tensor.shape[0] == 1:
        image_tensor = image_tensor.squeeze(0)
    pil_image = to_pil_image(image_tensor.cpu().float().clamp(0, 1))
    inputs = processor(images=pil_image, text=prompt, return_tensors="pt")
    for k, v in inputs.items():
        inputs[k] = v.to(model.device).half() if v.dtype == torch.float32 else v.to(model.device)
    with amp.autocast(device_type="cuda", dtype=torch.float16):
        output_ids = model.generate(
            **inputs, do_sample=False, max_new_tokens=1024,
            eos_token_id=processor.tokenizer.eos_token_id, early_stopping=False,
        )
    input_token_len = inputs["input_ids"].shape[1]
    return processor.tokenizer.decode(output_ids[0, input_token_len:], skip_special_tokens=True).strip()
