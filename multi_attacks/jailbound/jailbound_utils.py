"""Model-independent mathematics for the JailBound clean-room port.

The paper specifies layer-wise logistic probes and three crossing losses, but
does not release implementation code.  Keeping those equations here makes the
paper-derived part independently testable without loading a VLM.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch


def parse_layer_selection(selection: Any, n_layers: int) -> list[int]:
    """Resolve ``all``/``last10``/``last`` or an explicit layer list."""
    if n_layers <= 0:
        raise ValueError("JailBound requires at least one fusion layer")
    if isinstance(selection, str):
        value = selection.strip().lower()
        if value == "all":
            return list(range(n_layers))
        if value == "last10":
            return list(range(max(0, n_layers - 10), n_layers))
        if value == "last":
            return [n_layers - 1]
        try:
            selection = [int(part.strip()) for part in value.split(",") if part.strip()]
        except ValueError as exc:
            raise ValueError(
                "JailBound layers must be all, last10, last, or comma-separated integers"
            ) from exc
    elif isinstance(selection, int):
        selection = [selection]
    elif not isinstance(selection, Iterable):
        raise TypeError("JailBound layers must be a string, integer, or iterable")

    resolved = []
    for raw in selection:
        index = int(raw)
        if index < 0:
            index += n_layers
        if not 0 <= index < n_layers:
            raise IndexError(f"JailBound layer {raw} is outside 0..{n_layers - 1}")
        if index not in resolved:
            resolved.append(index)
    if not resolved:
        raise ValueError("JailBound layer selection is empty")
    return resolved


def fit_linear_boundaries(
    features: dict[int, torch.Tensor],
    labels: torch.Tensor,
    *,
    threshold: float,
    max_iter: int,
    classifier_c: float,
    seed: int,
) -> dict[int, dict[str, Any]]:
    """Fit Eq. (2)-(5) layer-wise logistic probes on CPU.

    ``features[layer]`` is ``[N,D]`` and labels follow the paper's convention:
    safe=0, unsafe=1.  Returned tensors remain on CPU so the artifact is small,
    portable, and safe to load with ``weights_only=True``.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError("JailBound threshold must be in (0, 1)")
    labels = labels.detach().cpu().long()
    if labels.ndim != 1 or set(labels.tolist()) != {0, 1}:
        raise ValueError("JailBound probing needs both safe=0 and unsafe=1 labels")

    from sklearn.linear_model import LogisticRegression

    target_logit = math.log(threshold / (1.0 - threshold))
    result: dict[int, dict[str, Any]] = {}
    for layer, tensor in sorted(features.items()):
        x = tensor.detach().cpu().float()
        if x.ndim != 2 or x.shape[0] != labels.numel():
            raise ValueError(
                f"JailBound layer {layer} feature shape {tuple(x.shape)} "
                f"does not match {labels.numel()} labels"
            )
        classifier = LogisticRegression(
            C=classifier_c,
            max_iter=max_iter,
            random_state=seed,
            solver="lbfgs",
        )
        classifier.fit(x.numpy(), labels.numpy())
        w = torch.from_numpy(classifier.coef_[0]).float()
        b = float(classifier.intercept_[0])
        norm = w.norm(p=2).clamp_min(1e-12)
        scores = x @ w + b
        epsilon = (target_logit - scores).abs().div(norm).mean()
        predicted = torch.from_numpy(classifier.predict(x.numpy())).long()
        result[int(layer)] = {
            "w": w,
            "b": b,
            "v": w / norm,
            "w_norm": float(norm),
            "epsilon": float(epsilon),
            "training_accuracy": float((predicted == labels).float().mean()),
            "n_samples": int(labels.numel()),
        }
    return result


def boundary_losses(
    clean: dict[int, torch.Tensor],
    adversarial: dict[int, torch.Tensor],
    boundaries: dict[int, dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the published alignment and geometric losses (Eq. 6-7).

    The v3 paper/Algorithm 2 uses ``h_target = h - epsilon*v`` but writes the
    geometric term as ``normalize(delta_h) - v``.  Although those signs appear
    contradictory, this function intentionally preserves the published formula
    instead of silently correcting it.
    """
    if not boundaries:
        raise ValueError("JailBound has no trained boundaries")
    align = geo = None
    for layer, boundary in sorted(boundaries.items()):
        if layer not in clean or layer not in adversarial:
            raise KeyError(f"JailBound activation missing for layer {layer}")
        base = clean[layer]
        adv = adversarial[layer]
        direction = boundary["v"].to(device=adv.device, dtype=adv.dtype)
        epsilon = torch.as_tensor(boundary["epsilon"], device=adv.device, dtype=adv.dtype)
        target = base.to(device=adv.device, dtype=adv.dtype) - epsilon * direction
        delta = adv - base.to(device=adv.device, dtype=adv.dtype)
        layer_align = (adv - target).square().sum()
        layer_geo = (delta / delta.norm(p=2).clamp_min(1e-12) - direction).square().sum()
        align = layer_align if align is None else align + layer_align
        geo = layer_geo if geo is None else geo + layer_geo
    assert align is not None and geo is not None
    return align, geo
