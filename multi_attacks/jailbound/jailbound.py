"""JailBound (NeurIPS 2025) clean-room implementation from arXiv v3.

JailBound has two stages:

1. Safety Boundary Probing trains one logistic classifier per selected fusion
   layer and derives the normal vector and mean crossing distance (Algorithm 1).
2. Safety Boundary Crossing jointly updates an L_inf-bounded image perturbation
   and a discrete token suffix using alignment, semantic, and geometric losses
   (Algorithm 2).

No author code or public repository was available when this port was written.
Consequently it is marked ``adapted``: the paper's stated equations and
hyperparameters are implemented, while unstated sequence pooling, the suffix
regularizer, probe-pair construction, and the simulator's LLaVA-1.5 backend are
explicit clean-room choices rather than claimed official details.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from typing import Any

import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from core.attack import Attack, normalized_instruction_hash
from core.paths import portable_path, resolve_path
from core.registry import register_attack
from core.schema import Behavior, TestCase
from multimodalmodels.registry import LLAVA_15_HF_ID, LLAVA_15_REVISION

from .jailbound_utils import boundary_losses, fit_linear_boundaries, parse_layer_selection


@register_attack("JailBound")
class JailBoundAttack(Attack):
    official_repo = "https://arxiv.org/abs/2505.19610"
    official_commit = "v3-2026-02-25"
    algorithm_fidelity = "adapted"
    substitutes_model = True
    requires_images = True
    requires_surrogate = True

    def __init__(self, config=None):
        super().__init__(config)
        c = self.config
        self.backend = str(c.get("backend", "llava")).lower()
        self.source_model = str(c.get("source_model", LLAVA_15_HF_ID))
        self.source_model_revision = c.get("source_model_revision", LLAVA_15_REVISION)
        self.device = c.get("device")
        self.torch_dtype = str(c.get("torch_dtype", "bfloat16"))

        self.probe_dataset = str(c.get("probe_dataset", "MM-SafetyBench"))
        self.probe_rows = int(c.get("probe_rows", 64))
        self.probe_split = str(c.get("probe_split", "stratified"))
        self.probe_image_type = str(c.get("probe_image_type", "SD"))
        self.probe_unsafe_field = str(c.get("probe_unsafe_field", "rephrased_question_sd"))
        self.probe_safe_field = str(c.get("probe_safe_field", "changed_question"))
        self.attack_image_type = str(c.get("attack_image_type", "SD"))
        self.boundary_artifact = c.get("boundary_artifact")

        self.layers_config = c.get("layers", "all")
        self.representation_pooling = "last_non_padding_input_token"
        self.p0 = float(c.get("p0", 0.3))
        self.classifier_max_iter = int(c.get("classifier_max_iter", 1000))
        self.classifier_c = float(c.get("classifier_c", 1.0))

        self.lambda_1 = float(c.get("lambda_1", 2.0))
        self.lambda_2 = float(c.get("lambda_2", 1.0))
        self.eta_v = float(c.get("eta_v", 0.001))
        self.eta_t = float(c.get("eta_t", 0.0005))
        self.suffix_length = int(c.get("suffix_length", 20))
        self.epsilon = float(c.get("epsilon", 8.0)) / 255.0
        self.iterations = int(c.get("iterations", 100))
        self.vocab_chunk_size = int(c.get("vocab_chunk_size", 4096))
        self.seed = int(c.get("seed", 0))
        self.source_step = c.get("source_step", 0)
        self.suffix_regularizer = "conditional_next_token_nll"
        self.geo_sign_convention = "paper_v3_positive_v"

        if self.backend != "llava":
            raise ValueError("JailBound currently supports backend: llava")
        if self.torch_dtype not in {"bfloat16", "float32"}:
            raise ValueError("JailBound torch_dtype must be bfloat16 or float32")
        if self.probe_rows <= 0:
            raise ValueError("JailBound probe_rows must be positive (reserve a train split)")
        if self.probe_split not in {"stratified", "first"}:
            raise ValueError("JailBound probe_split must be stratified or first")
        if not 0.0 < self.p0 < 1.0:
            raise ValueError("JailBound p0 must be in (0, 1)")
        if any(value <= 0 for value in (
            self.classifier_max_iter, self.classifier_c, self.lambda_1,
            self.lambda_2, self.eta_v, self.eta_t, self.suffix_length,
            self.epsilon, self.iterations, self.vocab_chunk_size,
        )):
            raise ValueError("JailBound optimization settings must be positive")

        self._adapter = None
        self._probe_behaviors: list[Behavior] | None = None
        self._layer_indexes: list[int] = []
        self._boundaries: dict[int, dict[str, Any]] = {}
        self._boundary_path: str | None = None
        self._probe_metrics: dict[str, Any] = {}
        self._last_optimization: dict[str, Any] = {}

    def effective_config(self):
        return {
            "backend": self.backend,
            "source_model": self.source_model,
            "source_model_revision": self.source_model_revision,
            "torch_dtype": self.torch_dtype,
            "probe_dataset": self.probe_dataset,
            "probe_rows": self.probe_rows,
            "probe_split": self.probe_split,
            "probe_image_type": self.probe_image_type,
            "probe_unsafe_field": self.probe_unsafe_field,
            "probe_safe_field": self.probe_safe_field,
            "attack_image_type": self.attack_image_type,
            "layers": self.layers_config,
            "representation_pooling": self.representation_pooling,
            "p0": self.p0,
            "classifier_max_iter": self.classifier_max_iter,
            "classifier_c": self.classifier_c,
            "lambda_1": self.lambda_1,
            "lambda_2": self.lambda_2,
            "eta_v": self.eta_v,
            "eta_t": self.eta_t,
            "suffix_length": self.suffix_length,
            "epsilon": self.epsilon,
            "iterations": self.iterations,
            "vocab_chunk_size": self.vocab_chunk_size,
            "suffix_regularizer": self.suffix_regularizer,
            "geo_sign_convention": self.geo_sign_convention,
            "seed": self.seed,
            "source_step": self.source_step,
            "exclude_train_from_eval": self.exclude_train_from_eval,
        }

    def provenance(self):
        value = super().provenance()
        value.update({
            "probe_metrics": self._probe_metrics,
            "selected_layers": self._layer_indexes,
            "paper_code_available": False,
            "paper_reproduction": False,
        })
        return value

    def _load_adapter(self):
        if self._adapter is None:
            from .jailbound_adapter import LlavaJailBoundAdapter

            self._adapter = LlavaJailBoundAdapter(
                self.source_model,
                revision=self.source_model_revision,
                device=self.device,
                torch_dtype=self.torch_dtype,
            )
        return self._adapter

    def _training_corpus(self) -> list[Behavior]:
        if self._probe_behaviors is not None:
            return self._probe_behaviors
        import core.datasets  # noqa: F401
        from core import get_dataset

        candidates = [b for b in get_dataset(self.probe_dataset)().load() if b.images]
        usable = [
            b for b in candidates
            if self._field(b, self.probe_unsafe_field) and self._field(b, self.probe_safe_field)
        ]
        if not usable:
            raise ValueError(
                f"JailBound probe dataset {self.probe_dataset!r} has no paired "
                f"{self.probe_unsafe_field!r}/{self.probe_safe_field!r} prompts"
            )
        limit = min(self.probe_rows, len(usable))
        if self.probe_split == "first":
            selected = usable[:limit]
        else:
            groups: dict[str, list[Behavior]] = {}
            for behavior in usable:
                groups.setdefault(behavior.category or "", []).append(behavior)
            rng = random.Random(self.seed)
            for values in groups.values():
                rng.shuffle(values)
            selected = []
            while len(selected) < limit:
                progressed = False
                for category in sorted(groups):
                    if groups[category]:
                        selected.append(groups[category].pop())
                        progressed = True
                        if len(selected) == limit:
                            break
                if not progressed:
                    break
        self._probe_behaviors = selected
        return selected

    @staticmethod
    def _field(behavior: Behavior, name: str) -> str:
        if hasattr(behavior, name):
            value = getattr(behavior, name)
            if value:
                return str(value)
        value = behavior.meta.get(name)
        return str(value) if value else ""

    def _image_path(self, behavior: Behavior, image_type: str) -> str:
        by_type = behavior.meta.get("images_by_type", {})
        if image_type and isinstance(by_type, dict) and by_type.get(image_type):
            return str(by_type[image_type])
        index = len(behavior.images) - 1 if self.source_step == "last" else int(self.source_step)
        if index < 0:
            index += len(behavior.images)
        if not 0 <= index < len(behavior.images):
            raise IndexError(
                f"JailBound image step {self.source_step!r} outside 0..{len(behavior.images) - 1}"
            )
        return behavior.images[index]

    def training_behavior_ids(self) -> list[str]:
        return [behavior.behavior_id for behavior in self._training_corpus()]

    def training_instruction_hashes(self) -> set[str]:
        hashes = set()
        for behavior in self._training_corpus():
            hashes.add(normalized_instruction_hash(self._field(behavior, self.probe_unsafe_field)))
            hashes.add(normalized_instruction_hash(self._field(behavior, self.probe_safe_field)))
        return hashes

    def training_behavior_fingerprints(self) -> set[str]:
        fingerprints = set()
        for behavior in self._training_corpus():
            payload = {
                "behavior_id": behavior.behavior_id,
                "unsafe": self._field(behavior, self.probe_unsafe_field),
                "safe": self._field(behavior, self.probe_safe_field),
                "image_type": self.probe_image_type,
                "image": self._image_path(behavior, self.probe_image_type),
            }
            blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
            fingerprints.add(hashlib.sha256(blob).hexdigest())
        return fingerprints

    def _lineage_inputs(self):
        return {
            f"probe_image_{index:04d}": resolve_path(
                self._image_path(behavior, self.probe_image_type)
            )
            for index, behavior in enumerate(self._training_corpus())
        }

    def _lineage_code_files(self):
        root = os.path.dirname(__file__)
        return super()._lineage_code_files() + [
            os.path.join(root, "jailbound_adapter.py"),
            os.path.join(root, "jailbound_utils.py"),
        ]

    def _probe(self) -> dict[int, dict[str, Any]]:
        adapter = self._load_adapter()
        self._layer_indexes = parse_layer_selection(self.layers_config, adapter.n_layers)
        rows: dict[int, list[torch.Tensor]] = {layer: [] for layer in self._layer_indexes}
        labels: list[int] = []
        from tqdm import tqdm

        examples = []
        for behavior in self._training_corpus():
            image_path = resolve_path(self._image_path(behavior, self.probe_image_type))
            examples.extend([
                (image_path, self._field(behavior, self.probe_safe_field), 0),
                (image_path, self._field(behavior, self.probe_unsafe_field), 1),
            ])
        for image_path, prompt, label in tqdm(examples, desc="JailBound probing"):
            image = Image.open(image_path).convert("RGB")
            image01 = adapter.prepare_image(image)
            context = adapter.prepare_context(image, prompt)
            features = adapter.extract_features(image01, context, self._layer_indexes)
            for layer in self._layer_indexes:
                rows[layer].append(features[layer].detach().cpu())
            labels.append(label)
        stacked = {layer: torch.stack(values) for layer, values in rows.items()}
        boundaries = fit_linear_boundaries(
            stacked,
            torch.tensor(labels),
            threshold=self.p0,
            max_iter=self.classifier_max_iter,
            classifier_c=self.classifier_c,
            seed=self.seed,
        )
        accuracies = [float(item["training_accuracy"]) for item in boundaries.values()]
        self._probe_metrics = {
            "n_behaviors": len(self._training_corpus()),
            "n_labeled_pairs": len(labels),
            "n_layers": len(boundaries),
            "min_training_accuracy": min(accuracies),
            "mean_training_accuracy": sum(accuracies) / len(accuracies),
        }
        return boundaries

    def _save_boundaries(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "schema_version": 1,
            "layers": self._layer_indexes,
            "boundaries": self._boundaries,
            "probe_metrics": self._probe_metrics,
            "effective_config": self.effective_config(),
        }, path)

    def _load_boundaries(self, path: str) -> None:
        from multimodalmodels.loaders import load_torch_checkpoint

        payload = load_torch_checkpoint(path, map_location="cpu", weights_only=True)
        if payload.get("schema_version") != 1 or not payload.get("boundaries"):
            raise ValueError(f"Invalid JailBound boundary artifact: {path}")
        self._layer_indexes = [int(value) for value in payload["layers"]]
        self._boundaries = {int(key): value for key, value in payload["boundaries"].items()}
        self._probe_metrics = dict(payload.get("probe_metrics", {}))

    def prepare(self, out_dir: str) -> None:
        if self._boundaries:
            return
        if self.boundary_artifact:
            artifact = resolve_path(self.boundary_artifact)
            if not os.path.isfile(artifact):
                raise FileNotFoundError(f"JailBound boundary artifact not found: {artifact}")
            if not self.verify_lineage(artifact):
                raise ValueError(
                    "JailBound boundary lineage is missing or does not match this config/data/code"
                )
            self._load_boundaries(artifact)
            self._boundary_path = artifact
            return
        self._boundaries = self._probe()
        self._boundary_path = os.path.join(out_dir, "jailbound_boundaries.pt")
        self._save_boundaries(self._boundary_path)
        self.write_lineage(self._boundary_path)

    def _optimize(self, image: Image.Image, instruction: str):
        adapter = self._load_adapter()
        image0 = adapter.prepare_image(image).detach()
        clean_context = adapter.prepare_context(image, instruction)
        clean = adapter.extract_features(image0, clean_context, self._layer_indexes)

        generator = torch.Generator(device=adapter.device)
        generator.manual_seed(self.seed)
        suffix_ids = adapter.random_suffix(self.suffix_length, generator)
        adv_context = adapter.prepare_context(image, instruction, suffix_ids)
        delta = torch.zeros_like(image0)
        history: list[float] = []

        from tqdm import tqdm

        progress = tqdm(range(self.iterations), desc="JailBound crossing")
        last_parts = {}
        for _ in progress:
            image_adv = (image0 + delta).clamp(0, 1).detach().requires_grad_(True)
            joint = adapter.forward_joint(
                image_adv, adv_context, suffix_ids, self._layer_indexes
            )
            align, geo = boundary_losses(clean, joint.features, self._boundaries)
            image_sem = (image_adv - image0).square().sum()
            semantic = image_sem + joint.fluency_loss
            total = align + self.lambda_1 * semantic + self.lambda_2 * geo
            if not torch.isfinite(total):
                raise RuntimeError("JailBound optimization produced a non-finite loss")
            grad_image, grad_suffix = torch.autograd.grad(
                total, (image_adv, joint.suffix_embeddings)
            )
            if grad_image is None or grad_suffix is None:
                raise RuntimeError("JailBound loss is disconnected from image or suffix")
            with torch.no_grad():
                candidate = image_adv - self.eta_v * grad_image
                delta = (candidate - image0).clamp(-self.epsilon, self.epsilon)
                delta = (image0 + delta).clamp(0, 1) - image0
                desired = joint.suffix_embeddings - self.eta_t * grad_suffix
                suffix_ids = adapter.nearest_tokens(desired, self.vocab_chunk_size)
            value = float(total.detach().cpu())
            history.append(value)
            last_parts = {
                "loss": value,
                "align_loss": float(align.detach().cpu()),
                "semantic_loss": float(semantic.detach().cpu()),
                "geo_loss": float(geo.detach().cpu()),
            }
            progress.set_postfix(loss=f"{value:.4f}")

        final_image = (image0 + delta).detach().cpu().float().clamp(0, 1)
        suffix = adapter.decode_suffix(suffix_ids)
        self._last_optimization = {
            "iterations": self.iterations,
            "initial_loss": history[0] if history else None,
            "final_step_loss": history[-1] if history else None,
            "max_abs_delta": float(delta.detach().abs().max().cpu()),
            "suffix_token_count": int(suffix_ids.numel()),
            **last_parts,
        }
        return final_image, suffix

    def build(self, behavior: Behavior, out_dir: str) -> list[TestCase]:
        if not self._boundaries:
            self.prepare(out_dir)
        source = resolve_path(self._image_path(behavior, self.attack_image_type))
        image = Image.open(source).convert("RGB")
        adversarial, suffix = self._optimize(image, behavior.instruction)
        os.makedirs(out_dir, exist_ok=True)
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", behavior.behavior_id)
        output = os.path.join(out_dir, f"{safe_id}_jailbound.png")
        to_pil_image(adversarial.squeeze(0)).save(output)
        text = behavior.instruction + ((" " + suffix) if suffix else "")
        return [TestCase(
            test_case_id=self.test_case_id(self.name, behavior.behavior_id),
            behavior_id=behavior.behavior_id,
            attack=self.name,
            text=text,
            images=[portable_path(output)],
            meta={
                "source_image_type": self.attack_image_type,
                "joint_image_text_optimization": True,
                "boundary_layers": self._layer_indexes,
                "boundary_artifact": portable_path(self._boundary_path) if self._boundary_path else None,
                "epsilon_255": self.epsilon * 255.0,
                **self._last_optimization,
            },
        )]
