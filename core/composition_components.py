"""Concrete stages used by the composed-jailbreak dataset generator.

The adapters in this module deliberately distinguish two cases:

* ``full stage`` adapters call the existing attack implementation with its
  configured optimization loop intact (JPS, JailBound, FigStep, MML).
* ``decomposed`` adapters extract one operation from a method so that another
  method can consume its output (CS-DJ splitting, HADES top patch, JOOD
  augmentation).  These are always labelled adaptations.

No composed output is an official paper reproduction, even when an individual
source stage uses an official-profile configuration.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from typing import Any, Mapping

from PIL import Image

from .attack import file_sha256
from .composition import (
    ComponentResult,
    ComposedArtifact,
    CompositionComponent,
    strict_image_hashes,
)
from .paths import portable_path, resolve_path
from .schema import Behavior


def _safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "sample"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    return f"{stem[:64]}_{digest}"


def _selection_index(tag: str, count: int) -> int:
    if count <= 0:
        raise ValueError("cannot select from an empty candidate collection")
    return int.from_bytes(hashlib.sha256(tag.encode("utf-8")).digest()[:8], "big") % count


def _bottom_update_mask_like(tensor, preserve_top_fraction: float):
    """Binary mask that exactly preserves a preceding top-region component."""

    import torch

    fraction = float(preserve_top_fraction)
    if not 0 <= fraction < 1:
        raise ValueError("preserve_top_fraction must be in [0, 1)")
    mask = torch.ones_like(tensor)
    if fraction:
        rows = max(1, int(mask.shape[-2] * fraction))
        mask[..., :rows, :] = 0
    return mask


def _input_image(artifact: ComposedArtifact, behavior: Behavior) -> str:
    images = artifact.images or behavior.images
    if not images:
        raise ValueError("composition stage requires an input image")
    path = resolve_path(images[0])
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def _derived_behavior(artifact: ComposedArtifact, source: Behavior) -> Behavior:
    """Feed the current payload to a full attack without altering score truth."""

    metadata = copy.deepcopy(source.meta)
    # JailBound and similar attacks must consume the immediately preceding
    # artifact, not silently jump back to a dataset-specific original image.
    metadata.pop("images_by_type", None)
    return Behavior(
        behavior_id=source.behavior_id,
        dataset=source.dataset,
        instruction=artifact.text,
        eval_type=source.eval_type,
        category=source.category,
        category_orig=source.category_orig,
        keyword=source.keyword,
        keyword_type=source.keyword_type,
        statement=artifact.text,
        target=source.target,
        images=list(artifact.images or source.images),
        meta=metadata,
    )


class AttackBackedComponent(CompositionComponent):
    """Base for components backed by an already registered attack class."""

    attack_name = ""
    algorithm_fidelity = "adapted_composition_stage"

    def __init__(self, stage_spec: Mapping[str, Any], **kwargs):
        super().__init__(stage_spec, **kwargs)
        import core.attacks  # noqa: F401  (registration side effects)
        from .registry import get_attack

        configured = str(stage_spec.get("attack") or self.attack_name)
        if configured.casefold() != self.attack_name.casefold():
            raise ValueError(
                f"component {self.component_name} requires attack {self.attack_name}, "
                f"got {configured}"
            )
        self.source_attack = self.attack_name
        self.attack = get_attack(self.attack_name)(copy.deepcopy(self.config))

    @property
    def effective_config(self) -> dict[str, Any]:
        effective = self.attack.effective_config()
        return copy.deepcopy(effective if effective else self.config)

    def prepare(self, out_dir: str) -> None:
        super().prepare(out_dir)
        self.attack.prepare(out_dir)

    def evaluation_exclusion_reason(self, behavior: Behavior) -> str | None:
        return self.attack.evaluation_exclusion_reason(behavior)

    def preflight(self) -> None:
        from .data import attack_requirements

        missing = [path for path in attack_requirements(self.attack_name, self.config)
                   if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(f"{self.component_name} missing inputs: {missing}")

    def _attack_provenance(self) -> dict[str, Any]:
        provenance = copy.deepcopy(self.attack.provenance())
        # The source stage's own provenance remains visible, but cannot promote
        # a cross-method composition to a paper reproduction.
        provenance["composition_stage_only"] = True
        provenance["component_prepare_seconds"] = self.prepare_seconds
        return provenance

    def provenance(self) -> dict[str, Any]:
        value = super().provenance()
        value["source_provenance"] = self._attack_provenance()
        return value


class FullAttackComponent(AttackBackedComponent):
    """Run an existing build() against the current composition artifact."""

    def _iteration_plan(self) -> dict[str, Any]:
        values = {}
        for key in (
            "attempts", "steps", "epochs", "num_epochs", "attack_steps",
            "rounds", "iterations", "max_iter", "max_query",
        ):
            if hasattr(self.attack, key):
                values[key] = getattr(self.attack, key)
        return values

    def transform(
        self, artifact: ComposedArtifact, behavior: Behavior, out_dir: str
    ) -> ComponentResult:
        current = _derived_behavior(artifact, behavior)
        missing = self.attack.missing_requirements(current)
        if missing:
            raise ValueError(
                f"{self.component_name} missing derived requirements: {', '.join(missing)}"
            )
        os.makedirs(out_dir, exist_ok=True)
        cases = self.attack.build(current, out_dir)
        if not cases:
            raise RuntimeError(f"{self.component_name} produced no test case")
        output_index = int(self.stage_spec.get("output_index", 0))
        if not 0 <= output_index < len(cases):
            raise IndexError(
                f"{self.component_name} output_index {output_index} outside 0..{len(cases)-1}"
            )
        case = cases[output_index]
        output = artifact.with_payload(
            text=case.text,
            images=case.images,
            metadata={f"{self.component_name}_case_meta": copy.deepcopy(case.meta)},
        )
        iteration_result = {
            "outputs_built": len(cases),
            "selected_output_index": output_index,
        }
        for key in (
            "iterations", "initial_loss", "final_step_loss", "max_abs_delta",
            "suffix_token_count", "matched", "match_count", "iters",
        ):
            if key in case.meta:
                iteration_result[key] = copy.deepcopy(case.meta[key])
        if self.attack_name == "JPS":
            iteration_result.update({
                "universal_artifact_prepared": True,
                "optimization_scope": "once_per_component_config",
                "last_pgd_stats": copy.deepcopy(
                    getattr(self.attack, "_last_pgd_stats", {})
                ),
            })
        return ComponentResult(
            artifact=output,
            full_iterations_completed=True,
            iteration_plan=self._iteration_plan(),
            iteration_result=iteration_result,
            metadata={"source_provenance": self._attack_provenance()},
        )


class CSDJSplitComponent(AttackBackedComponent):
    component_name = "cs_dj_split"
    attack_name = "CS-DJ"
    algorithm_fidelity = "adapted_decomposition"

    def prepare(self, out_dir: str) -> None:
        # Full CS-DJ prepare() embeds a 10k contrast-image library.  The Split
        # component is explicitly the query-decomposition subroutine and must
        # not perform the later retrieval/collage stage.
        CompositionComponent.prepare(self, out_dir)

    def preflight(self) -> None:
        # The splitter consumes no collage library or font.
        pass

    def transform(self, artifact, behavior, out_dir):
        subquestions = self.attack._split_question(artifact.text, behavior.behavior_id)
        if len(subquestions) != self.attack.subquestion_count:
            raise RuntimeError("CS-DJ split did not return the configured number of questions")
        decomposed = "\n".join(
            f"{index}. {question}" for index, question in enumerate(subquestions, 1)
        )
        return ComponentResult(
            artifact=artifact.with_payload(text=decomposed),
            iteration_plan={
                "subquestion_count": self.attack.subquestion_count,
                "split_max_retries": self.attack.split_max_retries,
                "split_max_new_tokens": self.attack.split_max_new_tokens,
            },
            iteration_result={"subquestions_generated": len(subquestions)},
            metadata={"source_provenance": self._attack_provenance()},
        )


class CSDJFullComponent(FullAttackComponent):
    """Retain CS-DJ's retrieval, three text cards, and complete 3x4 collage."""

    component_name = "cs_dj_full"
    attack_name = "CS-DJ"

    def preflight(self) -> None:
        super().preflight()
        self.attack._sample_library()
        self.attack._font(self.attack.text_font_size)

    def transform(self, artifact, behavior, out_dir):
        result = super().transform(artifact, behavior, out_dir)
        attack = self.attack
        row_height = attack.tile_height + attack.number_font_size + 10
        boxes = [
            [(index % attack.images_per_row) * attack.tile_width,
             (index // attack.images_per_row) * row_height,
             (index % attack.images_per_row + 1) * attack.tile_width,
             (index // attack.images_per_row) * row_height + attack.tile_height]
            for index in range(attack.collage_image_count)
        ]
        return ComponentResult(
            artifact=result.artifact.with_payload(metadata={"cs_dj_contrast_boxes": boxes}),
            iteration_plan=result.iteration_plan,
            iteration_result=result.iteration_result,
            metadata={**result.metadata, "contrast_boxes": boxes,
                      "text_cards_preserved": True},
        )


class UltraBreakPatchComponent(AttackBackedComponent):
    """Apply a trained UltraBreak patch without replacing the incoming image.

    Training uses the existing optimizer unchanged. Application to a collage,
    reserved side strip, or disjoint region is an explicitly adapted operation.
    A run-wide cache shares one lineage-verified optimization across processes.
    """

    component_name = "ultrabreak_patch"
    attack_name = "UltraBreak"
    algorithm_fidelity = "adapted_decomposition"

    def __init__(self, stage_spec, **kwargs):
        super().__init__(stage_spec, **kwargs)
        self.placement = str(stage_spec.get("placement", "right_strip"))
        self.fraction = float(stage_spec.get("patch_fraction", 0.35))
        self.preserve_top = float(stage_spec.get("preserve_top_fraction", 0.0))
        self.tile_index = int(stage_spec.get("tile_index", 8))
        if self.placement not in {"right_strip", "cs_dj_contrast", "bottom_right"}:
            raise ValueError(f"unsupported UltraBreak patch placement: {self.placement}")
        if not 0 < self.fraction <= 1 or not 0 <= self.preserve_top < 1:
            raise ValueError("invalid UltraBreak patch/preserved-region fraction")
        if not 0 <= self.tile_index < 9:
            raise ValueError("UltraBreak contrast tile_index must be in [0, 8]")

    @property
    def effective_config(self):
        return {**super().effective_config, "patch_application": {
            "placement": self.placement, "patch_fraction": self.fraction,
            "preserve_top_fraction": self.preserve_top, "tile_index": self.tile_index,
        }}

    def prepare(self, out_dir):
        import fcntl

        CompositionComponent.prepare(self, out_dir)
        if self.attack.config.get("adv_image"):
            self.attack.prepare(out_dir)
            if not self.attack.verify_lineage(resolve_path(self.attack._universal_rel)):
                raise ValueError("UltraBreak supplied patch has missing or mismatched lineage")
            return
        identity = {"config": self.attack.effective_config(),
                    "implementation": self.attack._implementation_sha256()}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        root = os.environ.get("VLM_COMPOSITION_CACHE_ROOT", out_dir)
        cache = os.path.join(os.path.abspath(root), "ultrabreak", key)
        os.makedirs(cache, exist_ok=True)
        with open(os.path.join(cache, ".lock"), "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = os.path.join(cache, "universal_adv.png")
            if os.path.exists(path) or os.path.exists(path + ".lineage.json"):
                if not self.attack.verify_lineage(path):
                    raise ValueError(f"UltraBreak cache lineage mismatch: {path}")
                self.attack._universal_rel = portable_path(path)
            else:
                self.attack.prepare(cache)
                if not os.path.isfile(path) or not self.attack._lineage_verified:
                    raise RuntimeError("UltraBreak optimization did not produce a verified patch")
        # Application is CPU-only. Do not keep a second Qwen2-VL on the GPU
        # beside JPS/HADES after the universal training has finished.
        self.attack._adapter = None
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def transform(self, artifact, behavior, out_dir):
        if self.attack._universal_rel is None:
            raise RuntimeError("UltraBreak patch was not prepared")
        source = _input_image(artifact, behavior)
        with Image.open(source) as image:
            original = image.convert("RGB")
        with Image.open(resolve_path(self.attack._universal_rel)) as image:
            patch = image.convert("RGB")
        width, height = original.size
        canvas = original.copy()
        if self.placement == "right_strip":
            side = max(1, round(min(width, height) * self.fraction))
            margin = max(1, side // 20)
            canvas = Image.new("RGB", (width + side + 2 * margin, height), "white")
            canvas.paste(original, (0, 0))
            left, top = width + margin, (height - side) // 2
        elif self.placement == "cs_dj_contrast":
            boxes = artifact.metadata.get("cs_dj_contrast_boxes")
            if not boxes or len(boxes) != 9:
                raise ValueError("CS-DJ contrast placement requires the full collage stage")
            x0, y0, x1, y1 = map(int, boxes[self.tile_index])
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise ValueError("CS-DJ contrast box lies outside the input image")
            side = max(1, int(min(x1 - x0, y1 - y0) * self.fraction))
            left, top = x0 + (x1 - x0 - side) // 2, y0 + (y1 - y0 - side) // 2
        else:
            protected = max(self.preserve_top,
                            float(artifact.metadata.get("hades_top_fraction", 0.0)))
            first_row = math.ceil(height * protected)
            side = max(1, int(min(width, height - first_row) * self.fraction))
            if side > height - first_row:
                raise ValueError("UltraBreak has no space outside the protected HADES region")
            left, top = width - side, height - side
        canvas.paste(patch.resize((side, side), Image.Resampling.LANCZOS), (left, top))
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{_safe_stem(behavior.behavior_id)}_ultrabreak.png")
        canvas.save(path)
        text = artifact.text
        if self.attack.tpg_suffix and not text.endswith(self.attack.tpg_suffix):
            text += self.attack.tpg_suffix
        return ComponentResult(
            artifact=artifact.with_payload(text=text, images=[portable_path(path)]),
            iteration_plan={"epochs": self.attack.epochs, "train_rows": self.attack.train_rows},
            iteration_result={"universal_optimization_completed": True,
                              "optimization_scope": "once_per_verified_config",
                              "artifact_reused": self.attack._artifact_reused,
                              "application_completed": True},
            metadata={"placement": self.placement,
                      "patch_box": [left, top, left + side, top + side],
                      "input_size": [width, height], "output_size": list(canvas.size),
                      "universal_patch_sha256": file_sha256(resolve_path(self.attack._universal_rel)),
                      "source_provenance": self._attack_provenance()},
        )


class FigStepRenderComponent(FullAttackComponent):
    component_name = "figstep_render"
    attack_name = "FigStep"
    algorithm_fidelity = "adapted_decomposition"


class MMLLayoutComponent(FullAttackComponent):
    component_name = "mml_layout"
    attack_name = "MML"
    algorithm_fidelity = "adapted_decomposition"


class JPSJointComponent(FullAttackComponent):
    component_name = "jps_joint"
    attack_name = "JPS"


class JailBoundJointComponent(FullAttackComponent):
    component_name = "jailbound_joint"
    attack_name = "JailBound"


class JailbreakInPiecesEmbeddingComponent(AttackBackedComponent):
    component_name = "jailbreak_in_pieces_embedding"
    attack_name = "JailbreakInPieces"
    algorithm_fidelity = "adapted_decomposition"

    def transform(self, artifact, behavior, out_dir):
        target = _input_image(artifact, behavior)
        target_digest = file_sha256(target)
        reused = bool(target_digest and target_digest in self.attack._artifacts)
        optimized = self.attack._optimize(target, out_dir)
        output = artifact.with_payload(
            images=[optimized["image"]],
            metadata={"jip_embedding": copy.deepcopy(optimized)},
        )
        return ComponentResult(
            artifact=output,
            iteration_plan={
                "num_epochs": self.attack.num_epochs,
                "learning_rate": self.attack.learning_rate,
            },
            iteration_result={
                "epochs_completed": self.attack.num_epochs,
                "epochs_completed_this_call": 0 if reused else self.attack.num_epochs,
                "artifact_reused": reused,
                "initial_loss": optimized["initial_loss"],
                "final_loss": optimized["final_loss"],
            },
            metadata={"source_provenance": self._attack_provenance()},
        )


class HADESPatchComponent(AttackBackedComponent):
    component_name = "hades_patch"
    attack_name = "HADES"
    algorithm_fidelity = "adapted_decomposition"

    def transform(self, artifact, behavior, out_dir):
        import torch
        from torchvision.utils import save_image
        from multi_attacks.hades.white_box import apply_top_patch

        if self.attack._patch is None:
            raise RuntimeError("HADES patch was not prepared")
        current_images = artifact.images or behavior.images
        if not current_images:
            raise ValueError("HADES-Patch requires an input image")
        source_index = self.attack._step_index(
            self.attack.source_step, len(current_images)
        )
        source = resolve_path(current_images[source_index])
        surrogate = self.attack._load_surrogate()
        base = surrogate.to_unit01(Image.open(source).convert("RGB")).detach()
        patch = self.attack._patch
        expected = (
            base.shape[1], max(1, int(base.shape[2] * self.attack.patch_fraction)), base.shape[3]
        )
        if patch.shape[1:] != expected:
            raise ValueError(
                f"HADES patch/image mismatch: {tuple(patch.shape)} vs {tuple(base.shape)}"
            )
        adversarial = apply_top_patch(base, patch).detach().cpu().clamp(0, 1)
        if not torch.isfinite(adversarial).all():
            raise RuntimeError("HADES patch produced non-finite pixels")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{_safe_stem(behavior.behavior_id)}_hades_top.png")
        save_image(adversarial, path)
        stats = copy.deepcopy(self.attack._optimization_stats)
        return ComponentResult(
            artifact=artifact.with_payload(
                images=[portable_path(path)],
                metadata={"hades_top_fraction": self.attack.patch_fraction},
            ),
            iteration_plan={
                "epochs": self.attack.epochs,
                "batch_size": self.attack.batch_size,
                "patch_fraction": self.attack.patch_fraction,
            },
            iteration_result={
                "universal_optimization_completed": True,
                "optimization_scope": "once_per_component_config",
                "application_completed": True,
                **stats,
            },
            metadata={
                "spatial_region": "top",
                "source_image_index": source_index,
                "source_provenance": self._attack_provenance(),
            },
        )


class DeltaJPComponent(AttackBackedComponent):
    component_name = "delta_jp"
    attack_name = "DeltaJP"
    algorithm_fidelity = "adapted_composition_stage"

    def transform(self, artifact, behavior, out_dir):
        import torch
        from torchvision.utils import save_image

        if self.attack._delta is None:
            raise RuntimeError("DeltaJP universal delta was not prepared")
        source = _input_image(artifact, behavior)
        surrogate = self.attack._load_surrogate()
        unit = surrogate.to_unit01(Image.open(source).convert("RGB"))
        delta = self.attack._delta.to(unit.device, dtype=unit.dtype)
        if delta.shape != unit.shape:
            raise ValueError(
                f"DeltaJP delta/image shape mismatch: {tuple(delta.shape)} vs {tuple(unit.shape)}"
            )
        preserve_top = float(self.stage_spec.get("preserve_top_fraction", 0.0))
        if preserve_top:
            delta = delta * _bottom_update_mask_like(delta, preserve_top)
        adversarial = (unit + delta).clamp(0, 1).detach().cpu()
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{_safe_stem(behavior.behavior_id)}_delta.png")
        save_image(adversarial, path)
        return ComponentResult(
            artifact=artifact.with_payload(images=[portable_path(path)]),
            iteration_plan={
                "steps": self.attack.steps,
                "batch_size": self.attack.batch_size,
                "epsilon_255": self.attack.epsilon * 255,
            },
            iteration_result={
                "universal_optimization_completed": True,
                "optimization_scope": "once_per_component_config",
                "optimization_steps": self.attack.steps,
                "application_completed": True,
            },
            metadata={
                "preserve_top_fraction": preserve_top,
                "spatial_region": "bottom" if preserve_top else "full",
                "source_provenance": self._attack_provenance(),
            },
        )


class JOODAugComponent(CompositionComponent):
    component_name = "jood_aug"
    source_attack = "JOOD"
    algorithm_fidelity = "adapted_decomposition"

    def __init__(self, stage_spec, **kwargs):
        super().__init__(stage_spec, **kwargs)
        self.aug = str(self.config.get("aug", "randaug1"))
        if not self.aug.startswith("randaug"):
            raise ValueError("JOOD-Aug composition currently requires a randaugN profile")
        try:
            self.n = int(self.aug.removeprefix("randaug"))
        except ValueError as exc:
            raise ValueError(f"invalid JOOD augmentation: {self.aug}") from exc
        if not 1 <= self.n <= 14:
            raise ValueError("JOOD randaugN requires N in [1, 14]")
        self.strengths = [float(value) for value in self.config.get("lams", [0.5])]
        if not self.strengths or any(not 0 <= value <= 1 for value in self.strengths):
            raise ValueError("JOOD strengths must be non-empty values in [0, 1]")
        configured_seed = self.config.get("seed", 0)
        self.seed = 0 if configured_seed is None else int(configured_seed)

    def transform(self, artifact, behavior, out_dir):
        from multi_attacks.jood.jood_utils import RandAug

        source = _input_image(artifact, behavior)
        os.makedirs(out_dir, exist_ok=True)
        candidates: list[str] = []
        seeds: list[int] = []
        for attempt, strength in enumerate(self.strengths):
            tag = f"{behavior.behavior_id}\0{self.aug}\0{strength}\0{self.seed}"
            seed = int.from_bytes(hashlib.sha256(tag.encode("utf-8")).digest()[:4], "big")
            image = RandAug(self.n, strength).apply(source, seed).convert("RGB")
            path = os.path.join(
                out_dir, f"{_safe_stem(behavior.behavior_id)}_{attempt:02d}.png"
            )
            image.save(path)
            candidates.append(portable_path(path))
            seeds.append(seed)
        selected = _selection_index(
            f"{behavior.behavior_id}\0{self.component_name}\0{self.seed}", len(candidates)
        )
        return ComponentResult(
            artifact=artifact.with_payload(images=[candidates[selected]]),
            iteration_plan={"augmentations": len(self.strengths), "randaug_n": self.n},
            iteration_result={
                "augmentations_completed": len(candidates),
                "selected_index": selected,
            },
            metadata={
                "candidate_images": candidates,
                "candidate_image_sha256s": strict_image_hashes(candidates),
                "candidate_seeds": seeds,
                "strengths": self.strengths,
                "selection_policy": "deterministic_hash_not_asr",
            },
        )



COMPONENTS: dict[str, type[CompositionComponent]] = {
    cls.component_name: cls
    for cls in (
        CSDJSplitComponent,
        CSDJFullComponent,
        UltraBreakPatchComponent,
        FigStepRenderComponent,
        MMLLayoutComponent,
        JPSJointComponent,
        JailBoundJointComponent,
        JailbreakInPiecesEmbeddingComponent,
        HADESPatchComponent,
        DeltaJPComponent,
        JOODAugComponent,
    )
}


def make_component(stage_spec: Mapping[str, Any]) -> CompositionComponent:
    name = str(stage_spec.get("component", ""))
    try:
        component_cls = COMPONENTS[name]
    except KeyError:
        raise KeyError(
            f"unknown composition component {name!r}; available: {sorted(COMPONENTS)}"
        ) from None
    return component_cls(stage_spec)
