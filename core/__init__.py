"""Unified core for the VLM jailbreak simulator.

This package defines the shared contract that every attack, dataset loader, and
evaluator agrees on, so that adding a new attack (FigStep, HADES, JOOD, ASO,
FORCE, DMN, CS-DJ, ...) does not require touching the pipeline.

Three concepts:
  - Behavior:  one harmful request drawn from a benchmark (the unit we score).
  - TestCase:  the concrete model input an attack produces from a Behavior
               (one or more images + a text prompt, possibly several per
               behavior and several attempts).
  - Completion: a target-model response to a TestCase, later judged for ASR.

See core/schema.py for the data classes and core/registry.py for how attacks,
datasets, models, judges, and defenses register themselves.
"""

from .schema import Behavior, TestCase, Completion, EvalType
from .registry import (
    register_attack,
    register_dataset,
    get_attack,
    get_dataset,
    list_attacks,
    list_datasets,
)
from .dataset import Dataset, REPO_ROOT, DATA_ROOT
from .dataset_policy import (
    JAILBREAK_EVAL_TYPES,
    eval_type_counts,
    parse_eval_type_spec,
    require_harmful_only,
    select_eval_types,
)
from .paths import DATA_ROOT_ENV, portable_path, resolve_path
from .attack import Attack
from .model import TargetModel, Judge
from .metrics import (
    attack_budget_curve,
    attack_success_metrics,
    binary_detection_metrics,
    bootstrap_mean_interval,
    continuous_detection_metrics,
    efficiency_metrics,
    efficiency_overhead_metrics,
    evaluation_coverage_metrics,
    grouped_attack_metrics,
    judge_agreement_metrics,
    mcnemar_exact_test,
    paired_binary_utility_metrics,
    paired_defense_metrics,
    pass_at_k_curve,
    response_quality_metrics,
    seed_stability_metrics,
    success_efficiency_metrics,
    transfer_attack_metrics,
    wilson_confidence_interval,
)
from .multimodal_metrics import (
    aggregate_image_pair_metrics,
    cosine_similarity,
    image_perturbation_metrics,
    modality_ablation_metrics,
    transformation_robustness_metrics,
    variant_robustness_metrics,
)
from .io import read_jsonl, write_jsonl, append_jsonl, iter_jsonl

__all__ = [
    # schema
    "Behavior",
    "TestCase",
    "Completion",
    "EvalType",
    # registry
    "register_attack",
    "register_dataset",
    "get_attack",
    "get_dataset",
    "list_attacks",
    "list_datasets",
    # base classes
    "Dataset",
    "JAILBREAK_EVAL_TYPES",
    "parse_eval_type_spec",
    "select_eval_types",
    "require_harmful_only",
    "eval_type_counts",
    "Attack",
    "TargetModel",
    "Judge",
    # metrics
    "attack_success_metrics",
    "binary_detection_metrics",
    "wilson_confidence_interval",
    "bootstrap_mean_interval",
    "pass_at_k_curve",
    "attack_budget_curve",
    "grouped_attack_metrics",
    "transfer_attack_metrics",
    "response_quality_metrics",
    "success_efficiency_metrics",
    "continuous_detection_metrics",
    "paired_defense_metrics",
    "paired_binary_utility_metrics",
    "mcnemar_exact_test",
    "evaluation_coverage_metrics",
    "judge_agreement_metrics",
    "seed_stability_metrics",
    "efficiency_metrics",
    "efficiency_overhead_metrics",
    "cosine_similarity",
    "image_perturbation_metrics",
    "aggregate_image_pair_metrics",
    "modality_ablation_metrics",
    "variant_robustness_metrics",
    "transformation_robustness_metrics",
    # paths
    "REPO_ROOT",
    "DATA_ROOT",
    "DATA_ROOT_ENV",
    "portable_path",
    "resolve_path",
    # io
    "read_jsonl",
    "write_jsonl",
    "append_jsonl",
    "iter_jsonl",
]
