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
from .paths import DATA_ROOT_ENV, portable_path, resolve_path
from .attack import Attack
from .model import TargetModel, Judge
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
    "Attack",
    "TargetModel",
    "Judge",
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
