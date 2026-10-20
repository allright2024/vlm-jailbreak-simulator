"""Judges. Importing this package registers all judges.

A judge scores a Completion against the ORIGINAL harmful instruction. The
primary judge is HarmBench (reusing the existing wrapper + prompt in
eval_utils.py); add others here as needed.
"""

from .beaver_dam import BeaverDamJudge  # noqa: F401
from .harmbench import HarmBenchJudge  # noqa: F401
from .llama_guard import LlamaGuardJudge  # noqa: F401
from .mml_clas import MMLCLASJudge  # noqa: F401

__all__ = ["BeaverDamJudge", "HarmBenchJudge", "LlamaGuardJudge", "MMLCLASJudge"]
