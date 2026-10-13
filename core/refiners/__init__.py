"""Auxiliary refiner LLMs for attacks that rewrite prompts (HADES, JPS agents).

A refiner rewrites an attack prompt when an attempt fails. The OpenAI refiner
loads its key from the environment (or a .env file via python-dotenv).
"""

from .openai_refiner import OpenAIRefiner

__all__ = ["OpenAIRefiner"]
