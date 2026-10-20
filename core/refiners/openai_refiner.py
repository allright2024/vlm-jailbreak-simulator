"""OpenAI-backed refiner for prompt rewriting (used by HADES and the JPS agents).

Sends ``refine_prompt`` + ``user_prompt`` as one user message to an OpenAI chat
model and returns the rewritten prompt. The API key is read from the environment
(``OPENAI_API_KEY`` by default); a project-root ``.env`` is loaded if present so
the key never has to live in a tracked file.
"""

from __future__ import annotations

import os

from ..dataset import REPO_ROOT


class OpenAIRefiner:
    name = "openai"

    def __init__(self, model: str = "gpt-4o-2024-11-20", key_env: str = "OPENAI_API_KEY",
                 base_url: str | None = None, max_tokens: int = 3000):
        # load .env from the repo root if present (no-op if python-dotenv absent)
        try:
            from dotenv import load_dotenv
            load_dotenv(os.path.join(REPO_ROOT, ".env"))
        except Exception:
            pass
        key = os.environ.get(key_env)
        if not key:
            raise RuntimeError(
                f"No API key in ${key_env}. Put it in {REPO_ROOT}/.env "
                f"(e.g. {key_env}=sk-...) or export it."
            )
        from openai import OpenAI
        self.model = model
        self.max_tokens = max_tokens
        self._client = OpenAI(api_key=key, base_url=base_url or os.environ.get("OPENAI_BASE_URL"))

    def chat(self, system_prompt: str, user_prompt: str, *,
             max_tokens: int | None = None, temperature: float = 0) -> str | None:
        """Chat with distinct system/user roles using deterministic decoding.

        JPS's official agents rely on the system role rather than prepending the
        system instruction to a user message.  ``refine`` below deliberately
        retains its historical one-user-message contract for the other attacks.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens or self.max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:  # noqa: BLE001
            print(f"Error calling auxiliary refiner: {exc}")
            return None

    def refine(self, refine_prompt: str, user_prompt: str) -> str | None:
        """Return the rewritten prompt, or None on API error."""
        return self.chat("", f"{refine_prompt}\n\n{user_prompt}",
                         max_tokens=self.max_tokens, temperature=1)
