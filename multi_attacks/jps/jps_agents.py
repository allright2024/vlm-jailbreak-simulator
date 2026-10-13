"""JPS multi-agent textual steering (feedback -> summarize -> revision).

Faithful to thu-coai/JPS revision/*.py, but the backing agent LLM is an OpenAI
model (via OpenAIRefiner) instead of the paper's local Qwen2.5-14B. Each round,
for each angle, the three agents run in sequence and update that angle's steer
prompt; the two angle prompts are joined and prepended to the target query.
"""

from __future__ import annotations

import re

from core.refiners import OpenAIRefiner
from . import jps_prompts as P


def _extract_steer(text: str) -> str | None:
    """Pull the new steer prompt from @@...@@ (revision agent output format)."""
    m = re.findall(r"@@(.*?)@@", text or "", re.DOTALL)
    return m[-1].strip() if m else None


class JPSAgents:
    """Three-agent steering refinement, one steer prompt per angle."""

    def __init__(self, model: str = "gpt-4o-2024-11-20"):
        self.refiner = OpenAIRefiner(model=model, max_tokens=2048)

    def _chat(self, system: str, user: str) -> str:
        # OpenAIRefiner concatenates (refine_prompt, user_prompt) as one user msg;
        # pass the system text as the first arg to prepend it.
        return self.refiner.refine(system, user) or ""

    def revise(self, qa_list: list[tuple[str, str]], prompt_list: list[str]) -> list[str]:
        """qa_list: (question, response) pairs. prompt_list: current steer prompt per angle."""
        new_prompts = list(prompt_list)
        for i, angle in enumerate(P.ANGLES):
            # 1) feedback per QA pair (keep only well-formed 'Rating: [[n]] Suggestion: ...')
            qa_feedback = []
            for q, r in qa_list:
                fb = self._chat(P.FEEDBACK_SYSTEM, P.FEEDBACK_PROMPT[angle].format(question=q, response=r))
                if re.search(r"Rating:\s*\[\[\d+\]\]", fb):
                    qa_feedback.append((q, r, fb))
            if not qa_feedback:
                continue

            # 2) summarize feedback (+ history of the current prompt)
            prev = prompt_list[i]
            history = (
                f'The following prompt is the steer prompt previously sent to the model: "{prev}". '
                f"When providing suggestions, consider its shortcomings." if prev else ""
            )
            qa_fb_text = "".join(
                f'\nQuestion {j+1}: "{q}"\nAnswer {j+1}: "{a}"\nEvaluation {j+1}: "{e}"\n'
                for j, (q, a, e) in enumerate(qa_feedback)
            )
            summary = self._chat(
                P.FEEDBACK_SYSTEM,
                P.SUMMARIZE_PROMPT[angle].format(qa_feedback_prompt=qa_fb_text, history_prompt=history),
            )

            # 3) revise: produce the new steer prompt (surrounded by @@)
            hist2 = f'Your previous steer prompt was: "{prev}".' if prev else ""
            out = self._chat(
                P.FEEDBACK_SYSTEM,
                P.REVISION_PROMPT[angle].format(qa_summarize=summary, history_prompt=hist2),
            )
            steer = _extract_steer(out)
            if steer:
                new_prompts[i] = steer
        return new_prompts
