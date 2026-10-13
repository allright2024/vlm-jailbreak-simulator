"""JPS (ACM MM 2025) ported to the core interface.

JPS co-optimizes a visual perturbation and a textual steering prompt:
  * Visual: momentum PGD on a Qwen2-VL-7B surrogate drives the model toward an
    affirmative target prefix ("Sure, here is ...") for each training query,
    conditioned on the current steer prompt.
  * Textual: a three-agent loop (feedback -> summarize -> revision) refines two
    steer prompts (instruction-following and harmfulness angles).
They alternate for ``rounds`` iterations. The output is one universal
adversarial image + a steer prompt, paired with every query at attack time.

Ported from thu-coai/JPS (MIT). Two substitutions, both configurable and
documented: the agent LLM is an OpenAI model (paper: local Qwen2.5-14B), and the
visual PGD reuses this repo's Qwen2-VL adapter infrastructure. requires_surrogate.
"""

from __future__ import annotations

import os
import csv
import hashlib
import json
from fractions import Fraction

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from core.attack import Attack
from core.dataset import DATA_ROOT, REPO_ROOT
from core.paths import resolve_path
from core.registry import register_attack
from core.schema import Behavior, TestCase
from .jps_prompts import JB_PROMPT_TEMPLATE
from multi_attacks.ultrabreak.ultrabreak_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD


from multimodalmodels.registry import QWEN2_VL_REVISION  # pinned in multimodalmodels/registry.py


def _affirmative_target(instruction: str) -> str:
    """Affirmative-prefix target, mirroring JPS goal construction / AdvBench targets."""
    return f"Sure, here is {instruction[0].lower() + instruction[1:]}"


def _parse_ratio(value) -> float:
    """Parse numbers such as ``32/255`` without executing config text."""
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(Fraction(str(value).strip()))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"Invalid numeric ratio: {value!r}") from exc


def _project_linf(original: torch.Tensor, candidate: torch.Tensor, epsilon: float):
    """Project a candidate onto one immutable original-image L_inf ball."""
    delta = (candidate - original).clamp(-epsilon, epsilon)
    return (original + delta).clamp(0, 1)


@register_attack("JPS")
class JPSAttack(Attack):
    official_repo = "https://github.com/thu-coai/JPS"
    official_commit = "beacbc5"
    # Agent model and training subset are configurable substitutions.  The
    # bundled default uses GPT-4o and local AdvBench because the upstream
    # advbench_subset.csv is not distributed in the official repository.
    algorithm_fidelity = "adapted"
    substitutes_model = True  # Qwen2-VL surrogate + GPT-4o agents (vs paper's Qwen2.5-14B)
    requires_surrogate = True
    requires_images = False

    def __init__(self, config=None):
        super().__init__(config)
        c = self.config
        self.surrogate = c.get("surrogate", "Qwen/Qwen2-VL-7B-Instruct")
        self.surrogate_revision = c.get("surrogate_revision", QWEN2_VL_REVISION)
        self.agent_model = c.get("agent_model", "gpt-4o-2024-11-20")
        self.base_image = c.get("base_image", os.path.join(DATA_ROOT, "FORCE", "clean.jpeg"))
        self.image_size = int(c.get("image_size", 448))
        self.epsilon = _parse_ratio(c.get("epsilon", "32/255"))
        self.alpha = _parse_ratio(c.get("alpha", "1/255"))
        self.attack_steps = int(c.get("attack_steps", 300))
        self.rounds = int(c.get("rounds", 5))          # revision_iteration_num
        self.attack_data_num = int(c.get("attack_data_num", 50))
        self.momentum_decay = float(c.get("momentum_decay", 0.9))
        self._adapter = None
        self._agents = None
        self._universal_rel = None
        self._steer = ""
        self._train_behaviors = None

    def _rel(self, p):
        return resolve_path(p)

    def _load(self):
        if self._adapter is None:
            from multi_attacks.ultrabreak.ultrabreak_adapter import Qwen2Adapter
            self._adapter = Qwen2Adapter(
                self.surrogate,
                revision=self.surrogate_revision,
                patch_only=False,
                image_size=(self.image_size, self.image_size),
            )
            from .jps_agents import JPSAgents
            self._agents = JPSAgents(model=self.agent_model)
        return self._adapter

    # ---- visual PGD (momentum) toward affirmative targets, steer-conditioned ----
    def _pgd(self, adapter, train, steer, original_img, adv_img):
        device = adapter.device
        mean = torch.tensor(OPENAI_CLIP_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(OPENAI_CLIP_STD, device=device).view(1, 3, 1, 1)
        base = original_img.detach().clone()
        projected = _project_linf(base, adv_img.detach(), self.epsilon)
        delta = (projected - base).detach().requires_grad_(True)
        momentum = torch.zeros_like(base)
        best_loss = float("inf")
        best_adv = (base + delta).detach().clamp(0, 1)

        # pre-build per-sample (input_ids, labels, grid) with a placeholder pixel image
        prepared = []
        for q, target in train:
            text = (steer + "\n" if steer else "") + JB_PROMPT_TEMPLATE.format(
                attack_question=q, attack_target=target.split(":")[0] if ":" in target else target)
            ids, labels, _, grid, attn, _ = adapter.process_target(self._rel(self.base_image), text, target)
            prepared.append((ids, labels, grid, attn))

        for step in range(self.attack_steps):
            total = 0.0
            grad_accum = torch.zeros_like(base)
            for (ids, labels, grid, attn) in prepared:
                # delta stays in the graph; preprocess_patched is differentiable,
                # so a single forward gives a clean image-space gradient on delta.
                x = (base + delta).clamp(0, 1)
                pv = adapter.preprocess_patched(((x - mean) / std)[0].to(adapter.model.dtype), grid[0])
                out = adapter.model(input_ids=ids, attention_mask=attn, pixel_values=pv, image_grid_thw=grid)
                lab = labels[:, 1:]
                loss = F.cross_entropy(out.logits[:, :-1, :].reshape(-1, out.logits.size(-1)).float(),
                                       lab.reshape(-1), ignore_index=-100)
                grad_accum = grad_accum + torch.autograd.grad(loss, delta)[0]
                total += float(loss.detach())
            mean_loss = total / max(len(prepared), 1)
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_adv = (base + delta).detach().clamp(0, 1)

            # Official JPS uses L2-normalized gradient momentum with beta=0.9.
            g = grad_accum / max(len(prepared), 1)
            flat_norm = g.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1).clamp_min(1e-8)
            normalized = g / flat_norm
            momentum = self.momentum_decay * momentum + (1 - self.momentum_decay) * normalized
            delta = (delta.detach() - self.alpha * momentum.sign()).clamp(
                -self.epsilon, self.epsilon
            )
            delta = (_project_linf(base, base + delta, self.epsilon) - base)
            delta = delta.detach().requires_grad_(True)
            if step % 20 == 0:
                print(f"  [JPS pgd] step {step}/{self.attack_steps} loss={mean_loss:.4f}", flush=True)
        return best_adv

    def _optimize(self, behaviors, out_dir):
        adapter = self._load()
        device = adapter.device
        train = [(b.instruction, b.target or _affirmative_target(b.instruction))
                 for b in behaviors[: self.attack_data_num]]

        img = Image.open(self._rel(self.base_image)).convert("RGB").resize((self.image_size, self.image_size))
        original = transforms.ToTensor()(img).unsqueeze(0).to(device)
        adv = original.clone()
        steer_angles = ["", ""]

        # Upstream first optimizes the image once, then alternates
        # response-feedback/revision and another image optimization.
        adv = self._pgd(adapter, train, "", original, adv)
        for r in range(self.rounds):
            # textual revision: sample responses, then run the 3-agent loop
            from multi_attacks.ultrabreak.ultrabreak_utils import OPENAI_CLIP_MEAN  # noqa
            os.makedirs(out_dir, exist_ok=True)
            tmp = os.path.join(out_dir, "jps_tmp.png")
            transforms.ToPILImage()(adv.squeeze(0).cpu()).save(tmp)
            qa = []
            steer_now = "\n".join(s for s in steer_angles if s)
            for q, _ in train:
                prompt = (steer_now + "\n" if steer_now else "") + q
                qa.append((q, adapter.generate(prompt, tmp, max_new_tokens=128)))
            steer_angles = self._agents.revise(qa, steer_angles)
            adv = self._pgd(
                adapter, train, "\n".join(s for s in steer_angles if s), original, adv
            )
            print(f"  [JPS] round {r+1}/{self.rounds} steer updated", flush=True)

        os.makedirs(out_dir, exist_ok=True)
        fpath = os.path.join(out_dir, "jps_adv.png")
        transforms.ToPILImage()(adv.squeeze(0).cpu()).save(fpath)
        self._steer = "\n".join(s for s in steer_angles if s)
        # save the steer prompt next to the image so later runs can reuse both
        with open(os.path.splitext(fpath)[0] + "_steer.txt", "w", encoding="utf-8") as f:
            f.write(self._steer)
        return os.path.relpath(fpath, REPO_ROOT)

    def _training_corpus(self):
        """Fixed training set for the universal attack (JPS: AdvBench-subset)."""
        if self._train_behaviors is not None:
            return self._train_behaviors
        import core.datasets  # noqa: F401
        from core import get_dataset
        name = self.config.get("train_dataset", "AdvBench")
        candidates = get_dataset(name)().load()
        train_csv = self.config.get("train_csv")
        if train_csv:
            with open(self._rel(train_csv), newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))[: self.attack_data_num]
            wanted = {
                (r.get("goal") or r.get("prompt") or r.get("origin_question") or "").strip()
                for r in rows
            }
            selected = [b for b in candidates if b.instruction.strip() in wanted]
            if len(selected) != len(wanted):
                raise ValueError(
                    f"JPS train_csv matched {len(selected)}/{len(wanted)} rows in {name}"
                )
        else:
            selected = candidates[: self.attack_data_num]
        self._train_behaviors = selected
        return selected

    def training_behavior_ids(self) -> list[str]:
        return [b.behavior_id for b in self._training_corpus()]

    def training_instruction_hashes(self) -> set[str]:
        from core.attack import normalized_instruction_hash
        return {normalized_instruction_hash(b.instruction) for b in self._training_corpus()}

    def training_behavior_fingerprints(self) -> set[str]:
        fingerprints = set()
        for behavior in self._training_corpus():
            raw = json.dumps(behavior.to_dict(), ensure_ascii=False, sort_keys=True)
            fingerprints.add(hashlib.sha256(raw.encode("utf-8")).hexdigest())
        return fingerprints

    def effective_config(self):
        train_csv = self.config.get("train_csv")
        return {
            "surrogate": self.surrogate,
            "surrogate_revision": self.surrogate_revision,
            "agent_model": self.agent_model,
            "base_image": self._rel(self.base_image),
            "train_dataset": self.config.get("train_dataset", "AdvBench"),
            "train_csv": self._rel(train_csv) if train_csv else None,
            "image_size": self.image_size,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "attack_steps": self.attack_steps,
            "rounds": self.rounds,
            "attack_data_num": self.attack_data_num,
            "momentum_decay": self.momentum_decay,
            "exclude_train_from_eval": self.exclude_train_from_eval,
        }

    def _lineage_inputs(self):
        inp = {"base_image": self._rel(self.base_image)}
        tc = self.config.get("train_csv")
        if tc:
            inp["train_csv"] = self._rel(tc)
        return inp

    def _lineage_auxiliary_files(self, artifact_abs):
        steer_path = self.config.get("steer_path")
        if self._artifact_reused and steer_path:
            steer_abs = self._rel(steer_path)
        else:
            steer_abs = os.path.splitext(artifact_abs)[0] + "_steer.txt"
        return {"steer_prompt": steer_abs}

    def _lineage_code_files(self):
        here = os.path.dirname(__file__)
        ultra = os.path.join(here, "..", "ultrabreak")
        return super()._lineage_code_files() + [
            os.path.join(here, "jps_agents.py"),
            os.path.join(here, "jps_prompts.py"),
            os.path.join(ultra, "ultrabreak_adapter.py"),
            os.path.join(ultra, "ultrabreak_utils.py"),
            os.path.join(here, "..", "..", "multimodalmodels", "loaders.py"),
            os.path.join(here, "..", "..", "multimodalmodels", "registry.py"),
        ]

    def prepare(self, out_dir: str) -> None:
        """Co-optimize the universal image + steer once (or load a prebuilt pair)."""
        if self._universal_rel is not None:
            return
        pre = self.config.get("adv_image")  # reuse a pre-made image + steer (skip co-optimization)
        if pre:
            abs_art = self._rel(pre)
            self._universal_rel = os.path.relpath(abs_art, REPO_ROOT)
            sp = self.config.get("steer_path")
            # the steer file is part of the artifact: a reused image WITHOUT its
            # steer prompt is an incomplete/altered artifact.
            if sp and os.path.isfile(self._rel(sp)):
                self._steer = open(self._rel(sp)).read()
                self.verify_lineage(abs_art)
            else:
                self._steer = self.config.get("steer", "")
                self._artifact_reused = True
                self._lineage_verified = False  # missing steer -> cannot verify
        else:
            # optimize once over a FIXED training corpus (independent of eval behaviors)
            self._universal_rel = self._optimize(self._training_corpus(), out_dir)
            self.write_lineage(self._rel(self._universal_rel))

    def build(self, behavior: Behavior, out_dir: str) -> list[TestCase]:
        if self._universal_rel is None:  # prepare() normally runs first
            self.prepare(out_dir)
        text = (self._steer + "\n" if self._steer else "") + behavior.instruction
        return [TestCase(
            test_case_id=self.test_case_id(self.name, behavior.behavior_id),
            behavior_id=behavior.behavior_id,
            attack=self.name,
            text=text,
            images=[self._universal_rel],
            meta={"universal": True, "has_steer": bool(self._steer)},
        )]
