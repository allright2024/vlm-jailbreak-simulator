import numpy as np
import torch
import torch.nn.functional as N

from .base import BaseDefense


REFUSAL_WORDS = [
    "Sorry", "sorry", "unfortunately", "unfortunate", "sadly", "explicit",
    "deadly", "crim", "criminal", "illegal", "dangerous", "harmful",
    "warning", "alarm", "caution", "shame", "conspiracy", "Subject",
    "contrary", "shouldn",
]


class HiddenDetectDefense(BaseDefense):
    """HiddenDetect: hidden-state refusal-direction 기반 jailbreak 판정."""

    def __init__(self, config=None):
        super().__init__(config)
        self._refusal_vec = None

    def _build_refusal_vector(self, tokenizer, vocab_size, device):
        vec = torch.zeros(vocab_size)
        for w in REFUSAL_WORDS:
            ids = tokenizer.encode(w, add_special_tokens=False)
            if ids:
                vec[ids[0]] = 1.0
        return vec.to(device)

    def _get_norm_and_head(self, hf_model):
        lm_head = getattr(hf_model, "lm_head", None)
        if lm_head is None and hasattr(hf_model, "language_model"):
            lm_head = getattr(hf_model.language_model, "lm_head", None)

        norm = None
        if hasattr(hf_model, "language_model"):
            lm = hf_model.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "norm"):
                norm = lm.model.norm
            elif hasattr(lm, "norm"):
                norm = lm.norm
        if norm is None and hasattr(hf_model, "model"):
            inner = hf_model.model
            if hasattr(inner, "norm"):
                norm = inner.norm
            elif hasattr(inner, "language_model") and hasattr(inner.language_model, "norm"):
                norm = inner.language_model.norm

        if norm is None or lm_head is None:
            raise RuntimeError(
                "HiddenDetect: LLaVA 구조에서 norm/lm_head를 찾지 못했습니다. "
                "transformers 버전에 맞게 _get_norm_and_head 경로를 조정해야 합니다."
            )
        return norm, lm_head

    def apply(self, image_filename, prompt, model=None, image_dir=None, **kwargs):
        if model is None or image_dir is None:
            raise ValueError(
                "HiddenDetectDefense.apply() requires `model`(LLaVA) and `image_dir`."
            )

        import os
        from PIL import Image

        threshold = self.config.get("default_threshold", 0.2)
        s = self.config.get("layer_start", 16)
        e = self.config.get("layer_end", 29)

        hf_model = model.model
        processor = model.processor
        tokenizer = processor.tokenizer
        device = hf_model.device

        if self._refusal_vec is None:
            self._refusal_vec = self._build_refusal_vector(
                tokenizer, hf_model.config.text_config.vocab_size, device
            )

        norm, lm_head = self._get_norm_and_head(hf_model)

        raw_image = Image.open(os.path.join(image_dir, image_filename))
        conv_prompt = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's "
            f"questions. USER: <image>\n{prompt} ASSISTANT:"
        )
        inputs = processor(images=raw_image, text=conv_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = hf_model(**inputs, output_hidden_states=True)

        F = []
        ref = self._refusal_vec
        for r in outputs.hidden_states[1:]:
            layer_output = norm(r)
            logits = lm_head(layer_output)
            next_token_logits = logits[:, -1, :].float()
            cos_sim = N.cosine_similarity(next_token_logits, ref.unsqueeze(0))
            F.append(cos_sim.item())

        F_selected = F[s:e + 1]
        _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
        score = float(_trapz(np.array(F_selected))) if F_selected else 0.0
        print(f"[HiddenDetect] {image_filename} score={score:.4f} threshold={threshold}")
        if score > threshold:
            return dict(
                blocked=True,
                image_filename=image_filename,
                prompt=prompt,
                default_response="I'm sorry, I can't assist with that.",
            )

        return dict(
            blocked=False,
            image_filename=image_filename,
            prompt=prompt,
            default_response=None,
        )
