# Third-Party Software, Models, Datasets, and Assets

This document inventories the third-party components that the
**VLM-Jailbreak-Simulator** actually relies on **at runtime** — that is,
components that are imported, vendored in this repository, downloaded as
model weights, read as datasets, or called as remote APIs through
`run_generate_test_cases.py`, `run_generate_completions.py`,
`run_evaluate_completions.py`, the composed-data runner, and their wrappers.
Optional vendored model wrappers are included in this inventory even when
they are not selected by a particular run. The obsolete standalone runners
and archived attack packages have been removed.

These components are **not** covered by the MIT License in
[LICENSE](LICENSE). Each remains subject to its **own upstream license**,
notices, and use restrictions.

> ⚠️ Some entries below are **non-commercial only** (e.g., the Vicuna
> weights that MiniGPT-4 builds on), some are under **community licenses
> with their own restrictions** (e.g., the Llama 2 Community License),
> some are **closed/commercial APIs** governed by the provider's Terms
> of Use (e.g., OpenAI), some upstream repositories have **no explicit
> license attached**, and some assets are subject to **separate copyright**
> (e.g., the Arial font). Users and redistributors are solely responsible
> for confirming and complying with the applicable upstream terms before
> using, fine-tuning, or redistributing each component.

The lists below are **not exhaustive**; transitive dependencies of the
listed packages are governed by their own licenses and should be reviewed
independently.

---

## 1. Vendored / Adapted Attack and Defense Implementations

The following attack implementations are adapted or copied (in whole or
in part) from third-party research repositories and live inside this
repository. Original copyright and license terms remain with the upstream
authors.

| Component (path in this repo)        | Upstream                                                                                                                                                       | Upstream License (as observed) |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| `multi_attacks/figstep/`             | [thuccslab/figstep](https://github.com/thuccslab/figstep)                                                                                                       | **MIT**                        |
| `image_attacks/visual_adv/`          | [Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models](https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models)   | No explicit license            |
| `image_attacks/delta_jp/` | [abc03570128/Jailbreaking-Attack-against-Multimodal-Large-Language-Model](https://github.com/abc03570128/Jailbreaking-Attack-against-Multimodal-Large-Language-Model) | No explicit license            |
| `multi_attacks/hades/`               | [AoiDragon/HADES](https://github.com/AoiDragon/HADES)                                                                                                           | **MIT**                        |
| `multi_attacks/jood/`                | [naver-ai/JOOD](https://github.com/naver-ai/JOOD) — full upstream `LICENSE` file is preserved at `multi_attacks/jood/UPSTREAM_LICENSE`                         | **Apache-2.0**                 |
| `multi_attacks/cs_dj/`               | [TeamPigeonLab/CS-DJ](https://github.com/TeamPigeonLab/CS-DJ) — upstream notice is preserved at `multi_attacks/cs_dj/UPSTREAM_LICENSE`                         | **MIT**                        |
| `multi_attacks/jailbreak_in_pieces/` | [erfanshayegani/Jailbreak-In-Pieces](https://github.com/erfanshayegani/Jailbreak-In-Pieces) — upstream notice is preserved at `multi_attacks/jailbreak_in_pieces/UPSTREAM_LICENSE` | **MIT** |
| `multi_attacks/mml/`                 | [wangyu-ovo/MML](https://github.com/wangyu-ovo/MML) at `5966fca529011e8b5ccff61d70365997d6466abc`; released images are replayed, while the unreleased renderer is a clean-room adapter. Exact prompt literals are read from a user-supplied, hash-pinned checkout and are not redistributed. | No explicit license            |
| `multi_attacks/jailbound/`           | [JailBound, arXiv:2505.19610v3](https://arxiv.org/abs/2505.19610); clean-room implementation of the published equations because no author source repository is publicly linked | No upstream code distributed; paper terms apply |
| `defenses/cider.py`, `defenses/cider_models/` | [PandragonXIII/CIDER](https://github.com/PandragonXIII/CIDER) at `3c08d7e2d52e27b47d1e428a030e8ffda5efae3e` | **Apache-2.0** |
| `defenses/jailguard.py`, `defenses/JailGuard_utils/` | [shiningrain/JailGuard](https://github.com/shiningrain/JailGuard) at `419e1d3d08f4d02998a11a687f925518d2d5dd0c` | No explicit license |
| `defenses/asd.py`, `defenses/asd_assets/` | [AngelAlita/AsD](https://github.com/AngelAlita/AsD) at `ba71efe0972c92e38da156ec76911f9122bbe12f`; upstream notice preserved at `defenses/asd_assets/UPSTREAM_LICENSE` | **MIT** |
| `defenses/ecso.py`, `defenses/ecso_assets/` | [gyhdog99/ECSO](https://github.com/gyhdog99/ECSO) at `3878522f3a59ccb5f0edaa019086b3e954130719`; upstream license preserved at `defenses/ecso_assets/UPSTREAM_LICENSE` | **Apache-2.0** |

The AdaShield defense adapter in `defenses/adashield.py` follows
[SaFo-Lab/AdaShield](https://github.com/SaFo-Lab/AdaShield) at
`b6d02c6747bd6984a99ee2e1db1e52736a6a05cc`. The upstream repository had no
explicit license file at the reviewed revision. Its source is therefore not
vendored here; this project contains an independently integrated adapter and
users must obtain any upstream-generated prompt-pool artifacts themselves.

Upstream licenses verified against each project's `LICENSE` file (and
README, for projects that ship no `LICENSE`) on GitHub at the time of
writing.

> ℹ️ "No explicit license" means that, at the time of review, the upstream
> repository did not contain a `LICENSE` file or equivalent notice.
> Absence of a license does **not** grant any rights to use, copy,
> modify, or redistribute the work. Please contact the upstream authors
> if you need rights beyond what they explicitly permit.

---

## 2. Vendored / Adapted Target-Model Code

The simulator can be configured to load one of several Vision-Language
Models. The wrapper or model-loading code for each is vendored in this
repository.

| Component (path in this repo)              | Upstream                                                                              | Upstream License (as observed) | Runtime role |
| ------------------------------------------ | ------------------------------------------------------------------------------------- | ------------------------------ | ---------------------------- |
| `multimodalmodels/minigpt4/`               | [Vision-CAIR/MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)                      | BSD-3-Clause                   | Optional wrapper, loaded lazily |
| `multimodalmodels/llava/`                  | [haotian-liu/LLaVA](https://github.com/haotian-liu/LLaVA) (via Hugging Face `llava-hf/llava-1.5-7b-hf`) | Apache-2.0 (code)              | Target and attack-surrogate wrapper |
| `multimodalmodels/qwen/`                   | [QwenLM/Qwen-VL](https://github.com/QwenLM/Qwen-VL)                                    | **Tongyi Qianwen License Agreement** | Optional original Qwen-VL wrapper, loaded lazily |
| `multimodalmodels/instructblip/`           | [salesforce/LAVIS](https://github.com/salesforce/LAVIS)                                | BSD-3-Clause                   | Optional wrapper, loaded lazily |

> ℹ️ `multimodalmodels/__init__.py` resolves optional wrappers lazily.
> Importing the package does not load every wrapper or model. The upstream
> notices still apply to the vendored code retained in this repository.

---

## 3. Pre-Trained Model Weights Downloaded or Loaded at Runtime

The simulator downloads or loads the following third-party model weights
when the corresponding model is selected. Weights are separate from the
source-code license; users must obtain them from the upstream provider
and accept the applicable terms. A locally present checkpoint does not
imply permission to redistribute it.

| Component                                    | Upstream                                                                                                                  | Upstream License (as observed)                                                                                | Used by                                  |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| HarmBench classifier (Llama-2-13b-based)     | [https://huggingface.co/cais/HarmBench-Llama-2-13b-cls](https://huggingface.co/cais/HarmBench-Llama-2-13b-cls)             | **Llama 2 Community License** (Llama 2 derivative); HarmBench framework: MIT                                  | `run_evaluate_completions.py` when HarmBench is selected |
| LLaVA v1.5 (7B) — HF release                 | [https://huggingface.co/llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf)                         | LLaVA code: Apache-2.0; weights inherit upstream restrictions (Vicuna / **Llama 2 Community License** family) | ImgJP, DeltaJP, Hades, JOOD     |
| LLaVA v1.6 / LLaVA-NeXT Mistral 7B           | [https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf)       | Apache-2.0 (review the model card)                                                                           | Optional target; VisualAdv/DeltaJP/HADES CE surrogate |
| Qwen2.5-VL-7B-Instruct                       | [https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)                   | Apache-2.0                                                                                                   | Optional target                          |
| Qwen3-VL-2B/8B-Instruct                      | [https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct), [8B](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | Apache-2.0 | Optional target; VisualAdv/DeltaJP/HADES CE surrogate |
| InternVL3-2B/8B native-HF                    | [https://huggingface.co/OpenGVLab/InternVL3-2B-hf](https://huggingface.co/OpenGVLab/InternVL3-2B-hf), [8B](https://huggingface.co/OpenGVLab/InternVL3-8B-hf) | Apache-2.0 | Optional target; VisualAdv/DeltaJP/HADES CE surrogate |
| MiniGPT-4 checkpoint (`checkpoints/pretrained_minigpt4.pth`) and the underlying **Vicuna v0** language-model weights | [Vision-CAIR/MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) (user must download separately)                          | MiniGPT-4: BSD-3-Clause; Vicuna weights are based on the original LLaMA and inherit the **LLaMA Research License (non-commercial)** and the **Vicuna delta weights non-commercial terms**. | FigStep, Visual_Adv                      |
| PixArt-α XL-2-1024-MS                        | [https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS](https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS)         | Refer to the upstream model card (PixArt-α release terms; built on the OpenRAIL family)                       | Hades (text-to-image generation step)    |
| Qwen2.5-3B-Instruct                          | [https://huggingface.co/Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)                         | Apache-2.0                                                                                                   | CS-DJ query decomposition                |
| sentence-transformers CLIP ViT-L/14          | [https://huggingface.co/sentence-transformers/clip-ViT-L-14](https://huggingface.co/sentence-transformers/clip-ViT-L-14)   | Refer to the upstream model card and OpenAI CLIP terms                                                        | CS-DJ image retrieval                    |
| OpenAI CLIP ViT-L/14@336                      | [https://huggingface.co/openai/clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336)       | Refer to the upstream model card and OpenAI CLIP terms                                                        | JailbreakInPieces embedding optimization |
| Guided-diffusion 256×256 unconditional checkpoint | [OpenAI public checkpoint](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt), used by [OpenAI guided-diffusion](https://github.com/openai/guided-diffusion) | MIT for guided-diffusion code; review checkpoint terms upstream | CIDER image denoising |

The Llama 2 Community License contains specific restrictions (including
acceptable-use and large-scale-deployment clauses); review the full text
before any use or redistribution. The Vicuna-derived MiniGPT-4 pipeline
inherits LLaMA / Vicuna **non-commercial** terms. Other model families
may impose their own commercial-use limitations.

---

## 4. Closed-Source APIs Called at Runtime

The Hades and JOOD pipelines make remote calls to closed-source LLM
APIs. Use of these APIs is governed by the provider's Terms of Service
and pricing, not by this repository's MIT License.

| Provider | Model identifier used in code                            | Used by              |
| -------- | --------------------------------------------------------- | -------------------- |
| OpenAI   | `gpt-3.5-turbo` (caption / keyword / category generation) | Hades attack         |
| OpenAI   | `gpt-4-turbo-2024-04-09` (default JOOD target / proposer) | JOOD attack          |

Supply your own API key through `OPENAI_API_KEY` (see `.env.example`). The
provider's content, rate-limit, and acceptable-use policies apply
independently of this repository's license.

---

## 5. Datasets

The simulator reads the following datasets at runtime. Where data files
are vendored in this repository, they were copied or adapted from the
upstream sources listed.

| Component                                                  | Upstream                                                                                                | Upstream License (as observed) | Vendored here? |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------ | -------------- |
| HarmBench standard behaviors (`data/HarmBench/`) | [centerforaisafety/HarmBench](https://github.com/centerforaisafety/HarmBench) | MIT | No; external local data |
| AdvBench harmful-behavior CSVs (`data/advbench/`)          | [llm-attacks/llm-attacks](https://github.com/llm-attacks/llm-attacks)                                    | MIT                            | No; external local data |
| `data/VisualAdv/derogatory_corpus.csv` (VisualAdv targets) | Derived from the Visual Adversarial Examples upstream repo (see §1) | No explicit license | No; external local data |
| `data/FORCE/clean.jpeg` (VisualAdv base image)              | Per the Visual Adversarial Examples upstream repo                                                       | Refer to upstream              | No; external local data |
| LLaVA-CC3M-Pretrain-595K images (`data/CS-DJ/llava_images/`) | [liuhaotian/LLaVA-CC3M-Pretrain-595K](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K) | Refer to the dataset card and source-image terms | No |

> ℹ️ JOOD's `configs/attacks/JOOD.yaml` additionally references
> an external **AdvBenchM** image / prompt set (`data/AdvBenchM/...`).
> AdvBenchM is **not** vendored in this repository; users must obtain it
> from the JOOD upstream and comply with its terms before running the
> JOOD pipeline.

---

## 6. Python Packages (selected, actually imported at runtime)

The simulator imports the following major third-party Python packages
during the default test pipeline. Consult each package's own
distribution metadata for the authoritative license text and complete
dependency tree.

| Component                                  | Upstream                                                                                                       | License (typical) |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------------- | ----------------- |
| `torch` / `torchvision` / `torchaudio`     | [pytorch/pytorch](https://github.com/pytorch/pytorch)                                                          | BSD-3-Clause      |
| `transformers`                             | [huggingface/transformers](https://github.com/huggingface/transformers)                                        | Apache-2.0        |
| `sentence-transformers`                    | [UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers)                                | Apache-2.0        |
| `accelerate`                               | [huggingface/accelerate](https://github.com/huggingface/accelerate)                                            | Apache-2.0        |
| `diffusers` (used by Hades for PixArt-α)   | [huggingface/diffusers](https://github.com/huggingface/diffusers)                                              | Apache-2.0        |
| `spacy` (JailGuard response similarity; optional helpers) | [explosion/spaCy](https://github.com/explosion/spaCy)                                               | MIT               |
| `nltk` (MML word-replacement POS tagging)  | [nltk/nltk](https://github.com/nltk/nltk)                                                                      | Apache-2.0        |
| `datasketch` (used by `eval_utils.py`)     | [ekzhu/datasketch](https://github.com/ekzhu/datasketch)                                                        | MIT               |
| `opencv-python-headless` | [opencv/opencv-python](https://github.com/opencv/opencv-python)                                                | Apache-2.0 (with MIT components) |
| `Pillow`                                   | [python-pillow/Pillow](https://github.com/python-pillow/Pillow)                                                | MIT-CMU / HPND    |
| `PyYAML`                                   | [yaml/pyyaml](https://github.com/yaml/pyyaml)                                                                  | MIT               |
| `httpx`                                    | [encode/httpx](https://github.com/encode/httpx)                                                                | BSD-3-Clause      |
| `openai` (Python SDK; Hades + JOOD)        | [openai/openai-python](https://github.com/openai/openai-python)                                                | Apache-2.0        |
| `tqdm`, `numpy`, `pandas`, `matplotlib`, `scikit-image` | (standard scientific Python stack)                                                                 | BSD / PSF / MIT (per upstream) |

Other packages listed in `requirements.txt` (e.g., `peft`,
`webdataset`, `wandb`, `decord`, `sentencepiece`, `protobuf`,
`beautifulsoup4`, `anthropic`, `visual_genome`, `git+https://github.com/openai/CLIP.git`)
are environment- or method-specific dependencies. They remain governed by
their own upstream licenses when used, independently of which runner is selected.

---

## 7. External Assets

| Component   | Source / Distribution                                                                                                            | Notes                                                                                  |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| `Arial.ttf` | Required by the Hades attack (image text rendering); downloaded by the user from a third-party mirror at install time (see README). | Arial is a proprietary typeface owned by Monotype / Microsoft. Confirm that you have the right to use the font in your environment, or substitute a font you are licensed to use, before running the Hades pipeline. |
| `Super Moods.ttf` | Distributed by the CS-DJ upstream and required for strict official text-card rendering; not vendored here. | Font files may carry terms distinct from the repository's MIT code license. Confirm redistribution and use rights or configure a licensed substitute. |

---

## 8. Notes for Users and Redistributors

- This repository does **not** vendor most of the listed third-party
  packages, model weights, or fonts. Users obtain them directly from the
  upstream sources.
- "No explicit license" entries above do **not** grant any rights;
  absent an explicit license, default copyright applies. Contact the
  upstream authors if you need to use such material.
- **Non-commercial** terms (e.g., the LLaMA Research License /
  Vicuna delta-weights terms that flow into MiniGPT-4) and **community
  licenses** (e.g., the Llama 2 Community License covering the HarmBench
  classifier and the LLaVA v1.5 weights) impose restrictions beyond
  permissive open-source norms. Do not assume commercial use is
  permitted.
- The OpenAI APIs used by Hades and JOOD are subject to OpenAI's Terms
  of Use, content policy, and pricing, none of which are granted by this
  repository's MIT License.
- If you redistribute binaries, containers, fine-tuned weights,
  evaluation outputs, or packaged environments built from this
  repository, you are responsible for satisfying the notice, attribution,
  and license obligations of every bundled or referenced third-party
  component.
- The MIT License in [LICENSE](LICENSE) covers only the original code
  and documentation directly developed by ai.resilience as part of the
  VLM-Jailbreak-Simulator project.
