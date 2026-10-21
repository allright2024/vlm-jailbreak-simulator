# `core/` — unified pipeline contract

The shared interface every attack, dataset, target model, and judge agrees on,
so adding an attack (FigStep, HADES, JOOD, CS-DJ, …)
does not require editing the pipeline.

## Concepts (`core/schema.py`)
- **Behavior** — one harmful request from a benchmark; the unit ASR is averaged
  over. `instruction` is the *original* text and is always what the judge
  scores against. Optional fields (`keyword`, `statement`, `images`, `target`)
  cover the union of what the registered attacks need.
- **TestCase** — a concrete model input an attack builds from a Behavior: a text
  prompt + 0..N images, an `attempt` index (Pass@k), and an optional
  `parent_id` (for plug-in attacks like ASO).
- **Completion** — a target-model response to a TestCase, plus its judge `label`.

## Interfaces
- `Dataset` (`core/dataset.py`) — `load() -> list[Behavior]`; never mutates
  source files. Loaders live in `core/datasets/`.
- `Attack` (`core/attack.py`) — `build(behavior, out_dir) -> list[TestCase]`,
  plus `requires_*` flags so the runner skips incompatible (attack, dataset)
  pairs instead of crashing. Target-in-the-loop attacks override
  `build_interactive`.
- `TargetModel` / `Judge` (`core/model.py`) — thin bases, no torch imported here.

## Model loading (`multimodalmodels/`)
Every model the repo uses — the eval target, the HarmBench judge, attack
surrogates (UltraBreak, JPS, JailBound, VisualAdv, DeltaJP) and defense models
(CIDER, JailGuard) — is loaded through `multimodalmodels/loaders.py`; nothing
else calls `from_pretrained` / `torch.load` / `spacy.load`.
- `multimodalmodels/registry.py` pins each HF model id to an exact revision
  (single source of truth; pure data). Loaders use the pin when no revision is
  given.
- `multimodalmodels/__init__.py` is lazy: importing it (or reading a pin) does
  not import transformers; `from multimodalmodels import LLaVA_v1_5` still works.
- `core/tests/test_model_loading.py` fails if code outside `multimodalmodels/`
  loads a model, and checks the pins against the local HF cache.
- `core/targets/native_hf.py` registers LLaVA-1.6/LLaVA-NeXT, Qwen3-VL
  2B/8B, and native-HF InternVL3 2B/8B targets. `multimodalmodels/surrogates.py`
  exposes the same families to the shared CE contract used by VisualAdv,
  DeltaJP, and HADES white-box via `--surrogate-profile`. Model-specific
  JPS/UltraBreak/JailBound adapters are deliberately not inferred from
  target compatibility.

## Datasets (`core/datasets/`)
Import `core.datasets` to register all loaders. Benchmark files are external to
Git; set `VLM_JAILBREAK_DATA_ROOT` to the downloaded/mounted Drive directory.
Logical `data/...` config paths are resolved under that root. Run
`python -m core.validation.data` before a benchmark; see `DATA_LAYOUT.md`.

| Name | Behaviors | Images / behavior | Native fields | eval_type split |
|---|---|---|---|---|
| `AdvBench` | 520 | 0 | `target` | harmful 520 |
| `AdvBenchM` | 170 | 1 (scenario image) | `keyword` | harmful 170 |
| `HADES` | 750 | 6 (opt. steps 0–5) | `keyword`, `keyword_type` | harmful 750 |
| `MM-SafetyBench` | 1,680 | 3 (SD, SD_TYPO, TYPO) | `keyword`, `keyword_type` | harmful 972 / opinion 302 / advice 406 |
| `SafeBench` | 500 | 0 | `statement` | harmful 350 / advice 150 |

- `behavior_id` prefixes: `advbench_NNN`, `<Scenario>_NNNNNN` (HADES),
  `mmsb_SS_NNN`, `safebench_CC_TT`; unique across all datasets.
- `eval_type` follows each benchmark's official evaluation: MM-SafetyBench
  08/13 are scored as *opinion* (refusal to engage) and 10–12 as *advice*, per
  its `evaluation.py`; SafeBench categories 8–10 (legal/financial/health) are
  *advice*.
- Dataset-native prompts live in `meta` (HADES `official_prompt`;
  MM-SafetyBench `rephrased_question` for SD_TYPO/TYPO and
  `rephrased_question_sd` for SD).
- Source data is loaded as released; known upstream quirks are not corrected.
- Jailbreak generation is fail-closed and harmful-only by default:
  `run_generate_test_cases.py` excludes `advice`, `opinion`, and `copyright`
  before attack subsetting and records the exclusions in its manifest. An
  explicit `--eval-types all` is reserved for separate full-benchmark runs and
  must not be merged into the harmful-instruction jailbreak dataset.

## Attacks (`core/attacks/`)
Import `core.attacks` to register them. Ported so far:

| Name | Needs | Notes |
|---|---|---|
| `FigStep` | — | Renders the declarative `statement` (falls back to `instruction`) above an empty numbered list; fixed incitement prompt. Config: `configs/attacks/FigStep.yaml`. |
| `JOOD` | AdvBenchM | OOD-mixes the harmful item with a benign one. `aug` picks image mixing (`mixup`/`cutmix_original`/`cutmix_resizemix`) or text mixing (`textmix_*`). Builds one test case per (harmless item × λ); Pass@k scores "succeed on any mix". Config: `configs/attacks/JOOD.yaml`. |
| `HADES` | HADES | The default runs 3,000 white-box epochs to learn a universal top-quarter patch toward a `Sure...` target, then applies it with the keyword-masked prompt. `HADES_official.yaml` instead replays released image steps 0–5 (`step: all`, Pass@6). |
| `UltraBreak` | any (+ white base image) | **White-box, universal, transferable.** Optimizes ONE adversarial image on a Qwen2-VL-7B surrogate (`requires_surrogate`) with a semantic embedding-space loss + random patch transforms + TV, then pairs it with every query + Targeted Prompt Guidance. Optimized once and cached; `epochs` controls cost. Config: `configs/attacks/UltraBreak.yaml`, data under `data/UltraBreak/`. Needs `qwen-vl-utils`. |
| `VisualAdv` | any (+ benign base image) | **White-box (foundational).** PGD-optimizes a perturbation on a benign image (`requires_surrogate`; LLaVA) so the model emits toxic strings from a derogatory corpus (CE loss); one universal image reused per behavior. `constrained` = L_inf-bounded. Config: `configs/attacks/VisualAdv.yaml`. |
| `DeltaJP` | datasets with per-behavior images (AdvBenchM/HADES/MM-SafetyBench) | **White-box.** A universal L_inf perturbation (delta) added to each behavior's own image (`requires_surrogate`+`requires_images`), fit over a training subset toward affirmative targets, then applied per behavior. Paper: Niu et al. 2402.02309 (deltaJP). Config: `configs/attacks/DeltaJP.yaml`. |
| `JPS` | AdvBench, HarmBench, MM-SafetyBench | **White-box, collaborative.** Co-optimizes a global 32/255 visual perturbation on Qwen2-VL and two textual steer prompts through the official feedback→summarize→revision loop. `JPS.yaml` requires the external upstream CSV/base image and uses Qwen2.5-14B-Instruct. Default target: `qwen2_vl_7b`. Needs `qwen-vl-utils`. |
| `JailBound` | datasets with images (native probe: MM-SafetyBench) | **White-box, per-sample, multimodal.** Trains layer-wise logistic safety boundaries, then jointly updates an L-inf image perturbation and a discrete 20-token suffix with the paper's alignment/semantic/geometric objectives. The paper has no public code; LLaVA-1.5, last-token pooling and suffix NLL are documented adaptations. Config: `configs/attacks/JailBound.yaml`. |

Run stage 1: `python run_generate_test_cases.py --attack FigStep --dataset SafeBench --save_dir runs/<name>`.

## Compositions (`core/composition*.py`)

`run_generate_composed_dataset.py` executes the versioned 18-recipe plan in
`configs/compositions/jailbreak_dataset_v3.yaml`. `ComposedArtifact` carries the
current text/images, while every `StageRecord` authenticates its input and
output text hash, ordered image paths and SHA-256 values, effective config,
duration, and iteration completion. Continuity is enforced between adjacent
records and the final sample identity is SHA-256(normalized text + ordered
image SHA-256 values).

Components that run an existing optimizer preserve its configured loop.
Decomposed stages are separately labelled `adapted_decomposition`; the final
artifact is always `adapted_composition`, never an official paper reproduction.
UltraBreak inserts its trained patch into
a protected side strip, CS-DJ contrast tile, or disjoint bottom region. A
lineage-verified cache shares training across the eight execution groups.
JOOD-Aug generates its configured candidates and chooses one by deterministic
hash, not ASR. DeltaJP and VisualAdv are excluded by config validation.
Periodic transactional checkpoints persist cases and lineage every 10 candidates
and at recipe boundaries; an incomplete checkpoint is eligible for strict resume.
`core.validation.composed_dataset` independently checks harmful-only
parents, counts, duplicate hashes, artifact existence, stage continuity, and
source-group split isolation.

## Validation (`core/validation/`)

Use `python -m core.validation.data`, `composed_dataset`,
`official_reproduction`, or `artifact_lineage` (with the same package prefix).
The CLI arguments are unchanged. `data --data-root PATH` also works when core
was already imported; loader directories and logical image paths both follow
the explicitly selected root. Package imports remain lazy and model-free.

## IO (`core/io.py`)
All three record types persist as JSONL (one object per line), which supports
multiple test cases and attempts per behavior — unlike the old single
`{behavior_id: [image, prompt]}` dict.

## Query/evaluation protocols (`core/protocols/`)

The shared completion CLI defaults to `--protocol standard`. Selecting
`--protocol mml_official` dispatches to `core/protocols/mml.py`, preserving
interleaved CLAS judging, score-5 stopping, and MML ASR/RSR/query metrics without
a separate root runner. Method-specific options require that explicit mode;
the normal three-stage and defense paths are unchanged.

## Registries (`core/registry.py`)
`@register_attack("Name")` / `@register_dataset("Name")`; resolve with
`get_attack` / `get_dataset`. Import `core.datasets` to register loaders.

## Tests

`pytest` always runs the code-only suite. Tests marked `data` run automatically
when the complete external data tree is present; GPU tests remain explicitly
gated by `RUN_GPU_TESTS=1`.
- `python -m core.tests.test_core_contract` — full
  dataset → attack → target → judge → JSONL flow with stand-ins (no models).
- `python -m core.tests.test_datasets` — every loader against the real files
  in `data/`: counts, unique ids, expected fields, image paths, eval_type split.
- `python -m core.tests.test_pipeline` — the three stages with stand-ins,
  including a blocking defense and Pass@k.
- `python -m core.tests.test_figstep` — FigStep rendering, schema, and the
  statement/instruction fallback (no model).
- `python -m core.tests.test_jood` — AdvBenchM loader + JOOD image-mix and
  text-mix construction, reproducibility, requirement gating (no model).
- `python -m core.tests.test_hades` — HADES masked-prompt construction and step
  selection against the real HADES data (no model).
- `python -m core.tests.test_ultrabreak` — UltraBreak config/TPG and the
  vendored semantic-loss / transform / TV utilities (no model; the optimization
  runs on GPU with the Qwen2-VL surrogate).
- `python -m core.tests.test_visual_adv` — VisualAdv config, corpus loading, and
  normalize round-trip (no model; the PGD runs on GPU with the LLaVA surrogate).
- `python -m core.tests.test_delta_jp` — DeltaJP config and affirmative
  target; the universal-delta PGD runs on GPU with the LLaVA surrogate.
- `python -m core.tests.test_jps` — JPS target preprocessing/prompt invariance,
  3-agent role and token-budget contract, strict output parsing, config guards,
  and Qwen2-VL target registration (no model).
- `python -m core.tests.test_jailbound` — layer selection, logistic boundary
  fitting, published losses, joint image/text gradients, L-inf projection, and
  boundary-artifact lineage with a tiny model-free adapter.
