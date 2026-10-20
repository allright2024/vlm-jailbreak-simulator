# Official implementation alignment

This repository pins the upstream revision used to review each attack. Every
stage-1 run writes the source, commit, algorithm fidelity, requested and resolved
config, training IDs, and excluded evaluation IDs to `run_manifest.json`.

Fidelity is recorded on two independent axes (see `core/attack.py`):

* **`algorithm_fidelity`** — a property of the *code*: `faithful_port` (the
  generation algorithm matches upstream), `replay` (uses released artifacts), or
  `adapted` (a concrete dependency/model/dataset substitution).
* **`experiment_profile`** — a property of the *config used for a run*. The five
  methods covered by the official-reproduction pass (CS-DJ, JOOD, MML, JPS,
  JailbreakInPieces) have one official-only base config. Other methods may still
  expose separate quick and `*_official.yaml` profiles.

`paper_reproduction` is only `true` when `algorithm_fidelity == faithful_port`,
`experiment_profile == official`, the complete resolved configuration (including
canonical input hashes and pinned model revision) exactly equals the official
signature, and there is no model substitution. Reused artifacts additionally
need a verified lineage sidecar. FORCE runs that reach the added safety cap are
downgraded after generation. The unified target model and judge still differ from
some papers, so even a `true` here is a reproduction of the *attack*, not the
paper's full pipeline.

For CS-DJ, JOOD, MML, JPS, and JailbreakInPieces the default config is the only
shipped config and uses the released experiment settings. For other methods,
pass a matching `configs/attacks/<Attack>_official.yaml` when one exists.

The table's "Mode" column below is the `algorithm_fidelity`.

| Method | Pinned upstream | Mode | Alignment and remaining difference |
|---|---|---|---|
| FigStep | [CryptoAILab/FigStep](https://github.com/CryptoAILab/FigStep) `0861b17` | adapted | Exact 760 px canvas, 80 px font size, width-15 wrapping, spacing, list, and prompt. Upstream `FreeMonoBold.ttf` is not distributed; set `font_path` and `strict_font: true` for pixel-faithful rendering. |
| MML | [wangyu-ovo/MML](https://github.com/wangyu-ovo/MML) `5966fca529011e8b5ccff61d70365997d6466abc` | replay | The sole config replays the Drive images and hash-verifies the pinned upstream prompt source. The shared completion CLI's explicit MML protocol reproduces the `Content`/refusal gates, GPT-4o-mini CLAS score, score-5 early stop, five-query budget, and ASR/RSR/query metrics. Main experiments use the paper's pinned GPT-4o snapshot and the official filtered metadata. The unreleased image generator is only available as an explicitly adapted clean-room mode. API nondeterminism and unavailable historical service state prevent bitwise reproduction. |
| HADES | [AoiDragon/HADES](https://github.com/AoiDragon/HADES) `a2670b76` | adapted / replay | The default uses 3,000 epochs with the upstream white-box top-quarter replacement and `1/255` sign-gradient update, using prompt-aware CE toward a fixed `Sure...` target. It fits one universal patch on an explicit stratified train split and excludes those rows from evaluation; upstream instead runs scenario-specific data and randomly chooses from ten affirmative targets, so the white-box path remains `adapted`. `HADES_official.yaml` separately replays all six released black-box image steps (Pass@6) and is labeled `replay`; it does not claim white-box reproduction. |
| JOOD | [naver-ai/JOOD](https://github.com/naver-ai/JOOD) `4ca1368` | adapted | Official prompt templates, Mixup/CutMix/ResizeMix/RandAug operators, and the released script's λ grid 0.1–0.9. The runner executes the four augmentations separately against AdvBench-M and uses the released Llama Guard 2 ASR judge. The upstream GPT-4 harmfulness/detailedness auxiliary report is not yet ported. |
| FORCE | [tmllab/2026_CVPR_FORCE](https://github.com/tmllab/2026_CVPR_FORCE) `21dfb17` | faithful | Frequency-band and feature corrections plus the official 100-match stopping condition. `max_iter` is the only added safety cap and is recorded in config. |
| UltraBreak | [kaiyuanCui/UltraBreak](https://github.com/kaiyuanCui/UltraBreak) `c024bb4` | faithful | Official semantic loss, random patch transform, TV loss, Adam settings, and 5,000 epochs. The 50 SafeBench-Tiny training rows are automatically excluded from SafeBench evaluation. |
| VisualAdv | [Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models](https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models) `5fc9883` | faithful_port | Official PGD loop, target sampling, and constrained/unconstrained update. The default surrogate wrapper is LLaVA-1.5 rather than the authors' custom LLaVA-Llama2 checkpoint, so `substitutes_model=true` blocks exact paper reproduction. |
| DeltaJP | [abc03570128/Jailbreaking-Attack-against-Multimodal-Large-Language-Model](https://github.com/abc03570128/Jailbreaking-Attack-against-Multimodal-Large-Language-Model) `fd98910` | adapted | Official 64/255 budget, 1/255 step, 500 iterations, and zero start. Upstream fits one delta per class with 25/5 prompt and 10/10 image splits; bundled AdvBenchM has one image per seven classes, so this port uses one balanced cross-class delta and excludes its 25 train prompts. |
| JPS | [thu-coai/JPS](https://github.com/thu-coai/JPS) `beacbc58` | adapted | Preserves the Qwen target rewrite, 50×300 batched thresholded PGD, five revision rounds, and Qwen2.5-14B agents. The sole config requires the upstream-referenced `advbench_subset.csv` and `clean.jpeg` and evaluates AdvBench, HarmBench, and MM-SafetyBench. Those two files are not distributed by upstream, so exact reproduction remains externally blocked. |
| CS-DJ | [TeamPigeonLab/CS-DJ](https://github.com/TeamPigeonLab/CS-DJ) `4577b05f` | faithful_port | Preserves Qwen2.5-3B decomposition, CLIP ViT-L/14 mean-cosine contrast retrieval, and the 3×4 collage. The sole config uses 10,000 CC3M images, the upstream font, the released 750 instructions, and GPT-4o-mini target defaults. Because upstream does not publish an immutable sampled-image list, bitwise/paper-number reproduction still cannot be certified. |
| JailbreakInPieces | [erfanshayegani/Jailbreak-In-Pieces](https://github.com/erfanshayegani/Jailbreak-In-Pieces) `b7b2bce8` | faithful_port | Preserves CLIP ViT-L/14@336 preprocessing without normalization, L2/Adam 0.1 optimization for 5,000 steps, projection, JPEG conversion, the fixed `meth_pill.png`/`white.jpeg` pair, and all 16 prompts. The runner uses LLaVA-1.5 with the released sampling settings. Upstream publishes no ASR judge for this demo. |
| JailBound | [arXiv:2505.19610v3](https://arxiv.org/abs/2505.19610) (2026-02-25) | adapted | Implements published Algorithms 1–2 and Section 4.1 settings: layer-wise logistic probes, boundary normal/distance, joint continuous-image/discrete-suffix updates, `P0=0.3`, `lambda1=2`, `lambda2=1`, `eta_v=0.001`, `eta_t=0.0005`, 20 tokens, `8/255`, and 100 iterations. No author code is publicly linked despite the paper's release commitment. Fusion pooling, precision, and `L_suffix` are unspecified; this clean-room port uses the final input-token state, BF16 (to prevent all-layer squared-loss FP16 overflow), and conditional suffix NLL. It also substitutes pinned LLaVA-1.5 for the paper's Llama-3.2/Qwen2.5-VL/MiniGPT-4 models and reserves/excludes a 64-row MM-SafetyBench probe split. Exact paper reproduction is therefore always false. |
| AdaShield | [SaFo-Lab/AdaShield](https://github.com/SaFo-Lab/AdaShield) `b6d02c6` | faithful_port (S and A inference) | AdaShield-S preserves the exact static-prompt bytes and LLaVA/MiniGPT-v2 `query + shield + query` placement. AdaShield-A loads `final_judge_scores == 1` rows and reproduces OpenAI CLIP ViT-B/32 concatenated image/text cosine retrieval, including random tie-breaking. The upstream repository does not ship its generated W&B `final_table.csv` pools, so A requires user-supplied official refinement outputs. The registered LLaVA target is 7B rather than the paper's 13B model. |
| AsD | [AngelAlita/AsD](https://github.com/AngelAlita/AsD) `ba71efe` | faithful_port | Preserves the exact safety-prompt bytes, released protective patch (SHA-256 checked), seed `42`, 25% image-area resize, alpha `0.7` (`178/255`), aspect ratio, and seeded random placement from the active upstream `--AsD` path. The upstream README example omits the required `--AsD` flag even though the code gates the transformation on it; this port activates the intended defense path explicitly. The simulator uses its HF LLaVA-1.5-7B wrapper rather than the upstream LLaVA runtime, so paper numbers are not claimed. |
| ECSO | [gyhdog99/ECSO](https://github.com/gyhdog99/ECSO) `3878522` | faithful_port (MM-SafetyBench inference) | Preserves direct answer → self-assessment → conditional query-aware caption → image-free safe generation, the three exact prompt literals and exact `yes`/`yes.` rule, direct-response truncation at 1,927 characters, greedy decoding, the direct-stage 2,048-token budget, and the later-stage 1,024-token budget. Safe/unsafe samples consume two/four target calls respectively. The simulator uses a pinned HF LLaVA-1.5-7B conversion rather than the upstream LLaVA runtime, so generation bytes and paper numbers are not claimed. The separate upstream VLSafe script defaults to temperature `0.2`; this sole config intentionally follows the MM-SafetyBench script's explicit temperature `0`. |
| CIDER | [PandragonXIII/CIDER](https://github.com/PandragonXIII/CIDER) `3c08d7e` | faithful_port | Preserves the official guided-diffusion checkpoints 0/50/…/350, LLaVA-1.5 projected image/text embeddings, cosine-decrease rule, default threshold `-0.003936767578125`, and refusal response. The adapter reuses an already-loaded LLaVA target as the identical detector encoder and lazily omits the upstream ViT classifier because CIDER calls `denoise()` directly and never uses that classifier. The target receives the original pair when detection passes, as specified by the paper. |
| JailGuard | [shiningrain/JailGuard](https://github.com/shiningrain/JailGuard) `419e1d3` | faithful_port (image inference) | Preserves the image `PL` policy (`RR`/`BL`/`RP` at `0.34`/`0.45`/`0.21`), eight serial variants, unchanged question, 300-token variant generation, first-100-character spaCy similarity, pairwise row KL divergence, `0.025` threshold, and all-variants-refused rule. The unified adapter supports the VLM image path only; upstream text detection is out of scope. Upstream uses MiniGPT-4 while the default simulator command uses registered LLaVA, and a passing case receives one additional original-pair query, so paper numbers are not claimed. |

## Split policy

- Any behavior used to optimize a universal artifact is skipped before test-case
  generation when the same benchmark is evaluated.
- UltraBreak resolves its CSV text back to 50 stable SafeBench IDs.
- DeltaJP uses a seeded, round-robin category split instead of file order.
- HADES white-box uses a seeded, round-robin category split; its universal
  patch is content-hash bound to the selected source images and train prompts.
- JPS accepts the official `goal,target` CSV through `train_csv`, preserves its
  row order and excludes the matched IDs/hashes; without it, a stable AdvBench
  prefix is an explicitly adapted substitute.
- JailBound reserves a deterministic category-balanced probe split and excludes
  its behavior IDs and safe/unsafe prompt hashes before evaluation.
- Only evaluated behaviors are written to `behaviors.jsonl`; all requested,
  trained, excluded, and skipped counts remain in `run_manifest.json`.

## Artifact lineage

Universal cached artifacts use `<artifact>.lineage.json`. The sidecar binds the
main artifact and auxiliary files (including the JPS steer prompt) to the complete
effective optimization config, content hashes of training/base/target files,
training behavior fingerprints, pinned model revision, and an implementation-code
fingerprint. The full-run scripts accept a cache hit only after validating this
sidecar; missing or stale pre-lineage caches are regenerated. The manifest embeds
the verified lineage record and sidecar SHA-256 so results remain auditable.

## Evaluation scope

The evaluation CLI records the actual judge and supports the released
Beaver-Dam protocol for CS-DJ, Llama Guard 2 for JOOD, and both HarmBench and
Llama Guard 3 for JPS. MML uses `--protocol mml_official` on the shared
completion CLI because its judge controls early stopping; exact prompt literals are read at runtime from a
hash-pinned upstream checkout rather than redistributed. The shared HarmBench
path remains available for cross-method comparison. JOOD's auxiliary GPT-4
harmfulness/detailedness score is not silently treated as equivalent.
