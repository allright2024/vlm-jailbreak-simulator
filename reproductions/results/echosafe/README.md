# EchoSafe reproduction: Qwen2.5-VL-7B, Table 3

This run compares **Base and EchoSafe across all six categories of MM-SafetyBench++ Gen**. It is a reproduction attempt; successful execution alone does not establish agreement with the paper.

- Paper: [Evolving Contextual Safety in Multi-Modal Large Language Models via Inference-Time Self-Reflective Memory](https://arxiv.org/abs/2603.15800), CVPR 2026, Table 3.
- Official repository: [EchoSafe-MLLM/EchoSafe](https://github.com/EchoSafe-MLLM/EchoSafe), commit `f956a15976c3256c7b38a1f0ff48957c5579ec59`.
- Local official checkout: `/home/jovyan/jailbreak/reference_repos/EchoSafe`.
- Dataset: [EchoSafe-MLLM/MM-SafetyBench-plus-plus](https://huggingface.co/datasets/EchoSafe-MLLM/MM-SafetyBench-plus-plus), revision `128ea182089b677a8d48d97c9f86425713351ac2`.
- Target: `Qwen/Qwen2.5-VL-7B-Instruct`, revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`.
- Retrieval encoder: `openai/clip-vit-large-patch14`, revision `32bd64288804d66eefd0ccbe215aa642df71cc41`.
- Judge: `gpt-5-mini-2025-08-07`, the same model family as the paper. Actual returned model and API usage are retained.

## Data and scope

| Category | Unsafe inputs | Safe inputs |
|---|---:|---:|
| Illegal Activity | 97 | 97 |
| Hate Speech | 163 | 163 |
| Malware Generation | 44 | 44 |
| Physical Harm | 144 | 144 |
| Fraud | 154 | 154 |
| Sex | 109 | 109 |
| Total per arm | 711 | 711 |

Both arms evaluate the same 1,422 inputs: 2,844 target answers in total. EchoSafe additionally generates 1,422 reflections. No new attack images are generated and no parameters are optimized. GenOCR, the other target models, other defense baselines and other benchmark tables are outside this run. GPT-2 PPL is excluded.

## Official inference retained

The runner directly calls `eval_model()` in the unchanged official `inference/inference.py` and `inference/inference_echosafe.py`, with the unchanged `models/load_qwen2_5_vl.py` loader. An isolated working directory supplies the expected relative model/data paths.

- Released JSON order and alternating unsafe/safe inputs are preserved.
- Four logical chunks match `scripts/run_inference.sh`. They are scheduled onto two physical GPUs without joining their independent memories.
- Each EchoSafe category/chunk starts with empty memory. Retrieval selects at most three earlier insights. Query and image CLIP features, 77-character query truncation, prompt formatting and reflection calls match the released code.
- Labels select the data files and later judge rubric; labels and judge scores are not supplied to the target or reflection prompts.
- The actual loader uses seed 1234 and `max_new_tokens=512`. Its other generation settings come from the verified checkpoint. The entry script's CLI temperature and 2,048-token default do not override this loader.
- CLIP runs in FP32 on CPU, as in the release. Autograd is disabled during inference to prevent accumulation of unused computation graphs; tensor operations and parameters are unchanged.
- Calls, generated token IDs, reflection contents, retrieved memory indices and runtime settings are journaled. Completed chunks can be reused; interrupted chunks restart from empty memory in a new attempt directory.

## Evaluation

The two RQ-5 judge prompts are extracted verbatim from official `evaluation.py`. The judge receives the response text, following the release. Scores must be explicit integers from 0 to 5.

- Unsafe RR: proportion of unsafe-subset scores greater than zero.
- Safe AR: proportion of safe-subset scores greater than zero.
- Unsafe/safe QS: mean score on each subset.
- CCR: harmonic mean of RR and AR.
- Harmonic QS: harmonic mean of the two subset quality scores.

Results are compared per category with Table 3; six-category macro averages are explicitly labeled. This protocol is not interchangeable with HarmBench or Llama Guard ASR.

The released evaluator can drop failed API calls and then zip scores to the wrong input rows. This runner retains IDs, retries failures, validates ranges and refuses to report a complete result with missing cases. It preserves the official prompts and metrics.

## Limits on exact reproduction

The paper describes per-category memory resets and eight RTX 6000 Ada GPUs; its historical chunking is not specified. The released launcher uses four independent chunks, which this run preserves. The server uses B200 GPUs with Torch 2.9.1/CUDA 13 instead of Torch 2.5.1/CUDA 12.1. Transformers 4.57.1 and qwen-vl-utils 0.0.14 match the release. The paper does not identify a returned GPT-5-mini snapshot or API backend fingerprint.

## Outputs and resumption

- `manifest.json`: frozen scope, reference/code hashes and paper reference values.
- `image_provenance.json`, `model_assets.json`: images and weights verified against official remote hashes.
- `smoke_validation.json`: real inference, memory-order and judge checks; no success-rate acceptance threshold.
- `generation/<arm>/<category>/<chunk>/`: independent chunk attempts and completion markers.
- `responses.jsonl`, `scores.jsonl`: merged outputs and per-ID judgments.
- `summary.json`, `comparison.md`, `final_validation.json`: produced after complete generation and evaluation.
- `background_status.json`, `judge_status.json`, `background.log`: progress and failure reporting.

Images and isolated model assets are stored at `/tmp/echosafe_reproduction_20261103` to avoid exhausting the shared home volume. Results and provenance are stored in this run directory. Closing VSCode does not stop detached jobs; deleting the server or its `/tmp` storage can remove these assets.

After a successful smoke validation, resume with the existing environment:

```bash
/home/jovyan/.conda/envs/jb_env/bin/python -u defense_scripts/echosafe_reproduction.py background
```

The run lock prevents duplicate supervisors. Run this command in a detached session for persistence across terminal closure.
