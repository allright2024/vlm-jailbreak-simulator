# EchoSafe: Table 3, Qwen2.5-VL-7B, Gen

All six released categories, paired safe/unsafe inputs; GPT-5-mini-2025-08-07 judge.

| Category | Method | Unsafe RR paper / run | Safe AR paper / run | CCR paper / run | Harmonic QS paper / run |
|---|---|---:|---:|---:|---:|
| 01-Illegal_Activity | Base | 29.90 / 35.05 | 100.00 / 100.00 | 45.90 / 51.91 | 2.00 / 1.77 |
| 01-Illegal_Activity | EchoSafe | 83.50 / 16.49 | 95.90 / 58.76 | 89.30 / 25.76 | 3.60 / 0.69 |
| 02-HateSpeech | Base | 30.70 / 36.20 | 100.00 / 98.77 | 47.00 / 52.98 | 2.10 / 1.90 |
| 02-HateSpeech | EchoSafe | 92.60 / 25.15 | 93.80 / 63.19 | 93.20 / 35.98 | 3.60 / 0.94 |
| 03-Malware_Generation | Base | 11.40 / 9.09 | 100.00 / 100.00 | 20.50 / 16.67 | 1.00 / 0.59 |
| 03-Malware_Generation | EchoSafe | 95.50 / 11.36 | 91.60 / 47.73 | 93.50 / 18.36 | 3.80 / 0.57 |
| 04-Physical_Harm | Base | 20.10 / 20.14 | 100.00 / 99.31 | 33.40 / 33.49 | 1.30 / 1.27 |
| 04-Physical_Harm | EchoSafe | 81.00 / 15.28 | 88.00 / 53.47 | 84.40 / 23.77 | 3.30 / 0.70 |
| 06-Fraud | Base | 19.50 / 21.43 | 100.00 / 100.00 | 32.70 / 35.29 | 1.30 / 1.27 |
| 06-Fraud | EchoSafe | 79.90 / 25.32 | 98.10 / 55.19 | 88.10 / 34.72 | 3.60 / 0.91 |
| 07-Sex | Base | 13.80 / 10.09 | 99.10 / 97.25 | 24.20 / 18.29 | 1.00 / 0.65 |
| 07-Sex | EchoSafe | 70.60 / 12.84 | 89.00 / 53.21 | 78.70 / 20.69 | 3.00 / 0.61 |

Macro averages (average of the six category metrics):

{
  "Base": {
    "unsafe_rr": 21.999662989632156,
    "unsafe_qs": 0.8971007456184132,
    "safe_ar": 99.22104468542386,
    "safe_qs": 2.39918185424299,
    "ccr": 34.77001343776745,
    "harmonic_qs": 1.2428929197176497
  },
  "EchoSafe": {
    "unsafe_rr": 17.743057626215123,
    "unsafe_qs": 0.7201161759696414,
    "safe_ar": 55.25972999427166,
    "safe_qs": 0.7725026081264247,
    "ccr": 26.54615524372193,
    "harmonic_qs": 0.7374739511433348
  }
}

Deviations and scope:

- Reproduces Table 3 Qwen2.5-VL-7B Base and EchoSafe under Gen across all six categories; other models, defenses and GenOCR are outside this run.
- Preserves four independent memory chunks from the released launcher, scheduled on two physical GPUs. The paper only specifies per-category reset; its exact historical chunking is not stated.
- Torch 2.9.1/CUDA 13/B200 replaces Torch 2.5.1/CUDA 12.1/RTX 6000 Ada. Transformers 4.57.1 and qwen-vl-utils 0.0.14 match the release.
- Disables autograd for inference to avoid retaining CLIP computation graphs in memory; model parameters and numeric operations are unchanged.
- Uses the official GPT-5-mini judge family pinned to 2025-08-07. The authors do not report their returned API snapshot or backend fingerprint.
- Judge retries preserve sample IDs; failed requests are never dropped or scored as zero. This fixes the released evaluation script's unsafe zip alignment after API errors.
