# VisualAdv: one-benchmark reproduction check

Source: https://arxiv.org/abs/2306.13213v2

Public RTP challenging subset: 1199 prompts (paper: 1225); three repeats.
Blank measurements indicate incomplete coverage. Exact historical reproduction is unverified.

| Input | Scorer | Attribute | Paper (%) | Measured mean ± std (%) | Δ pp | Status |
|---|---|---|---:|---:|---:|---|
| clean | detoxify | any | 29.10 | 30.80 ± 0.92 | 1.70 | complete |
| clean | detoxify | identity_attack | 1.50 | 1.33 ± 0.34 | -0.17 | complete |
| clean | detoxify | obscene | 22.40 | 23.80 ± 1.00 | 1.40 | complete |
| clean | detoxify | severe_toxicity | 0.60 | 0.53 ± 0.08 | -0.07 | complete |
| clean | detoxify | insult | 11.00 | 12.09 ± 0.12 | 1.09 | complete |
| clean | detoxify | threat | 0.90 | 0.92 ± 0.18 | 0.02 | complete |
| clean | detoxify | toxicity | 28.90 | 30.33 ± 0.79 | 1.43 | complete |
| eps16 | detoxify | any | 46.40 | 47.65 ± 0.46 | 1.25 | complete |
| eps16 | detoxify | identity_attack | 5.00 | 4.78 ± 0.35 | -0.22 | complete |
| eps16 | detoxify | obscene | 33.70 | 34.22 ± 0.57 | 0.52 | complete |
| eps16 | detoxify | severe_toxicity | 2.30 | 1.97 ± 0.49 | -0.33 | complete |
| eps16 | detoxify | insult | 23.60 | 24.66 ± 0.70 | 1.06 | complete |
| eps16 | detoxify | threat | 2.20 | 2.28 ± 0.26 | 0.08 | complete |
| eps16 | detoxify | toxicity | 46.10 | 47.12 ± 0.65 | 1.02 | complete |
