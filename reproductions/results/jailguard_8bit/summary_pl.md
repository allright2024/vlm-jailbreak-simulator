# JailGuard image detection reproduction (PL, threshold 0.025)

Records: 1000 / 1000 official items. Labels: Benign=600, Typographic+Blank Image=100, Typographic+Stable Diffusion=100, Visual Adversarial Attack=200

| Setting | Acc (%) | Pre (%) | Rec (%) | TP | FP | TN | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Paper Table 4, PL (MiniGPT-4) | 82.90 | 74.00 | 88.25 | | | | |
| This run, official rule | 77.00 | 69.23 | 76.50 | 306 | 136 | 464 | 94 |
| This run, divergence rule only | 76.60 | 68.95 | 75.50 | 302 | 136 | 464 | 98 |
| This run, keyword rule only | 61.00 | 100.00 | 2.50 | 10 | 0 | 600 | 390 |

| Label | n | flagged (%) | refused-all (%) | median max_div |
| --- | --- | --- | --- | --- |
| Benign | 600 | 22.7 | 0.0 | 0.0108 |
| Typographic+Blank Image | 100 | 61.0 | 1.0 | 0.0352 |
| Typographic+Stable Diffusion | 100 | 68.0 | 0.0 | 0.0398 |
| Visual Adversarial Attack | 200 | 88.5 | 4.5 | 0.0716 |

| Threshold | Acc (%) | Pre (%) | Rec (%) |
| --- | --- | --- | --- |
| 0.005 | 54.40 | 46.72 | 99.75 |
| 0.01 | 67.30 | 55.21 | 96.75 |
| 0.015 | 75.80 | 63.67 | 92.00 |
| 0.02 | 76.70 | 66.60 | 83.75 |
| 0.025 | 77.00 | 69.23 | 76.50 |
| 0.03 (best) | 77.70 | 71.74 | 73.00 |
| 0.04 | 76.10 | 73.20 | 63.50 |
| 0.05 | 73.70 | 72.46 | 55.25 |
| 0.075 | 70.00 | 71.55 | 41.50 |
| 0.1 | 68.00 | 72.99 | 31.75 |
| 0.15 | 65.20 | 74.53 | 19.75 |
| 0.2 | 62.40 | 73.08 | 9.50 |

Policy sub-mutator draws: BL=3558, RP=1705, RR=2737
Mean seconds per item: 27.1; mean new tokens per variant: 143.6
