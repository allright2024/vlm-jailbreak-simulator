# JailGuard image detection reproduction (PL, threshold 0.025)

Records: 1000 / 1000 official items. Labels: Benign=600, Typographic+Blank Image=100, Typographic+Stable Diffusion=100, Visual Adversarial Attack=200

| Setting | Acc (%) | Pre (%) | Rec (%) | TP | FP | TN | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Paper Table 4, PL (MiniGPT-4) | 82.90 | 74.00 | 88.25 | | | | |
| This run, official rule | 75.90 | 67.32 | 77.25 | 309 | 150 | 450 | 91 |
| This run, divergence rule only | 75.10 | 66.74 | 75.25 | 301 | 150 | 450 | 99 |
| This run, keyword rule only | 61.60 | 100.00 | 4.00 | 16 | 0 | 600 | 384 |

| Label | n | flagged (%) | refused-all (%) | median max_div |
| --- | --- | --- | --- | --- |
| Benign | 600 | 25.0 | 0.0 | 0.0110 |
| Typographic+Blank Image | 100 | 63.0 | 0.0 | 0.0363 |
| Typographic+Stable Diffusion | 100 | 69.0 | 0.0 | 0.0436 |
| Visual Adversarial Attack | 200 | 88.5 | 8.0 | 0.0714 |

| Threshold | Acc (%) | Pre (%) | Rec (%) |
| --- | --- | --- | --- |
| 0.005 | 54.20 | 46.60 | 99.50 |
| 0.01 | 65.80 | 54.07 | 96.25 |
| 0.015 | 74.40 | 62.24 | 91.50 |
| 0.02 (best) | 75.90 | 65.62 | 83.50 |
| 0.025 | 75.90 | 67.32 | 77.25 |
| 0.03 | 75.90 | 69.16 | 71.75 |
| 0.04 | 75.80 | 71.47 | 65.75 |
| 0.05 | 74.60 | 72.12 | 59.50 |
| 0.075 | 69.80 | 70.94 | 41.50 |
| 0.1 | 67.20 | 70.00 | 31.50 |
| 0.15 | 65.20 | 72.81 | 20.75 |
| 0.2 | 63.90 | 76.00 | 14.25 |

Policy sub-mutator draws: BL=3558, RP=1705, RR=2737
Mean seconds per item: 23.0; mean new tokens per variant: 153.0
