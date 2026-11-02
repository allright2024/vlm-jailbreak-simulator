# AdaShield official-protocol reproduction (qr, LLaVA-1.5-13B)

Official code: https://github.com/SaFo-Lab/AdaShield @ b6d02c6747bd6984a99ee2e1db1e52736a6a05cc. Target: liuhaotian/llava-v1.5-13b (original LLaVA runtime, T=0.2, top_p=0.7, 3000 tokens). Defender: lmsys/vicuna-13b-v1.5. Recheck judge: gpt-4o (paper/upstream: GPT-4 / gpt-4-1106-preview).

| Arm | ASR run (%) | Recheck run (%) | n | paper ASR | paper Recheck |
| --- | --- | --- | --- | --- | --- |
| vanilla | 77.20 | 77.75 | 1443 | 75.75 | 67.71 |
| static | 20.37 | 20.72 | 1443 | 24.43 | 20.61 |
| adaptive | 10.26 | 12.34 | 1443 | 15.22 | 15.43 |

Per-scenario ASR / Recheck (%):

| Scenario | vanilla | static | adaptive |
| --- | --- | --- | --- |
| 01-Illegal_Activitiy | 33.3 / 77.4 | 1.2 / 3.6 | 1.2 / 2.4 |
| 02-HateSpeech | 68.3 / 56.8 | 4.3 / 0.7 | 4.3 / 1.4 |
| 03-Malware_Generation | 71.1 / 57.9 | 5.3 / 2.6 | 5.3 / 2.6 |
| 04-Physical_Harm | 62.6 / 79.7 | 8.1 / 5.7 | 4.9 / 4.9 |
| 05-EconomicHarm | 88.7 / 50.9 | 16.0 / 19.8 | 15.1 / 20.8 |
| 06-Fraud | 65.9 / 84.1 | 5.3 / 5.3 | 5.3 / 6.8 |
| 07-Sex | 90.4 / 87.2 | 42.6 / 47.9 | 27.7 / 34.0 |
| 08-Political_Lobbying | 96.2 / 81.7 | 18.3 / 24.4 | 13.7 / 22.9 |
| 09-Privacy_Violence | 59.2 / 90.0 | 10.8 / 16.7 | 14.2 / 17.5 |
| 10-Legal_Opinion | 81.1 / 87.4 | 11.7 / 13.5 | 5.4 / 8.1 |
| 11-Financial_Advice | 91.6 / 93.0 | 58.7 / 60.8 | 16.8 / 18.9 |
| 12-Health_Consultation | 88.3 / 69.1 | 58.5 / 45.7 | 5.3 / 5.3 |
| 13-Gov_Decision | 93.8 / 78.9 | 17.2 / 13.3 | 10.9 / 9.4 |

AdaShield-A training: 160 training samples, 104 accepted prompts, mean refinements 0.35.
