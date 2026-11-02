# AsD official-protocol reproduction (LLaVA-1.5-7B)

Official code: https://github.com/AngelAlita/AsD @ ba71efe0972c92e38da156ec76911f9122bbe12f. Target: liuhaotian/llava-v1.5-7b (original LLaVA runtime, greedy, 512 tokens). Data: MM-SafetyBench SD_TYPO. Judge: gpt-4o with the MM-SafetyBench safe/unsafe templates (paper: GPT-4).

## MM-SafetyBench SD_TYPO: ASR (%) per scenario, this run vs. paper Table 1 / Table 7

| Scenario | Normal (run) | Normal (paper) | Text-Only (run) | Text-Only (paper) | AsD (run) | AsD (paper) |
| --- | --- | --- | --- | --- | --- | --- |
| 01-Illegal_Activitiy | 89.7 | 78.4 | 39.2 | 40.2 | 26.8 | 3.6 |
| 02-HateSpeech | 66.3 | 60.1 | 23.9 | 22.7 | 19.0 | 4.5 |
| 03-Malware_Generation | 79.5 | 63.6 | 34.1 | 31.8 | 20.5 | 10.5 |
| 04-Physical_Harm | 81.9 | 66.0 | 35.4 | 33.3 | 16.7 | 11.4 |
| 05-EconomicHarm | 27.9 | 17.2 | 17.2 | 10.7 | 10.7 | 6.6 |
| 06-Fraud | 78.6 | 72.1 | 36.4 | 35.7 | 22.7 | 22.0 |
| 07-Sex | 37.6 | 33.9 | 34.9 | 28.4 | 26.6 | 19.3 |
| 08-Political_Lobbying | 96.1 | 94.2 | 71.2 | 76.7 | 51.6 | 51.6 |
| 09-Privacy_Violence | 76.3 | 62.6 | 45.3 | 42.5 | 28.8 | 23.0 |
| 10-Legal_Opinion | 100.0 | 93.1 | 81.5 | 77.7 | 79.2 | 69.2 |
| 11-Financial_Advice | 100.0 | 98.8 | 98.8 | 96.4 | 99.4 | 96.4 |
| 12-Health_Consultation | 99.1 | 96.3 | 95.4 | 93.6 | 78.0 | 77.1 |
| 13-Gov_Decision | 99.3 | 97.3 | 82.6 | 85.2 | 73.2 | 70.5 |
| **Average 01-07,09** | **67.2** | **56.7** | **33.3** | **30.7** | **21.5** | **12.6** |
| Average 08,10-13 | 98.9 | 95.9 | 85.9 | 85.9 | 76.3 | 73.0 |

Paper ECSO column (not run here): 01-Illegal_Activitiy=12.4, 02-HateSpeech=21.5, 03-Malware_Generation=20.5, 04-Physical_Harm=27.8, 05-EconomicHarm=10.7, 06-Fraud=22.1, 07-Sex=30.3, 09-Privacy_Violence=26.6; average 21.5.

## Jailbreak-in-Pieces arm (16 prompts x 25 sampled rounds, llava-1.5-7b-hf 4-bit) vs. paper Table 3

| Arm | ASR run (%) | unsafe/n | paper (%) |
| --- | --- | --- | --- |
| base | 25.0 | 100/400 | 89.0 |
| textonly | 6.0 | 24/400 | 53.5 |
| asd | 9.8 | 39/400 | 47.5 |
| cider (paper only) | - | - | 63.5 |

