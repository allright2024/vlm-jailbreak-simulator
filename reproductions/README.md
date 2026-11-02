# 논문 재현 실험 기록

2026-11-02 기준, 공격 **10종**과 방어 **6종**의 **24개 실험 기록**을 정리했다.
완료된 실행과 논문 성능의 일치는 별개다. 아래 수치는 저장된 결과에서 가져왔으며,
이 공개 작업에서 새 모델 추론, API 평가 또는 재학습을 수행하지 않았다.
공식 산출물 평가, 직접 최적화, 자동 평가 대체, 소규모 검증을 구분한다.

## 공격 결과

단위는 별도 표시가 없으면 %. ASR@5는 최대 다섯 번 중 한 번 이상 성공한 질문의 비율이다.
각 행의 데이터·판정기가 다르므로 공격 간 순위표로 사용하지 않는다.

| 방법 / 범위 | 대상 모델·데이터 | 측정 | 논문·공식 기준 | 해석 / 원본 기록 |
|---|---|---|---|---|
| FigStep, 500개 자동 재평가 | LLaVA-1.5-7B, SafeBench 500 | vanilla 50.6 → FigStep 89.0 ASR@5 | 인간 평가 57.4 → 84.0 | Qwen2.5-14B의 범주별 자동 rubric. 인간 평가는 미완료. [결과](results/figstep/summary.json) |
| MML-M | GPT-4o-2024-08-06, SafeBench 500 | 97.8 (489/500), FigStep 19.0 | Table 1: 97.6, FigStep 33.0 | MML-M 수치는 근접, 기준선 차이 있음. GPT-4o-mini CLAS score 5, 최대 5회. [CSV](results/mml/comparison.csv) |
| JOOD Mixup | GPT-4-Turbo-2024-04-09, AdvBench-M FW 17 | vanilla 5.88 → JOOD 41.18; HF 0.294 → 6.529 | Table 1 GPT-4V/FW: 12 → 47; HF 0.6 → 7.2 | 이미지+텍스트 한 열만 평가. Llama Guard 2 ASR와 GPT-4 HF를 구분. [CSV](results/jood/comparison.csv) |
| CS-DJ | 원본 OneVision Chat 7B, 5범주 750 | 53.47 (401/750) | Table 6: 55.87 | Beaver-Dam; 생성 실패 3개도 분모 유지. SDPA MATH로 퇴화 응답 수정. [CSV](results/cs_dj/comparison.csv) |
| HADES 공개 이미지 | 원본 LLaVA-1.5-7B, 750 | 66.67 (500/750) | Table 2 +Adv image: 90.26 | 논문 수치 미재현. 공개 최종 이미지와 OCR 재시도 프로토콜 평가. [CSV](results/hades/comparison.csv) |
| HADES issue 14 검증 | 동일 모델, 범주별 10개, 총 50 | 동일 기하 blank 64 → 학습 후 70 | 직접 대응하는 논문 수치 없음 | 최종 크기에서 최적화하는 소규모 검증; 750개 결과로 해석하지 않음. [CSV](results/hades_issue14/comparison.csv) |
| JPS 공개 산출물 | Qwen2-VL-7B, HarmBench 200 | 최종 5라운드 88.5 | Table 3: 89.5 | 1·2라운드는 61/57로 논문 92/90.5와 불일치. [CSV](results/jps_released/comparison.csv) |
| JPS 직접 최적화 | AdvBench-subset 50 학습, HarmBench 200 평가 | 최종 86.5; 원본 질문으로 판정하면 85.0 | Table 3: 89.5 | 전체 공격 입력 판정이 기본 비교. 평가용 200개를 학습하지 않음. MIFR 미측정. [CSV](results/jps_trained/comparison.csv) |
| UltraBreak 공개 이미지 | Qwen2-VL / Qwen2.5-VL, SafeBench 315 | 공식 추출 83.49 / 60.63; 응답만 79.68 / 55.87 | Table 1: 81.59 / 60.32 | 공식 추출기가 입력의 `assistant`도 찾는 문제 때문에 두 결과를 병기. [CSV](results/ultrabreak_released/comparison.csv) |
| UltraBreak 직접 최적화 | 동일 평가, 무작위 이미지에서 1,300 업데이트 | 공식 추출 25.08 / 27.30; 응답만 23.17 / 25.40 | Table 1: 81.59 / 60.32 | 학습 완료, 논문 성능 미재현. 논문과 공개 코드의 수식 차이 기록. [결과](results/ultrabreak_trained/summary.json) |
| Jailbreak in Pieces | Table 6 변형, 역사적 LLaVA-13B 후보, 총 1,100 응답 | 원본 집계·주제별 자동 ASR 보존 | 논문은 인간 평가 | 체크포인트·이미지 대응 미확인. Table 1 원본 32개 셀은 미실행. [Table 6](results/jip_table6/summary.json), [Table 1 상태](results/jip_table1/table1_status.json) |
| VisualAdv 공개 이미지 | MiniGPT-4/Vicuna v0, RTP challenging 1,199 × 3회 | Detoxify Any: clean 30.80±0.92 → eps16 47.65±0.46 | Table 2: 29.1±1.0 → 46.4±1.1 | 공개 데이터는 논문의 1,225개와 다름. [결과](results/visualadv_released/report.md) |
| VisualAdv 직접 최적화 | 동일 평가 | clean 30.80±0.92 → eps16 54.77±0.31 | 29.1±1.0 → 46.4±1.1 | 증가 경향 확인, 수치 불일치. [결과](results/visualadv_trained/report.md) |
| JailBound | Qwen2.5-VL, 법률/금융/건강 122/160/101개 | joint 비거절률 37.70 / 67.50 / 67.33 | Table 3: 87.91 / 94.87 / 100 | 자체 probe·분할, 비거절 대리지표. 유해 요청 이행률로 해석 불가. [CSV](results/jailbound/table3_comparison.csv) |

FigStep의 초기 350개 HarmBench 평가는 [별도 기록](results/figstep_harmbench/summary.json)이며,
500개 rubric 평가와 합치지 않았다. CS-DJ의 기존 GPT-4o 실험도
[과거 결과](results/cs_dj_gpt4o/comparison.csv)로 보존했다.
JPS의 원본 질문 기반 HarmBench ASR은 논문의 QwQ-32B **MIFR이 아니다**.

## 방어 결과

ASR은 낮을수록, harmless rate와 accuracy는 높을수록 좋다.
검출기의 우회율과 실제 타깃 응답의 ASR은 서로 다른 지표다.

| 방법 | 범위 | 측정 | 논문·공식 기준 | 해석 / 기록 |
|---|---|---|---|---|
| AdaShield | LLaVA-1.5-13B, FigStep 430; vanilla/static/adaptive | ASR 68.37 / 3.26 / 3.95 | 70.47 / 26.05 / 10.47 | 방어 경향 확인, 절대값 차이. GPT-4o recheck 대체 별도 보고. [결과](results/adashield_figstep/summary.md) |
| AdaShield | 같은 모델, QR 1,443 | ASR 77.20 / 20.37 / 10.26 | 75.75 / 24.43 / 15.22 | 학습·평가 분할과 recheck 기록 포함. [결과](results/adashield_qr/summary.md) |
| AsD | LLaVA-1.5-7B, MM-SafetyBench SD_TYPO 8개 유해 범주 평균 | ASR base 67.2 → AsD 21.5 | 56.7 → 12.6 | GPT-4o 판정으로 대체; 논문 수치 미일치. JIP 변형은 별도 표. [결과](results/asd/summary.md) |
| CIDER | 공식 test 160 × 8이미지; 공격 응답 640쌍 | 공식 tau DSR 86.25, FPR 8.59; ASR 58.44 → 6.72 | README DSR 79.69; 논문 ASR 약 60 → 0 | 임계값 재산정 결과도 보존. 검출률과 응답 ASR을 구분. [결과](results/cider/summary.md) |
| ECSO | LLaVA-1.5-7B, 8범주 macro harmless rate | SD_TYPO 32.1 → 73.5; TYPO 34.6 → 71.1; SD 70.2 → 81.6 | 방어 후 86.4 / 90.3 / 95.4 | GPT-4o가 역사적 GPT-4를 대체. 개선 확인, 절대값 불일치. [결과](results/ecso/summary.md) |
| JailGuard | MiniGPT-4, PL, 공식 이미지 1,000개 | Acc 75.90 / Precision 67.32 / Recall 77.25 | Table 4: 82.90 / 74.00 / 88.25 | fp16·배치 생성 등 실행 환경 차이; threshold 0.025. [결과](results/jailguard/summary_pl.md) |
| VLMShield | 공개 검출기, AdvBench-M 242 / VLSafe 1,110 / MM-Vet 218 | ASR 0.41 / 1.71; benign ACC 96.79 | Tables 2–3: 0.41 / 1.62 / 96.33 | 공개된 세 데이터셋만 평가. 타깃 VLM 답변은 생성하지 않음. [결과](results/vlmshield/summary.md) |

## 파일과 검증

- [experiments.json](experiments.json): 범위, 방법, 판정, 실행 코드, 근거 파일의 기계 판독 목록.
- [files.json](files.json): 각 원본 집계·프로토콜·실행 스크립트의 SHA-256, 크기, 로컬 상대 경로.
- `results/`: 저장된 집계 CSV/JSON과 실험 조건. 원본 바이트를 보존한다.
- `run_sources/`: `runs/` 또는 참조 저장소에만 있던 실험 실행 코드의 원본 사본.

```bash
# CPU 파일 검증만 수행; 모델 로딩, 네트워크, API 호출 없음.
python reproductions/verify.py

# 보존된 실행 코드의 목록과 안전한 복원. 기존 파일이 다르면 중단한다.
python reproductions/restore_sources.py --list
python reproductions/restore_sources.py --recipe jps_train_20261024
```

검증 명령은 공개 파일의 무결성과 연결을 확인한다. 원시 응답을 다시 판정하거나
논문 성능을 새로 입증하는 명령이 아니다. `runs/`, 데이터, 이미지, 모델 가중치,
원시 타깃 응답과 API 로그는 이 결과 묶음에 포함하지 않는다.
원본 집계 파일의 절대 경로와 일부 로컬 링크는 당시 실행을 기록한 것으로,
새 머신에서 유효한 경로라는 뜻은 아니다.

## 실행 코드 사용

공통 파이프라인은 [루트 README](../README.md), 전용 실행기는
[실행 안내](RUNNERS.md)를 참고한다. `attack_scripts/`와 `defense_scripts/`에는
전용 프로토콜·집계·재시작 스크립트가 추가되어 있다.
JPS, 원본 OneVision, HADES, UltraBreak는 저자 코드와 별도 Transformers 환경을
사용하므로, 보존한 실행 코드를 복원하고 입력 자산·참조 저장소·환경을 준비해야 한다.
원본 실행기를 메인 환경에서 무조건 실행하는 방식은 지원하지 않는다.

결과에 남은 차이는 `assessment`, `scope`, 각 `plan.json`/`summary.json`에서 확인할 수 있다.
모든 방법이 원 논문과 동일하게 재현됐다는 포괄적인 주장을 하지 않는다.
