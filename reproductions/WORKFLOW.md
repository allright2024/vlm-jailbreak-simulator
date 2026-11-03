# 방법론별 논문 재현

`reproduction` 브랜치는 각 jailbreak·jailbreak defense 방법론의 논문 재현을
실행하고 기록하는 용도다. `master`는 공통 simulator 코드 관리에 사용한다.
공통 구현·모델 로더·데이터 로더는 재현 실행기의 의존성으로 이 브랜치에도 유지한다.

## 시작하기

```bash
git fetch origin reproduction
git switch reproduction
python reproductions/restore_sources.py --list
```

위 `origin`은 이 저장소를 직접 clone했을 때의 remote 이름이다. 현재 연구 서버의
전용 checkout은 `/home/jovyan/jailbreak/vlm-jailbreak-reproduction`이고, 이 서버의
게시 remote 이름은 `allright`다. checkout을 만들거나 소스를 복원하는 명령은
GPU 추론·학습·API 평가를 시작하지 않는다.

1. 아래 표에서 방법론과 실험 범위를 고른다. 상세 설정은
   [설정 목록](../configs/experiments/reproduction_settings/index.json)과
   [실험 catalog](experiments.json)의 `settings`·`scope`를 확인한다.
2. 해당 논문·공식 revision의 데이터, 모델, 환경을 준비한다.
   [실행기 안내](RUNNERS.md)에 원본 환경과 외부 checkout 의존성을 기록했다.
   실제 실행 manifest가 [당시 실행기 기본값](../configs/experiments/reproduction_settings/runner_defaults.json)보다 우선한다.
3. 입력 생성, 타깃 답변 생성, 판정을 해당 방법론의 실행기로 수행한다.
   원본 실행 디렉터리가 필요한 경우 먼저 `restore_sources.py`로 복원한다.
4. 새 출력은 `runs/<method>_<date>_<variant>/` 또는 공식 reference repo의
   별도 실행 디렉터리에 저장한다. 과거 결과와 섞지 않는다.
5. 입력 수·분모·모델·judge·seed·버전·논문 대비 변경점을 manifest에 기록하고,
   완료 후 집계값과 논문 비교표를 `reproductions/results/<experiment_id>/`에 보존한다.
   [실험 catalog](experiments.json)와 SHA-256 [파일 목록](files.json)도 함께 갱신한다.

## 방법론별 진입점

| 방법론 | 재현 범위 | 진입점 / 실험 ID |
|---|---|---|
| FigStep | SafeBench 500, LLaVA-1.5-7B, 5회; 자동 판정과 원 논문의 인간 평가 차이 기록 | [figstep_full500.py](../attack_scripts/figstep_full500.py), `figstep`, `figstep_harmbench` |
| CS-DJ | 750개, 원본 LLaVA-NeXT OneVision, Beaver-Dam | `cs_dj_reference_onevision_20261022` 복원 후 [orchestrate.py](run_sources/runs/cs_dj_reference_onevision_20261022/orchestrate.py), `cs_dj` |
| JOOD | AdvBench-M 170개·7범주, MiniGPT-4-7B, GPT-4.1 HF 선택 및 Llama Guard 2 ASR | [jood_minigpt4_dual_gpu.py](../attack_scripts/jood_minigpt4_dual_gpu.py), `jood_minigpt4` |
| JPS | 공개 산출물 평가와 50개 학습·200개 평가를 별도 기록 | `jps_paper_20261024` / `jps_train_20261024` 복원, `jps_released`, `jps_trained` |
| MML | SafeBench 500, MML-M 및 WR/R/B64/SC 각각 평가 | [mml_full_reproduction.py](../attack_scripts/mml_full_reproduction.py), [mml_variants_reproduction.py](../attack_scripts/mml_variants_reproduction.py) |
| HADES | 공개 이미지 750개 평가, issue 14 방식 50개 검증을 별도 기록 | `hades_paper_llava15_20261022` 복원, [issue14_consistent_probe.py](run_sources/reference_repos/HADES/white_box/issue14_consistent_probe.py) |
| VisualAdv | 원본 MiniGPT-4, RTP 1,199개 × 3회, Detoxify; 공개 이미지/직접 최적화 | [visualadv_reproduction.py](../attack_scripts/visualadv_reproduction.py), [프로토콜](../image_attacks/visual_adv/REPRODUCTION.md) |
| JailBound | 자체 probe 기반 세 범주 검증; 비거절률을 논문 ASR과 구분 | [jailbound_three_category_reproduction.py](../attack_scripts/jailbound_three_category_reproduction.py), `jailbound` |
| AdaShield | FigStep/QR의 vanilla·static·adaptive, LLaVA-1.5-13B | [adashield_official_repro.py](../defense_scripts/adashield_official_repro.py) |
| AsD | MM-SafetyBench SD_TYPO; 공개 패치 실행 경로 검증은 별도 실험 | [asd_official_repro.py](../defense_scripts/asd_official_repro.py), [asd_released_patch_check.py](../defense_scripts/asd_released_patch_check.py) |
| CIDER | HarmBench 기반 검출 및 방어 후 답변 ASR, LLaVA-1.5-7B | [cider_official_repro.py](../defense_scripts/cider_official_repro.py) |
| JailGuard | Table 4 PL, 공식 1,000개, 원본 MiniGPT-4-13B int8·직렬 생성 | [reference 실행 절차](RUNNERS.md#jailguard-reference-실행-스냅샷), `jailguard_reference_20261103` |
| VLMShield | 공개 classifier와 AdvBench-M/VLSafe/MM-Vet 검출 평가 | [vlmshield_official_repro.py](../defense_scripts/vlmshield_official_repro.py) |
| EchoSafe | Table 3 Gen 공개 6범주, Qwen2.5-VL-7B | [echosafe_reproduction.py](../defense_scripts/echosafe_reproduction.py) |

표는 실행 경로와 보존 범위를 나타낸다. 실행 완료나 논문 수치 일치 여부는
[결과 기록](README.md)의 평가에 따른다. 공통 코드에 있는 DeltaJP·HiddenDetect는
현재 catalog에 별도 재현 실험 기록이 없으며, 완료된 재현 방법으로 표시하지 않는다.

## 원본 실행 디렉터리 복원

```bash
python reproductions/restore_sources.py --recipe jps_train_20261024
python reproductions/restore_sources.py --recipe cs_dj_reference_onevision_20261022
python reproductions/restore_sources.py --recipe hades_paper_llava15_20261022
```

JPS 학습 실행기는 의존하는 평가 실행기도 함께 복원한다. 복원기는 소스 해시와
기존 파일 충돌을 확인하고 코드를 복사한다. 데이터·모델·환경은 별도 준비하며,
일부 역사적 실행기는 경로를 고정했으므로 실제 머신 경로와 환경을 먼저 확인한다.
예시 명령을 실행한다고 모든 방법의 실험이 시작되지는 않는다.

JailGuard reference 실험은 공식 repo의 `reproduction_runs/`에서 실행한다.
현재 실행 중인 프로세스는 원래 작업 경로를 유지하며 이 브랜치 생성 때문에
재시작하거나 중복 실행하지 않는다. 저장소의 상태 JSON은 실시간 상태가 아닌 스냅샷이다.

## 기록 범위

`experiments.json`의 `usage`는 `method_reproduction`, `diagnostic`,
`composition_archive`, `historical_removed_method`를 구분한다.
조합 데이터 생성과 진단은 논문 재현 성공으로 집계하지 않는다.
UltraBreak·JIP·ECSO는 이전 삭제 요청을 유지하여 구현을 복원하지 않았고,
과거 결과만 역사적 기록으로 보존한다. 테스트 소스도 제외한다.

새 실행의 원본 이미지·가중치·대량 응답·API 비밀 값은 Git에 포함하지 않는다.
공식 repository와 revision, 외부 자산의 조건은 [THIRD_PARTY.md](../THIRD_PARTY.md)를 따른다.
