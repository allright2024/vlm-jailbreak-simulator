# 재현 실행 코드와 환경

이 문서는 보존한 결과를 만든 코드의 위치를 안내한다. 데이터와 모델은 별도 자산이며,
집계 결과 파일만으로 GPU 실험 전체를 다시 실행할 수는 없다.
대부분의 준비 단계는 데이터를 읽거나 다운로드하고, 추론·평가 단계는 GPU 또는 API를 사용한다.

UltraBreak, JIP, ECSO의 실행 코드는 삭제했으며 과거 결과만 보존한다.

## 공통 실행기

저장소 루트에서 실행한다. `--help`로 해당 실행기의 단계와 출력 위치를 확인한다.

| 방법 | 실행기 | 보존한 범위 |
|---|---|---|
| FigStep | `attack_scripts/figstep_reproduction.py`, `figstep_full500.py` | 500개 입력/5회 응답과 후속 500개 rubric 재평가 |
| MML-M | `attack_scripts/prepare_mml_reproduction.py`, `prepare_mml_full_reproduction.py`, `mml_full_reproduction.py`, `validate_mml_full_reproduction.py` | 공식 Drive 입력, 파일 검증, CLAS 성공 시 중단, 500개 집계 |
| MML-WR/R/B64/SC | `attack_scripts/prepare_mml_variants_reproduction.py`, `mml_variants_reproduction.py`, `mml_variants_background.py` | 각 500개, GPT-4o 응답·GPT-4o-mini CLAS; SC 로컬 렌더링 |
| JOOD | `attack_scripts/prepare_jood_reproduction.py`, `jood_reproduction.py` | `generate`, `guard`, `summarize` 단계; 17개 FW 문항 |
| JOOD MiniGPT-4-7B | `attack_scripts/prepare_jood_minigpt4_assets.py`, `jood_minigpt4_target.py`, `jood_minigpt4_evaluate.py`, `jood_minigpt4_background.py`, `jood_minigpt4_dual_gpu.py` | 7범주 170문항; HF 후보 선택과 ASR 판정을 별도로 수행 |
| CS-DJ | `attack_scripts/cs_dj_reproduction.py`, `cs_dj_onevision_reproduction.py` | 초기 GPT-4o 및 HF OneVision 실험용. 최종 원본 OneVision 결과는 아래 별도 실행 코드 사용 |
| HADES | `attack_scripts/hades_reproduction_preparation.py` | 공식 입력 준비; 최종 원본 LLaVA 및 issue 14 검증은 아래 코드 사용 |
| VisualAdv | `attack_scripts/visualadv_reproduction.py` | [전용 프로토콜](../image_attacks/visual_adv/REPRODUCTION.md), `configs/experiments/VisualAdv_paper.yaml` |
| JailBound | `attack_scripts/jailbound_three_category_reproduction.py` 및 `jailbound_*` 검증 코드 | 자체 probe의 세 범주 실험; 논문 ASR과 동일한 의미로 취급하지 않음 |
| AdaShield | `defense_scripts/adashield_official_repro.py` | FigStep/QR, vanilla/static/adaptive |
| AsD | `defense_scripts/asd_official_repro.py` | MM-SafetyBench SD_TYPO |
| AsD 공개 패치 검증 | `defense_scripts/asd_released_patch_check.py` | 공식 직렬 실행과 배치 어댑터의 64개 입력 비교; `asd_official_repro.py --patch_path` 지원 |
| CIDER | `defense_scripts/cider_official_repro.py` | threshold/detection/응답 ASR |
| JailGuard | `defense_scripts/jailguard_official_repro.py`, `jailguard_official_repro_eval.py` | PL 검출기, MiniGPT-4, 공식 1,000개 |
| VLMShield | `defense_scripts/setup_vlmshield.sh`, `vlmshield_official_repro.py` | 공개 classifier와 세 공개 평가 데이터셋 |
| EchoSafe | `defense_scripts/prepare_echosafe_reproduction.py`, `echosafe_reproduction.py` | 외부 EchoSafe checkout을 사용한 Table 3 Gen 6범주 평가 |
| 5개 방법 조합 | `run_generate_five_method_dataset.py`, `attack_scripts/run_five_method_dataset_background.sh`, `five_method_gallery.py` | [구성 안내](../FIVE_METHOD_DATASET.md); 최신 소스는 실제 이미지 MixUp 사용, 타깃 응답 평가는 별도 |

`defense_scripts/train_balanced_refusal_boundary.py`도 실행 코드로 포함한다.
이번 게시에 추가된 MM-SafetyBench 감사 기록은 baseline 추론 진단이며,
이 학습기의 완료된 실험 결과를 뜻하지 않는다.

루트 `requirements.txt`는 공통 환경이다. 원본 LLaVA 및 JPS 전용 실행기는
별도 환경을 사용했다. 기록된 패키지를 공통 환경 위에 덮어 설치하지 않는다.
`--help` 전에 import 또는 경로 검사를 수행하는 역사적 스크립트도 있다.
API 기반 평가의 모델 이름은 당시 실행 기록이며 새 실행을 자동으로 예약하지 않는다.

## 보존한 원본 실행 코드

`run_sources/`의 코드는 SHA-256으로 관리한다. JPS 평가 실행기의 Qwen 어댑터 import는
공통 모듈 이동에 맞춰 수정했고, 이전 해시와 변경 사유를 `files.json`에 기록했다.
그 외 보존 파일은 당시 바이트를 유지한다.
실험 디렉터리를 기준으로 경로를 계산하므로, 그 위치에서 바로 실행하지 말고
`restore_sources.py`로 `runs/<실험명>/` 구조에 복원한다.
복원은 실행하지 않으며, 기존 파일과 내용이 다르면 전체 복원을 중단한다.

```bash
python reproductions/restore_sources.py --list
python reproductions/restore_sources.py --recipe jps_train_20261024
# 별도 작업용 checkout으로 복원할 수도 있다.
python reproductions/restore_sources.py --recipe jps_train_20261024 \
  --destination /path/to/vlm-jailbreak-simulator
```

| 실험 디렉터리 | 주요 파일 | 당시 환경 / 의존성 |
|---|---|---|
| `jps_paper_20261024` | `paper_run.py` | `../reference_repos/JPS` at `beacbc58`; Transformers 4.48.2, Torch 2.9.1, SDPA; HarmBench standard 200 재구성 |
| `jps_train_20261024` | `train_run.py` | 위 실행기 자동 복원; TAP AdvBench subset 50, Qwen2-VL-7B, Qwen2.5-14B; original `clean.jpeg` 동일성 미확인 |
| `cs_dj_reference_onevision_20261022` | `native_onevision.py`, `evaluate_cs_dj.py`, `orchestrate.py` | CS-DJ와 LLaVA-NeXT 참조 코드, Transformers 4.45.2; SDPA MATH; Beaver-Dam |
| `hades_paper_llava15_20261022` | `paper_run.py` | HADES 원본 LLaVA와 공식 이미지; `hades_reproduction_preparation_20261022`의 준비 결과 필요 |

HADES issue 14 검증 코드는
`run_sources/reference_repos/HADES/white_box/issue14_consistent_probe.py`에 있다.
이 파일은 프로젝트에서 작성한 검증 실행기이며 저자 공개 코드 자체라는 뜻은 아니다.
HADES checkout의 해당 위치와 의존 입력을 준비한 후 사용한다.

### JailGuard reference 실행 스냅샷

`run_sources/reference_repos/JailGuard/reproduction_runs/20261103_reference_pl/`에는
프로젝트에서 작성한 `worker.py`, `supervise.py`, 로컬 설정과 두 runtime diff를 보존했다.
JailGuard 전체 원본 소스와 모델은 포함하지 않는다. 이 디렉터리는
`restore_sources.py`의 `runs/` 복원 목록에 포함되지 않으며,
JailGuard checkout의 `reproduction_runs/20261103_reference_pl/`로 별도 복사한다.

공식 JailGuard revision `419e1d3d08f4d02998a11a687f925518d2d5dd0c`를 준비한 뒤,
복사한 실행 디렉터리에서 원본을 건드리지 않고 runtime 사본을 만들 수 있다.

```bash
cp ../../JailGuard/main_img.py main_img_runtime.py
patch main_img_runtime.py < main_img_runtime.diff
cp ../../JailGuard/utils/minigpt_utils.py minigpt_utils_runtime.py
patch minigpt_utils_runtime.py < minigpt_utils_runtime.diff
```

원본 및 runtime 사본의 예상 해시, MiniGPT 의존 revision, 환경 차이는
[manifest](results/jailguard_reference_20261103/manifest.json)에 있다.
하드코딩된 GPU 선택과 사용하지 않는 Gradio import만 해당 diff로 변경한다.
`worker.py`는 별도 실행 환경과 모델·데이터·manifest 경로가 필요하다.
공개된 상태 파일은 진행 중 실행의 스냅샷이며 최종 성능이 아니다.

당시 Python 경로, Hugging Face cache snapshot 경로, 모델 저장 위치가 일부 코드와
shell wrapper에 고정되어 있다. **복원 명령은 그 경로·환경·데이터를 설치하거나 변경하지 않는다.**
다른 머신에서는 경로와 모델 revision을 먼저 확인하고, 사본에서 수정한 내용과
원본 파일 해시를 새 run manifest에 기록한다. 일부 원본 입력은 공개되지 않았으므로
같은 명령의 재실행이 논문과의 완전한 동등성을 보장하지 않는다.

## 소스·라이선스와 결과 범위

공식 저장소와 revision은 각 결과의 `plan.json`/`manifest.json`/`summary.json` 또는
[OFFICIAL_ALIGNMENT.md](../OFFICIAL_ALIGNMENT.md)에 기록되어 있다.
참조 저장소 전체와 모델 가중치는 포함하지 않는다.
[THIRD_PARTY.md](../THIRD_PARTY.md)의 원본 라이선스와 자산 준비 안내를 따른다.

새 실행은 과거 결과 디렉터리에 섞지 않는다. 평가 분모, 판정기, 입력 추출 방식,
학습/평가 분할을 바꾸면 별도 실험으로 기록한다. 특히 HADES 50개 검증,
JPS의 원본 질문 판정과 JailBound 비거절률은
각각 해당 논문의 전체 평가와 구분해야 한다.
