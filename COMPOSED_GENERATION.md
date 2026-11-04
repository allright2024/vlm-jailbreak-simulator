# Composed jailbreak generation

조합 데이터는 여러 공격의 텍스트·이미지 변환을 연결해 생성합니다.
생성 설정, 원본 질문, 이미지 해시와 적용 단계를 기록하며,
타깃 응답과 ASR은 생성된 입력을 대상으로 별도 평가합니다.

## 다운로드

[Google Drive 공유 폴더](https://drive.google.com/drive/folders/1zxLWKwmC8yzDeXKC3Th7CwqjBchQbh4r?usp=drive_link)에 다음 파일이 있습니다.

| 파일 | 용도 |
| --- | --- |
| `data.zip` | 벤치마크 데이터셋과 방법론별 생성 자산 |
| `3000_samples.zip` | 조합·분해 방식으로 생성한 3,000개 샘플 |

필요한 ZIP을 내려받아 저장소 상위 디렉터리에 놓은 뒤, 저장소 루트에서 압축을 풉니다.

```bash
unzip -tq ../data.zip
unzip -n ../data.zip -d .
unzip -tq ../3000_samples.zip
unzip -n ../3000_samples.zip -d .
```

`data.zip`은 `data/benchmarks/`와 `data/method_assets/`로 복원됩니다.
벤치마크 원본은 전자에, 방법론별 생성 자산은 후자에 둡니다. 생성 샘플은 `runs/`에 복원됩니다.
폴더별 내용은 [DATA_LAYOUT.md](DATA_LAYOUT.md)를 참고하세요.

## 5개 방법 · 3,000개 입력

[설정](configs/compositions/five_methods_3000.yaml)은 FigStep, CS-DJ, MML, JOOD, JPS를 사용합니다.

| 구성 | 레시피 수 | 레시피당 입력 | 합계 |
| --- | ---: | ---: | ---: |
| 두 방법 | 9 | 160 | 1,440 |
| 세 방법 | 7 | 180 | 1,260 |
| 네 방법 | 2 | 150 | 300 |
| 전체 | 18 | | 3,000 |

원본 질문은 SafeBench·HADES·MM-SafetyBench에서 각각 60개씩 선택합니다.
벤치마크별 조합 결과는 정확히 1,000개씩이며 레시피 안에서도 세 출처를 균등하게
배정합니다. JPS 학습 질문과 출처 간 중복 질문을 제외하고, 원본 질문의 모든 파생
샘플에 같은 train/validation/test split을 적용합니다.

JOOD 원본 이미지 가중치는 0.05~0.95의 19단계입니다. JOOD 입력 1,840개에서
각 비율은 96~97회, 무해 이미지 5종은 각각 368회 사용합니다.
비율×이미지 95개 조합도 모두 사용하며 실제 배정은 `sample_plan.json`에 기록합니다.

준비 경로는 데이터 루트의 `benchmarks/{SafeBench,HADES,MM-SafetyBench}/`,
`method_assets/CS-DJ/`, `method_assets/JOOD/images/harmless/`,
`method_assets/nltk_data/`, `method_assets/fonts/`입니다. JPS 이미지·steer prompt·학습 CSV는 YAML의
`jps.source_run`에서 검증된 자산을 읽습니다.
실행 전에 [환경 설치](README.md#2-installation-and-environment)를 완료하세요.

```bash
python -m pip install -r requirements-five-methods.txt
python -m core.prepare_five_method_assets
bash attack_scripts/run_five_method_dataset.sh preview
bash attack_scripts/run_five_method_dataset.sh review-preview --review-note '실제 이미지 검토 내용'
bash attack_scripts/run_five_method_dataset.sh full
bash attack_scripts/run_five_method_dataset.sh validate
```

미리보기는 레시피당 각 벤치마크 1개씩, 총 54개입니다.
JOOD 미리보기는 원본 이미지 가중치 5%·50%·95%를 포함합니다.
contact sheet와 원본 이미지를 검토한 뒤
해당 이미지 해시에 연결된 review record를 저장합니다. `full`은 이 기록을 확인합니다.
기본 출력은 `runs/five_methods_3000_mixed_sources_jood_ratios_20261104/`입니다.
새 실행은 `--output`으로 별도 경로를 지정할 수 있습니다.

```bash
bash attack_scripts/run_five_method_dataset.sh preview --output runs/five_methods_review
bash attack_scripts/run_five_method_dataset.sh review-preview \
  --output runs/five_methods_review --review-note '실제 이미지 검토 내용'
bash attack_scripts/run_five_method_dataset.sh full --output runs/five_methods_review
bash attack_scripts/run_five_method_dataset.sh validate --output runs/five_methods_review
```

백그라운드 실행·재개, JOOD MixUp과 다중 이미지 JPS 조합의 동작은
[FIVE_METHOD_DATASET.md](FIVE_METHOD_DATASET.md)를 참고하세요.

## 일반 단계 조합

[설정](configs/compositions/jailbreak_dataset_v3.yaml)은
**3개 레시피, 후보 720개, 최종 600개**입니다.

| ID | 적용 순서 | 최종 수량 |
| --- | --- | ---: |
| P01 | CS-DJ-Split → FigStep | 200 |
| P02 | CS-DJ-Split → MML | 200 |
| P03 | FigStep → JOOD | 200 |

`component`는 적용할 변환, `attack`은 원본 방법, `config`는 해당 방법의 YAML,
`overrides`는 이 조합에 적용할 값을 지정합니다.
`execution_groups`는 모든 레시피를 한 번씩 포함하며 실행 순서를 정합니다.

```bash
python run_generate_composed_dataset.py --dry-run
python run_generate_composed_dataset.py --preflight
python run_generate_composed_dataset.py --recipes P01,P02,P03 --per-recipe 3 \
  --save-dir runs/composed_pilot
python -m core.validation.composed_dataset runs/composed_pilot

CONFIG=configs/compositions/jailbreak_dataset_v3.yaml \
OUTPUT_ROOT=runs/composed_generation \
  bash attack_scripts/run_composed_dataset_3000.sh
```

직접 CLI의 `--per-recipe 3`은 선택한 레시피마다 최종 입력 3개를 생성합니다.
전체 실행기는 수량과 그룹을 YAML에서 읽습니다. `--recipes`로 범위를 선택하고,
`--resume`으로 설정·원본·아티팩트가 일치하는 체크포인트를 재개합니다.
완료된 후보는 10회마다 그리고 레시피 경계마다 저장합니다.

## 산출물과 평가

| 구성 | 주요 파일 |
| --- | --- |
| 5개 방법 조합 | `preview/test_cases.jsonl`, `full/test_cases.jsonl`, `parents.json`, `sample_plan.json`, 각 phase의 `behaviors.jsonl`·`manifest.json`·`qc.json`·`status.json` |
| 일반 단계 조합 | `test_cases.jsonl`, `behaviors.jsonl`, `images/`, `composition_manifest.json`, `composition_qc.json` |

조합 입력에는 `adapted_composition`과 단계별 provenance를 기록합니다.
구조 검사, 이미지 존재·해시, 단계 연결, 중복과 원본 질문별 split을 검증합니다.
일반 조합은 생성 폴더를, 5개 방법 조합은 `full/`을 Stage 2의 `--run_dir`로
지정합니다. 각 phase의 `behaviors.jsonl`에 원본 평가 지시문을 함께 저장하므로
Stage 3에서 같은 폴더를 `--run_dir`로 사용합니다. [README의 Stage 2–3 안내](README.md#6-single-attack-pipeline)에 따라
같은 원본 지시문을 기준으로 응답을 판정합니다.
