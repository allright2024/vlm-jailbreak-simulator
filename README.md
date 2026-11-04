
# VLM-Jailbreak-Simulator

This official open-source repository serves as a collaborative space for the participating institutions of the research project entitled *"Development of Resilient AI Technology Against Risk Factors,"* dedicated to disclosing and sharing key research outcomes and core software (SW).

**VLM-Jailbreak-Simulator** is a simulation framework for evaluation of multi-modal jailbreak attacks on vision-language models (VLMs). The framework allows users to configure attack methods, target models, and datasets to conduct controlled simulations and comparative analyses.

Jailbreak attacks are categorized into two types:

- **Image-based attacks** — manipulations relying solely on visual perturbations.  
- **Multi-modal attacks** — methods that jointly manipulate both visual and textual modalities through combined image and prompt-based strategies.

Each category includes recent state-of-the-art jailbreak implementations, enabling reproducible and extensible evaluations under a unified experimental setting.

> **License notice (please read before use).** The MIT License in
> [LICENSE](LICENSE) covers **only** the source code and documentation
> directly developed by ai.resilience as part of this project. When the
> simulator runs, it also vendors copies of third-party attack code,
> downloads third-party model weights, reads third-party datasets,
> renders text with a third-party font, and calls closed-source LLM
> APIs. Those components are **NOT** covered by the MIT License — they
> remain governed by their **original upstream terms**, which include
> the **Llama 2 Community License** (HarmBench classifier, LLaVA v1.5),
> the **LLaMA Research License / Vicuna non-commercial terms** (used
> via MiniGPT-4), the **Tongyi Qianwen License** (vendored Qwen-VL
> code), **Apache-2.0 / BSD-3-Clause / MIT** (most other vendored
> attack and model code), the **OpenAI Terms of Use** (Hades / JOOD
> API calls), the proprietary copyright of the **Arial** typeface
> (Hades font asset), and the upstream copyrights of a few attack
> repositories that carry **no explicit license at all** (notably
> Visual Adversarial Examples and Jailbreaking-Attack-against-MLLM).
> Users and redistributors are responsible for reviewing and complying
> with each upstream license. See [THIRD_PARTY.md](THIRD_PARTY.md) for
> a runtime-oriented inventory.


## Code and workflow guide

**실험 설정:** 기존 재현 실험의 설정은
[설정 안내](configs/experiments/reproduction_settings/README.md)에 정리했습니다.
최신 [5개 방법·3,000개 조합 데이터 구성](FIVE_METHOD_DATASET.md)은 JOOD 이미지 MixUp을
사용하며, 생성 완료와 타깃 응답·ASR 평가 완료를 구분합니다.

이 문서는 코드 구조, 데이터 로딩, 단일 공격, 방어,
공격 조합 생성 및 평가 실행 방법을 정리합니다. 데이터·모델 가중치는 별도로 준비합니다.
실험 설정과 실행 정보는 생성된 manifest에서 확인할 수 있습니다.

- [1. Repository layout](#1-repository-layout)
- [2. Installation and environment](#2-installation-and-environment)
- [3. Dataset loading](#3-dataset-loading)
- [4. Jailbreak attacks](#4-jailbreak-attacks)
- [5. Target and surrogate models](#5-target-and-surrogate-models)
- [6. Single-attack pipeline](#6-single-attack-pipeline)
- [7. Jailbreak defenses](#7-jailbreak-defenses)
- [8. Composed jailbreak generation](#8-composed-jailbreak-generation)
- [9. Evaluation and metrics](#9-evaluation-and-metrics)
- [10. Validation](#10-validation)
- [11. Related documentation](#11-related-documentation)

모든 명령어는 저장소 루트에서 실행합니다.

## 1. Repository layout

주요 코드와 실행 진입점은 다음과 같습니다.

```text
vlm-jailbreak-simulator/
├── data/
│   ├── benchmarks/             # 벤치마크 원본
│   └── method_assets/          # 방법론별 생성·최적화 자산
├── runs/                       # 생성 샘플·응답·평가 결과
├── core/
│   ├── schema.py                    # Behavior / TestCase / Completion
│   ├── dataset.py, datasets/        # 데이터셋 공통 인터페이스와 로더
│   ├── paths.py, data.py            # 외부 데이터 경로와 사전 검증
│   ├── dataset_policy.py            # harmful-only 정책
│   ├── attack.py, attacks/          # 공격 인터페이스와 등록 허브
│   ├── pipeline.py                 # 생성 → 응답 → 평가 공통 처리
│   ├── composition*.py             # 조합 실행, 단계 연결, 체크포인트
│   ├── targets/, judges/           # 평가 대상 모델과 응답 판정기
│   ├── metrics.py, multimodal_metrics.py
│   └── validation/                 # 데이터·재현 조건·조합·캐시 검증
├── multi_attacks/                   # 방법론별 멀티모달 공격
├── image_attacks/                   # VisualAdv, DeltaJP, 공통 CE surrogate
├── defenses/                       # 방어 구현과 보조 모듈
├── multimodalmodels/               # 모델 로딩, revision pin, surrogate 프로필
├── configs/
│   ├── attacks/                    # 개별 공격 YAML
│   ├── defense_configs/            # 방어 YAML
│   └── compositions/               # 조합 계획 YAML
├── attack_scripts/                 # 개별 공격·전이·조합 실행 래퍼
├── defense_scripts/                # 방어 자산 준비 및 실행 래퍼
├── run_generate_test_cases.py       # Stage 1
├── run_generate_completions.py      # Stage 2: 방어 선택 가능
├── run_evaluate_completions.py      # Stage 3
├── run_generate_composed_dataset.py # 조합 데이터셋 생성
└── run_extended_metrics.py          # 평가 JSONL의 지표 집계
```

실행은 `run_*.py`의 통합 JSONL 파이프라인을 기준으로 합니다.
쉘 래퍼는 `attack_scripts/`, `defense_scripts/`에서 제공합니다.
개발 인터페이스는 [core/README.md](core/README.md)를 참고하세요.

## 2. Installation and environment

설치 기준은 **Linux x86_64, Python 3.11.11, NVIDIA GPU, CUDA 13.0 빌드**입니다.
[requirements.txt](requirements.txt)는 실행 패키지와 전이 의존성을 동일한 기준 환경의
설치 버전으로 고정합니다. 모델 가중치와 데이터는 설치 후 별도로 준비합니다.

| 구성 요소 | 기준 버전 |
| --- | --- |
| Python | 3.11.11 |
| PyTorch / TorchVision / TorchAudio | 2.9.1 / 0.24.1 / 2.9.1, 모두 `+cu130` |
| Transformers / sentence-transformers | 4.57.1 / 5.1.2 |
| NumPy / SciPy / scikit-learn | 1.26.4 / 1.17.1 / 1.9.1 |
| spaCy / NLTK | 3.7.2 / 3.8.1 |
| qwen-vl-utils | 0.0.14 |

CUDA 13.x는 [NVIDIA 호환성 문서](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)에
따라 580 계열 이상의 호환 드라이버가 필요합니다. 먼저 `nvidia-smi`로 드라이버와
GPU를 확인하고, 사용할 모델에 맞는 GPU 여유 메모리를 확보하세요.

### Create and install

Conda 환경을 만들고 필요한 패키지를 설치합니다. 기존 환경을 사용 중이면 이름을 바꾸세요.

```bash
conda create -n jb_env python=3.11.11 -y
conda activate jb_env
conda env config vars set PYTHONNOUSERSITE=1
conda deactivate
conda activate jb_env

# CLIP 소스 빌드에 사용할 도구를 먼저 설치
python -m pip install "pip==24.3.1" "setuptools==75.8.0" "wheel==0.45.1"

# PyTorch의 공식 CUDA 13.0 인덱스에서 GPU 빌드를 먼저 설치
python -m pip install \
  "torch==2.9.1+cu130" "torchvision==0.24.1+cu130" "torchaudio==2.9.1+cu130" \
  --index-url https://download.pytorch.org/whl/cu130

python -m pip install --no-build-isolation -r requirements.txt
```

CLIP 소스 설치에 `git`이 필요합니다.

### Verify the environment

설치 버전과 GPU 사용 가능 여부를 확인합니다.

```bash
python -m pip check
python - <<'PY'
import importlib.metadata as metadata
import platform
import site
import sys
import torch

assert platform.python_version() == "3.11.11"
assert site.ENABLE_USER_SITE is False
expected = {
    "torch": "2.9.1+cu130", "torchvision": "0.24.1+cu130",
    "torchaudio": "2.9.1+cu130", "transformers": "4.57.1",
    "sentence-transformers": "5.1.2", "numpy": "1.26.4",
    "spacy": "3.7.2", "nltk": "3.8.1", "qwen-vl-utils": "0.0.14",
    "en-core-web-md": "3.7.1",
}
for package, version in expected.items():
    assert metadata.version(package) == version, package
assert torch.version.cuda == "13.0"
assert torch.cuda.is_available(), "NVIDIA GPU와 드라이버 설정을 확인하세요."
print("Python:", sys.executable)
print("Environment OK")
PY
```

VS Code·노트북에서도 `jb_env`의 Python 인터프리터·커널을 선택하세요.

### Data, defense assets and optional metrics

NLTK 태거는 `data.zip`에 포함됩니다. 누락된 경우 아래 명령으로 설치합니다.

```bash
export NLTK_DATA="${VLM_JAILBREAK_DATA_ROOT:-$PWD/data}/method_assets/nltk_data"
python -m nltk.downloader -d "$NLTK_DATA" averaged_perceptron_tagger
```

SSIM·LPIPS 지표를 사용하려면 추가 패키지를 설치합니다.

```bash
python -m pip install -c requirements.txt "scikit-image==0.25.2" "lpips==0.1.4"
python -m pip check
```

LPIPS·CLIP 유사도에는 모델 가중치도 필요합니다.

| 환경 변수 | 용도 |
| --- | --- |
| `VLM_JAILBREAK_DATA_ROOT` | 별도 준비한 데이터 루트. 미지정 시 `<repo>/data` |
| `OPENAI_API_KEY` | OpenAI 대상 모델, CLAS judge, API 기반 공격 보조 처리 |
| `HF_TOKEN` | 접근 승인이 필요한 Hugging Face 모델 |
| `PYTHON` | 쉘 래퍼가 사용할 Python 실행 파일 |
| `CUDA_VISIBLE_DEVICES` | 사용할 GPU 선택 |
| `PYTHONNOUSERSITE` | `1`로 설정하여 사용자 site-packages 혼입 방지 |
| `NLTK_DATA` | MML 등에서 사용하는 NLTK 자산 경로 |

환경 변수는 실행 전에 설정하세요. API 키 형식은 [.env.example](.env.example)을 참고하세요.

## 3. Dataset loading

### Benchmark inputs and method assets

`data/`는 벤치마크 원본과 방법론 자산을 분리해 관리합니다.

| 경로 | 용도 |
| --- | --- |
| `data/benchmarks/` | 평가·샘플 생성의 기준이 되는 벤치마크 원본 |
| `data/method_assets/` | 공격 샘플 생성·최적화·재현에 필요한 이미지·문장·폰트·공용 도구 |
| `runs/<실험명>/` | 생성한 샘플·패치, 모델 응답, 평가 결과 |

현재 폴더 구성입니다.

```text
data/
├── benchmarks/
│   ├── advbench/                # AdvBench 원본
│   ├── AdvBenchM/               # AdvBench-M 원본 공개본
│   ├── HADES/                   # HADES 원본
│   ├── MM-SafetyBench/          # MM-SafetyBench 원본
│   └── SafeBench/               # SafeBench 원본
└── method_assets/
    ├── CS-DJ/                   # 이미지 풀·지시문·폰트
    ├── JOOD/                    # JOOD 부분집합·혼합 이미지
    ├── JailbreakInPieces/       # 고정 입력 이미지
    ├── MML/                     # 공식 재현 이미지·메타데이터
    ├── VisualAdv/               # 최적화 목표 문장
    ├── fonts/                   # 조합 생성용 폰트
    ├── nltk_data/               # MML 단어 치환용 품사 태거
    └── provenance/              # 자산 출처·이동 기록
```

AdvBench-M 원본은 `data/benchmarks/AdvBenchM/`, 이전 `jood_subset`은 `data/method_assets/JOOD/`에 있습니다.
`--dataset AdvBenchM`은 JOOD의 170문항을 읽습니다.

### Download and extract

[Google Drive 공유 폴더](https://drive.google.com/drive/folders/1zxLWKwmC8yzDeXKC3Th7CwqjBchQbh4r?usp=drive_link)에서 다운로드합니다.

| 파일 | 내용 | 압축 해제 위치 |
| --- | --- | --- |
| `data.zip` | `benchmarks/`와 `method_assets/` | 저장소 루트의 `data/` |
| `3000_samples.zip` | 조합·분해 방식으로 생성한 3,000개 샘플 | 저장소 루트의 `runs/` |

`data.zip`을 저장소 상위 폴더에 두고 압축을 풉니다.

```bash
unzip -tq ../data.zip
unzip -n ../data.zip -d .
export VLM_JAILBREAK_DATA_ROOT="$PWD/data"
export NLTK_DATA="$VLM_JAILBREAK_DATA_ROOT/method_assets/nltk_data"
```

기존 평면 구조의 데이터를 사용한다면 아래 명령으로 이전합니다.

```bash
python -m core.migrate_data_layout --data-root "$VLM_JAILBREAK_DATA_ROOT" --dry-run
python -m core.migrate_data_layout --data-root "$VLM_JAILBREAK_DATA_ROOT"
```

공식 MML 재현 자산은 [MML 준비 안내](multi_attacks/mml/README.md)에 따라 별도로 준비합니다.
조합 생성용 폰트는 `python -m core.prepare_five_method_assets`로 준비합니다.

### Validate the data

`--dataset`으로 검사할 데이터셋을 지정합니다. `--layout-only`는 파일 구조만 검사합니다.

```bash
python -m core.validation.data --dataset SafeBench --dataset AdvBench --layout-only
python -m core.validation.data --dataset SafeBench --dataset AdvBench
```

### Dataset CLI paths

현재 준비된 데이터의 CLI 이름과 실제 입력 경로입니다. 경로는 `data/` 기준입니다.

| CLI 이름 | 입력 구분 | 경로 | 원본 행 수 |
| --- | --- | --- | ---: |
| [AdvBench](core/datasets/advbench.py) | 벤치마크 | `benchmarks/advbench/harmful_behaviors.csv` | 520 |
| [HADES](core/datasets/hades.py) | 벤치마크 | `benchmarks/HADES/hades.csv`, `benchmarks/HADES/images/` | 750 |
| [MM-SafetyBench](core/datasets/mm_safetybench.py) | 벤치마크 | `benchmarks/MM-SafetyBench/` | 1,680 |
| [SafeBench](core/datasets/safebench.py) | 벤치마크 | `benchmarks/SafeBench/safebench.csv` | 500 |
| [AdvBenchM](core/datasets/advbenchm.py) | JOOD 자산 | `method_assets/JOOD/` | 170 |
| [CS-DJ](core/datasets/cs_dj.py) | CS-DJ 자산 | `method_assets/CS-DJ/instructions/*.json` | 750 |

원본 AdvBench-M 공개본과 방법론별 파일 구성은 [DATA_LAYOUT.md](DATA_LAYOUT.md)를 참고하세요.

### Schema and harmful-only policy

데이터와 실험은 다음 레코드로 연결됩니다.

| 레코드 | 역할 |
| --- | --- |
| `Behavior` | 원본 지시문, 고유 ID, 데이터셋, 위험 유형, 이미지 및 보조 필드 |
| `TestCase` | 공격으로 만든 실제 입력 텍스트·이미지, `behavior_id`, 시도 번호 |
| `Completion` | 대상 모델 응답, 차단 여부, judge 라벨 및 계측 메타데이터 |

평가는 원본 지시문을 기준으로 합니다. 기본 생성 대상은 학습 행을 제외한 `harmful` 유형이며, 전체 유형은 `--eval-types all`로 선택합니다.

## 4. Jailbreak attacks

`--attack`으로 실행할 공격을 선택합니다.

| CLI 공격 이름 | 구현 위치 | 생성 방식 및 주요 의존성 | 실행 스크립트 |
| --- | --- | --- | --- |
| `FigStep` | [multi_attacks/figstep/](multi_attacks/figstep/) | 지시문을 이미지 목록으로 렌더링하고 고정 프롬프트 결합. 폰트 차이 기록 | [run_figstep.sh](attack_scripts/run_figstep.sh) |
| `CS-DJ` | [multi_attacks/cs_dj/](multi_attacks/cs_dj/) | Qwen 분해 → CLIP 대조 이미지 검색 → 3×4 콜라주. 외부 이미지 풀·폰트 필요 | [run_cs_dj.sh](attack_scripts/run_cs_dj.sh) |
| `MML` | [multi_attacks/mml/](multi_attacks/mml/) | 공개 이미지·프롬프트 replay. 공통 Stage 2의 MML 프로토콜로 CLAS 조기 종료 | [run_mml.sh](attack_scripts/run_mml.sh) |
| `HADES` | [multi_attacks/hades/](multi_attacks/hades/) | 기본 white-box 상단 패치 3,000 epochs. 별도 official 설정은 공개 이미지 6단계 replay | [run_hades.sh](attack_scripts/run_hades.sh) |
| `JOOD` | [multi_attacks/jood/](multi_attacks/jood/) | benign 항목과 Mixup/CutMix/ResizeMix/RandAug. 개별 runner는 AdvBenchM 대상 | [run_jood.sh](attack_scripts/run_jood.sh) |
| `JPS` | [multi_attacks/jps/](multi_attacks/jps/) | Qwen2 시각 최적화와 Qwen2.5-14B 에이전트의 피드백·수정. 공식 CSV·기저 이미지 별도 필요 | [run_jps.sh](attack_scripts/run_jps.sh) |
| `VisualAdv` | [image_attacks/visual_adv/](image_attacks/visual_adv/) | CE/PGD universal 이미지. 기저 이미지·목표 문장 코퍼스·미분 가능 surrogate | [run_visualadv.sh](attack_scripts/run_visualadv.sh) |
| `DeltaJP` | [image_attacks/delta_jp/](image_attacks/delta_jp/) | 원본 이미지에 universal delta 적용. 이미지 있는 데이터셋과 학습 분리 필요 | [run_deltajp.sh](attack_scripts/run_deltajp.sh) |

### Configs and reproduction scope

공격 설정은 `configs/attacks/`에서 확인하고 `--config`로 지정합니다.
필수 자산은 [DATA_LAYOUT.md](DATA_LAYOUT.md), 공식 설정과 재현 범위는 [OFFICIAL_ALIGNMENT.md](OFFICIAL_ALIGNMENT.md)를 참고하세요.

### Attack wrappers

쉘 래퍼는 `PYTHON`, `MODEL`, `DATASETS`, `LIMIT`로 실행 조건을 지정합니다.

| 스크립트 계열 | 추가 옵션 / 주의점 |
| --- | --- |
| FigStep·HADES·VisualAdv·DeltaJP | 기본 대상은 `llava_v1_5`. HADES는 기본 HADES 데이터셋, DeltaJP는 원본 이미지 필요 |
| `run_cs_dj.sh` | 기본 `MODEL=gpt-4o-mini`, `DATASETS=CS-DJ` |
| `run_jood.sh` | AdvBenchM의 네 `AUGS`를 각각 실행 |
| `run_jps.sh` | 기본 `MODEL=qwen2_vl_7b`, `CONFIG` 지정 가능 |
| `run_mml.sh` | 기본 GPT-4o snapshot 대상. `VARIANTS`, `JUDGE_MODEL`, `EVALUATE_LLAMA_GUARD` 지원 |
| [run_transfer.sh](attack_scripts/run_transfer.sh) | VisualAdv/DeltaJP의 Qwen2 surrogate → 별도 target 전이 실행 |
| [run_jps_full.sh](attack_scripts/run_jps_full.sh), [run_visualadv_full.sh](attack_scripts/run_visualadv_full.sh) | 장시간 프로필과 lineage 검증 캐시 실행 |

`LIMIT`는 원본 행 수 제한입니다. 전체 입력은 `LIMIT=0`으로 실행하고, 최적화 반복 수는 YAML에서 조절합니다.

## 5. Target and surrogate models

Target은 생성된 공격을 **평가할 모델**, surrogate는 공격 생성 시 사용하는
**미분 가능한 최적화 모델**입니다. 용도별로 아래 모델을 선택합니다.

### Target models

| 대표 target CLI 이름 | 체크포인트 / 서비스 |
| --- | --- |
| `llava_v1_5` | `llava-hf/llava-1.5-7b-hf` |
| `llava_v1_6` | `llava-hf/llava-v1.6-mistral-7b-hf` |
| `qwen2_vl_7b` | `Qwen/Qwen2-VL-7B-Instruct` |
| `qwen2_5_vl_7b` | `Qwen/Qwen2.5-VL-7B-Instruct` |
| `qwen3_vl_2b`, `qwen3_vl_8b` | `Qwen/Qwen3-VL-{2B,8B}-Instruct` |
| `internvl3_2b`, `internvl3_8b` | `OpenGVLab/InternVL3-{2B,8B}-hf` |
| `gpt-4o-mini`, `gpt-4o`, `gpt-4o-2024-08-06` | OpenAI API |

전체 alias는 [core/targets/__init__.py](core/targets/__init__.py)를 참고하세요.

```bash
# 평가 대상 변경: 기존 test cases를 그대로 사용
python run_generate_completions.py \
  --run_dir runs/example --model internvl3_2b --batch_size 1
```

### Surrogate models

VisualAdv·DeltaJP·HADES white-box의 최적화 모델은 `--surrogate-profile`로 선택합니다.

```bash
# 최적화 surrogate 변경: VisualAdv / DeltaJP / HADES white_box 공통 CE 경로
python run_generate_test_cases.py \
  --attack VisualAdv --dataset SafeBench --limit 3 \
  --surrogate-profile qwen3_vl_2b --save_dir runs/visualadv_qwen3
```

지원 프로필은 [surrogates.py](multimodalmodels/surrogates.py)를 참고하세요.

## 6. Single-attack pipeline

```text
Dataset loader → Behavior → Attack → TestCase
                                      ↓
                              Defense (선택) → Target → Completion → Judge → 지표
```

### Stage 1–3 quickstart

FigStep로 SafeBench 원본 3개의 공격 입력을 생성하고, 대상 모델 응답과 ASR을 평가합니다.

```bash
# 1. 공격 입력 생성
python run_generate_test_cases.py \
  --attack FigStep --dataset SafeBench --limit 3 \
  --save_dir runs/figstep_safebench

# 2. 대상 VLM 응답
python run_generate_completions.py \
  --run_dir runs/figstep_safebench \
  --model llava_v1_5 --batch_size 1

# 3. 원본 instruction 기준 판정 및 behavior-level ASR
python run_evaluate_completions.py \
  --run_dir runs/figstep_safebench --model llava_v1_5 \
  --judge_batch_size 1
```

단계별 출력 파일입니다.

| 파일 | 의미 |
| --- | --- |
| `behaviors.jsonl` | 실제 생성에 포함된 원본 Behavior |
| `test_cases.jsonl`, `images/` | 공격 입력과 생성 이미지 |
| `run_manifest.json` | 설정·코드 출처·정책·학습/제외 행·lineage |
| `completions_<model>.jsonl` | Stage 2 모델 응답 |
| `results_<model>.jsonl` | Stage 3 라벨이 붙은 응답 |
| `asr_results.jsonl` | 기본적으로 **실행 작업 디렉터리**에 append되는 ASR 요약 |

ASR 요약 경로는 `--asr_results_file`로 지정합니다. 실험마다 출력 폴더를 구분하세요.

MML 공식 평가에는 `--protocol mml_official`을 사용합니다.

```bash
python run_generate_completions.py --protocol mml_official \
  --run_dir runs/MML_word_replacement_HADES \
  --model gpt-4o-2024-08-06 --judge_model gpt-4o-mini-2024-07-18
```

최대 5회 시도하며 score 5에서 종료합니다. 소규모 실행은 Stage 1의 샘플 수를 줄이고 `--allow_subset`을 지정합니다.
방어 비교에는 `--protocol standard`를 사용합니다.

## 7. Jailbreak defenses

통합 Stage 2의 `--defense`로 연결된 방어는 5개입니다.
공격을 새로 생성하지 않고 같은 `test_cases.jsonl`에 적용하여 baseline과 비교합니다.

| CLI 방어 이름 | 구현 / 설정 | 동작 | 준비 / 실행 |
| --- | --- | --- | --- |
| `AdaShield` | [adashield.py](defenses/adashield.py) / [YAML](configs/defense_configs/AdaShield.yaml) | 기본 S 안전 프롬프트. A는 CLIP 기반 방어 프롬프트 검색 | [run_adashield.sh](defense_scripts/run_adashield.sh) |
| `AsD` | [asd.py](defenses/asd.py) / [YAML](configs/defense_configs/AsD.yaml) | 공식 protective patch와 안전 프롬프트 | [setup_asd.sh](defense_scripts/setup_asd.sh) / [run_asd.sh](defense_scripts/run_asd.sh) |
| `JailGuard` | [jailguard.py](defenses/jailguard.py) / [YAML](configs/defense_configs/JailGuard.yaml) | 이미지 변형별 응답 불일치로 탐지 | [setup_jailguard.sh](defense_scripts/setup_jailguard.sh) / [run_jailguard.sh](defense_scripts/run_jailguard.sh) |
| `VLMShield` | [vlmshield.py](defenses/vlmshield.py) / [YAML](configs/defense_configs/VLMShield.yaml) | CLIP 특징과 공개 MLP로 입력 쌍의 안전성 판정 | [setup_vlmshield.sh](defense_scripts/setup_vlmshield.sh) / [run_vlmshield.sh](defense_scripts/run_vlmshield.sh) |
| `CIDER` | [cider.py](defenses/cider.py) / [YAML](configs/defense_configs/CIDER.yaml) | diffusion denoising 전후 이미지·질문 유사도 변화로 탐지 | [setup_cider.sh](defense_scripts/setup_cider.sh) / [run_cider.sh](defense_scripts/run_cider.sh) |

### Setup and paired evaluation

사용할 방어의 자산을 준비합니다.

```bash
bash defense_scripts/setup_asd.sh
bash defense_scripts/setup_cider.sh
bash defense_scripts/setup_jailguard.sh
```

같은 공격 입력에 AdaShield를 적용하고 결과를 평가합니다.

```bash
python run_generate_completions.py \
  --run_dir runs/figstep_safebench --model llava_v1_5 \
  --batch_size 1 --defense AdaShield \
  --defense_config configs/defense_configs/AdaShield.yaml \
  --save_path runs/figstep_safebench/completions_llava_v1_5__defense-AdaShield.jsonl

python run_evaluate_completions.py \
  --run_dir runs/figstep_safebench --model llava_v1_5 \
  --judge_batch_size 1 \
  --completions_path runs/figstep_safebench/completions_llava_v1_5__defense-AdaShield.jsonl \
  --save_path runs/figstep_safebench/results_llava_v1_5__defense-AdaShield.jsonl
```

쉘 래퍼로도 실행할 수 있습니다.

```bash
RUN_DIR=runs/figstep_safebench MODEL=llava_v1_5 LIMIT=3 EVALUATE=1 \
  bash defense_scripts/run_adashield.sh
```

`RUN_DIR`·`MODEL`로 입력과 대상을 지정합니다. `LIMIT=0`은 전체 입력, `EVALUATE=1`은 평가까지 실행합니다.

### Algorithm and compatibility notes

방어별 자산·모델 조건은 위 YAML에서 확인하세요. 공식 설정과 재현 범위는 [OFFICIAL_ALIGNMENT.md](OFFICIAL_ALIGNMENT.md)에 정리했습니다.

## 8. Composed jailbreak generation

실제 3,000개 생성에 사용한 설정은 [five_methods_3000.yaml](configs/compositions/five_methods_3000.yaml)입니다.
FigStep·CS-DJ·MML·JOOD·JPS의 18개 조합을 사용하며, seed는 `20261104`입니다.

| 원본 데이터셋 | 선택한 질문 | 생성 샘플 |
| --- | ---: | ---: |
| SafeBench | 60 | 1,000 |
| HADES | 60 | 1,000 |
| MM-SafetyBench | 60 | 1,000 |
| 합계 | 180 | 3,000 |

| ID | 방법 조합 | 샘플 수 |
| --- | --- | ---: |
| P01 | FigStep + JOOD | 160 |
| P02 | FigStep + CS-DJ | 160 |
| P03 | FigStep + JPS | 160 |
| P04 | MML + JOOD | 160 |
| P05 | MML + CS-DJ | 160 |
| P06 | MML + JPS | 160 |
| P07 | JOOD + CS-DJ | 160 |
| P08 | JOOD + JPS | 160 |
| P09 | CS-DJ + JPS | 160 |
| T01 | FigStep + JOOD + CS-DJ | 180 |
| T02 | FigStep + JOOD + JPS | 180 |
| T03 | FigStep + CS-DJ + JPS | 180 |
| T04 | MML + JOOD + CS-DJ | 180 |
| T05 | MML + JOOD + JPS | 180 |
| T06 | MML + CS-DJ + JPS | 180 |
| T07 | JOOD + CS-DJ + JPS | 180 |
| Q01 | FigStep + CS-DJ + JOOD + JPS | 150 |
| Q02 | MML + CS-DJ + JOOD + JPS | 150 |
| **합계** | **18개 조합** | **3,000** |

JOOD가 포함된 1,840개 샘플은 원본 이미지 가중치 5~95%를 5% 간격으로 배정합니다.
19개 비율은 각각 96~97회, 무해 이미지 5종은 각각 368회 사용합니다.

미리보기 54개 생성 → 이미지 검토 → 전체 3,000개 생성 → 검증 순서입니다.
JPS 이미지·steer prompt·학습 CSV는 YAML의 `jps.source_run`에 준비합니다.

```bash
python -m pip install -r requirements-five-methods.txt
python -m core.prepare_five_method_assets
bash attack_scripts/run_five_method_dataset.sh preview
bash attack_scripts/run_five_method_dataset.sh review-preview --review-note '실제 이미지 검토 내용'
bash attack_scripts/run_five_method_dataset.sh full
bash attack_scripts/run_five_method_dataset.sh validate
```

위 명령은 [run_generate_five_method_dataset.py](run_generate_five_method_dataset.py)와 위 YAML을 사용합니다.
기본 출력은 `runs/five_methods_3000_mixed_sources_jood_ratios_20261104/`이며, 최종 샘플은 `full/test_cases.jsonl`에 저장됩니다.
다른 출력 경로는 모든 단계에 같은 `--output`을 지정합니다.

완성 샘플은 [Google Drive 공유 폴더](https://drive.google.com/drive/folders/1zxLWKwmC8yzDeXKC3Th7CwqjBchQbh4r?usp=drive_link)의 `3000_samples.zip`,
입력 자산은 같은 폴더의 `data.zip`에 있습니다.
상세 구성은 [FIVE_METHOD_DATASET.md](FIVE_METHOD_DATASET.md), 응답 생성·ASR 평가는 [6절](#6-single-attack-pipeline)을 참고하세요.

## 9. Evaluation and metrics

공격은 성공 여부와 응답 품질을, 방어는 탐지 성능과 정상 요청의 성능을 평가합니다.

### Jailbreak

| 지표 | 필요한 이유 | 확인하는 질문 |
| --- | --- | --- |
| ASR | 공격 효과를 성공 빈도로 비교 | 원래 공격 목표를 얼마나 자주 달성했는가? |
| Harmfulness | 성공 판정 안에서도 유해성의 정도를 구분 | 생성된 내용이 얼마나 유해한가? |
| Convincingness | 답변의 논리성과 일관성을 확인 | 내용이 일관되고 설득력 있게 구성되었는가? |
| Specificity | 일반적인 언급과 구체적인 도움을 구분 | 원래 요청에 얼마나 구체적으로 답했는가? |
| Contradiction Rate | 거절 문구와 유해한 도움의 동시 제공을 확인 | 거절하면서도 유해한 정보를 제공하는가? |
| Intent-conditioned ASR | 의도 이해 실패와 이해 후 거절을 구분 | 의도를 이해한 사례 중 얼마나 성공했는가? |

ASR은 판정된 원본 요청 중 하나 이상의 시도가 성공한 비율입니다. Intent-conditioned ASR은 의도를 이해했다고 판정된 사례를 분모로 사용합니다.

### Jailbreak defense

| 지표 | 필요한 이유 | 측정 방법 |
| --- | --- | --- |
| Detection Success Rate | 공격 탐지의 누락 정도를 확인 | 공격 중 탐지된 비율: `TP / (TP + FN)` |
| F1 Score | 공격 탐지와 정상 입력 오탐을 함께 고려 | 공격·정상 입력에서 Precision과 Recall의 조화평균 |
| Over-defense Rate | 정상 요청까지 차단하는 과잉 방어를 확인 | 정상 입력 중 잘못 차단하거나 거절한 비율 |
| Benign Performance | 정상 요청에 대한 답변 정확도·품질을 확인 | 동일 정상 과제의 방어 전후 정답률·평가 점수 비교 |
| ASR Reduction | 방어가 실제 공격 성공률을 줄이는지 확인 | 동일 공격 입력의 방어 전 ASR − 방어 후 ASR |
| Harmfulness Reduction | 방어 후 남은 유해 정보의 정도를 확인 | 동일 judge·루브릭으로 방어 전후 유해성 점수 비교 |

탐지 지표는 공격 입력을 positive로 둡니다. 방어 전후 비교에는 같은 입력·대상 모델·judge·루브릭·시도 예산을 사용합니다.

## 10. Validation

데이터 경로와 생성 결과를 검사합니다.

```bash
# 데이터셋 필수 경로 검사
python -m core.validation.data --layout-only

# 조합 결과 검사
python -m core.validation.composed_dataset runs/composed_pilot
```

캐시 검증 등 상세 안내는 [core/README.md](core/README.md)를 참고하세요.

## 11. Related documentation

| 문서 | 내용 |
| --- | --- |
| [COMPOSED_GENERATION.md](COMPOSED_GENERATION.md) | 조합 구성·실행·검증·평가 안내 |
| [DATA_LAYOUT.md](DATA_LAYOUT.md) | 별도 배포 데이터와 공격 자산의 경로 계약 |
| [core/README.md](core/README.md) | 스키마·인터페이스·registry |
| [OFFICIAL_ALIGNMENT.md](OFFICIAL_ALIGNMENT.md) | 공식 소스 revision, 재현 수준, 대체 사항 |
| [multi_attacks/mml/README.md](multi_attacks/mml/README.md) | MML replay·공통 CLI의 MML 프로토콜 세부사항 |
| [THIRD_PARTY.md](THIRD_PARTY.md) | 코드·데이터·가중치·폰트·API의 별도 라이선스 |

## Acknowledgement and citation
We thank the following open-source reposities. Each remains the property of
its upstream authors and is governed by its own license (see
[THIRD_PARTY.md](THIRD_PARTY.md) for license details).

    [1]  https://github.com/thuccslab/figstep
    [2]  https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models
    [3]  https://github.com/abc03570128/Jailbreaking-Attack-against-Multimodal-Large-Language-Model
    [4]  https://github.com/AoiDragon/HADES
    [6]  https://github.com/naver-ai/JOOD
    [7]  https://github.com/isXinLiu/MM-SafetyBench
    [8]  https://github.com/centerforaisafety/HarmBench
    [9]  https://github.com/TeamPigeonLab/CS-DJ
    [11] https://github.com/SaFo-Lab/AdaShield
    [12] https://github.com/PandragonXIII/CIDER
    [13] https://github.com/shiningrain/JailGuard
    [14] https://github.com/AngelAlita/AsD


## License

This project is released under a **dual-scope** licensing arrangement:

- The original source code and documentation directly developed by
  ai.resilience as part of the VLM-Jailbreak-Simulator project are
  released under the **MIT License** (see [LICENSE](LICENSE)).
- All third-party source code (vendored or imported), pre-trained model
  weights, datasets, closed-source APIs, and assets that the simulator
  exercises at runtime are **NOT** covered by the MIT License. They
  remain subject to their **original upstream licenses**, notices, and
  use restrictions, including (without limitation):

  - the **Llama 2 Community License** — governs the
    `cais/HarmBench-Llama-2-13b-cls` classifier used by
    `run_evaluate_completions.py` and the `llava-hf/llava-1.5-7b-hf`
    weights used as a target VLM;
  - the **LLaMA Research License** and **Vicuna delta-weights terms**
    (*non-commercial*) — flow into the MiniGPT-4 target pipeline via
    Vicuna v0 weights;
  - the **Tongyi Qianwen License Agreement** — governs the vendored
    Qwen-VL wrapper code under `multimodalmodels/qwen/`;
  - **Apache-2.0** / **BSD-3-Clause** terms — govern vendored
    MiniGPT-4, LLaVA, InstructBLIP / LAVIS, JOOD, HarmBench
    components, and most Python dependencies;
  - **MIT** — governs the FigStep, HADES, and HarmBench upstream
    repositories that the simulator vendors or imports from;
  - the **OpenAI Terms of Use** — govern OpenAI API calls made by
    the HADES and JOOD pipelines;
  - the original copyright of the **Arial** typeface (Monotype /
    Microsoft) — applies to the `Arial.ttf` font asset required by
    Hades;
  - **No explicit license** — applies to the Visual Adversarial
    Examples and Jailbreaking-Attack-against-MLLM upstream
    repositories. Absence of a license does **not** grant any rights;
    default copyright applies.

Users and redistributors are solely responsible for reviewing and
complying with the upstream license of every third-party component they
download, use, fine-tune, or redistribute through this repository. See
[THIRD_PARTY.md](THIRD_PARTY.md) for a runtime-oriented inventory of
the third-party components actually exercised by this project.

