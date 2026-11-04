
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

모든 예제는 저장소 루트에서 실행합니다. GPU 최적화, 가중치 다운로드, 외부 API 호출이
필요한 명령은 시간·메모리·API 비용을 발생시킬 수 있습니다.

## 1. Repository layout

주요 코드와 실행 진입점은 다음과 같습니다.

```text
vlm-jailbreak-simulator/
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

아래 명령은 저장소 루트에서 **새 환경을 구성할 때** 실행합니다.
`jb_env`가 이미 사용 중이면 다른 이름의 환경을 만들고, 실행 중인 실험의 패키지는 변경하지 마세요.

`PYTHONNOUSERSITE=1`은 [Python의 사용자 패키지 경로를 비활성화](https://docs.python.org/3.11/using/cmdline.html#envvar-PYTHONNOUSERSITE)하여
`~/.local`의 다른 버전이 섞이지 않게 합니다. Conda 환경 변수로 저장한 뒤 재활성화합니다.

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

PyTorch 설치 순서는 [공식 2.9.1 설치 안내](https://pytorch.org/get-started/previous-versions/#v291)를 따릅니다.
CLIP은 지정된 Git 커밋에서 빌드하므로 `git`이 필요하고,
JailGuard의 spaCy 벡터 모델 `en_core_web_md==3.7.1`도 고정된 공개 wheel로 설치합니다.

### Verify the environment

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

VS Code·노트북에서도 위 환경의 Python 인터프리터·커널을 선택하고,
환경 변수를 설정한 뒤에는 기존 커널을 재시작하세요.
모델 로딩은 [multimodalmodels/loaders.py](multimodalmodels/loaders.py),
체크포인트 revision은 [multimodalmodels/registry.py](multimodalmodels/registry.py)에서 관리합니다.

### Data, defense assets and optional metrics

데이터는 [3절](#3-dataset-loading), 방어용 이미지·확산 모델 자산은
[7절](#7-jailbreak-defenses)에 따라 준비합니다.
MML 단어 치환과 조합 데이터 생성에 필요한 영어 품사 태깅 모델은
`data.zip`의 `data/method_assets/nltk_data/`에 포함됩니다. 여러 실행이 공유하는 자산이며,
누락된 경우 아래 명령으로 준비합니다.

```bash
export NLTK_DATA="${VLM_JAILBREAK_DATA_ROOT:-$PWD/data}/method_assets/nltk_data"
python -m nltk.downloader -d "$NLTK_DATA" averaged_perceptron_tagger
```

SSIM·LPIPS 이미지 품질 지표를 사용할 때는 다음 패키지를 추가 설치합니다.
기준 패키지 버전은 `-c requirements.txt`로 유지합니다.

```bash
python -m pip install -c requirements.txt "scikit-image==0.25.2" "lpips==0.1.4"
python -m pip check
```

LPIPS·CLIP 유사도에는 해당 모델 가중치도 필요합니다.
실제 계산 여부와 옵션은 [9절](#9-evaluation-and-metrics)을 참고하세요.

| 환경 변수 | 용도 |
| --- | --- |
| `VLM_JAILBREAK_DATA_ROOT` | 별도 준비한 데이터 루트. 미지정 시 `<repo>/data` |
| `OPENAI_API_KEY` | OpenAI 대상 모델, CLAS judge, API 기반 공격 보조 처리 |
| `HF_TOKEN` | 접근 승인이 필요한 Hugging Face 모델 |
| `PYTHON` | 쉘 래퍼가 사용할 Python 실행 파일 |
| `CUDA_VISIBLE_DEVICES` | 사용할 GPU 선택 |
| `PYTHONNOUSERSITE` | `1`로 설정하여 사용자 site-packages 혼입 방지 |
| `NLTK_DATA` | MML 등에서 사용하는 NLTK 자산 경로 |

API 키 설정 형식은 [.env.example](.env.example)을 참고하세요.
일부 API 컴포넌트는 프로젝트 `.env`를 읽지만, 데이터 루트는 그 자동 로딩에 의존하지 말고
**Python 프로세스 시작 전에 export**하세요. 방어별 추가 자산 준비는 [7절](#7-jailbreak-defenses)에 있습니다.

## 3. Dataset loading

### Benchmark inputs and method assets

`data/`는 다음 두 디렉터리로 관리합니다. 실행 결과는 `runs/<실험명>/`에 저장합니다.

| 디렉터리 | 역할 | 예시 |
| --- | --- | --- |
| `data/benchmarks/` | 공격·방어 평가에 사용하는 벤치마크 원본 | AdvBench, AdvBench-M 원본 공개본, HADES, MM-SafetyBench, SafeBench |
| `data/method_assets/` | 각 방법론의 샘플 생성·최적화·재현에 필요한 입력과 공용 자산 | CS-DJ 이미지 풀, JOOD 부분집합·혼합 이미지, MML 공식 이미지, NLTK 태거 |

```text
data/
├── benchmarks/
│   ├── advbench/
│   ├── AdvBenchM/                    # 이전 original_release/의 전체 내용
│   ├── HADES/
│   ├── MM-SafetyBench/
│   └── SafeBench/
└── method_assets/
    ├── CS-DJ/
    ├── JOOD/                         # 이전 AdvBenchM/jood_subset/의 전체 내용
    ├── JailbreakInPieces/
    ├── VisualAdv/
    ├── nltk_data/
    └── MML/                          # 공식 replay용 선택 자산, 별도 다운로드
```

AdvBench-M 원본 공개본은 246개 질문 레코드와 160장 이미지의 범주별 풀입니다.
JOOD의 170문항·대표 이미지·혼합용 이미지는 `method_assets/JOOD/`에 따로 보관합니다.
CLI 이름 `--dataset AdvBenchM`은 `method_assets/JOOD/`의 170문항을 읽습니다.
원본 공개본과 JOOD 부분집합은 각각의 데이터 구성으로 관리합니다.

### Download and extract

새 `data.zip`에는 위의 `benchmarks/`와 `method_assets/` 구조가 들어 있습니다.
MML 단어 치환 생성과 조합 모드에서 사용하는 NLTK 태거는
`method_assets/nltk_data/`에 포함됩니다.
공식 MML 재현 자산은 [MML 준비 안내](multi_attacks/mml/README.md)에 따라
`data/method_assets/MML/`에 별도로 배치하세요.

[Google Drive의 기존 data.zip](https://drive.google.com/file/d/1z4Tw-D7GaQSGwhG86T0p2cJhsQxqCB-i/view?usp=drive_link)은
이전 평면 구조의 배포본입니다. 기존 ZIP을 사용하면 압축 해제 후 아래 이전 명령을 실행하세요.
이미 새 구조인 데이터에는 이전 명령을 다시 실행해도 파일을 옮기지 않습니다.
AdvBench-M 원본은 새 ZIP을 사용하거나 [공식 데이터 출처](https://github.com/abc03570128/Jailbreaking-Attack-against-Multimodal-Large-Language-Model/tree/fd98910ae4075fe587a1944095c00660aa24cc54/dataset)를
참고해 `benchmarks/AdvBenchM/`에 별도로 준비하세요.
다운로드 파일과 압축 해제본을 함께 보관하려면 최소 3 GB의 여유 공간을 준비하세요.

ZIP에는 최상위 `data/` 폴더가 들어 있습니다. **저장소 루트에서 압축을 해제**하여
`vlm-jailbreak-simulator/data/`에 배치합니다. `data/` 안에서 다시 압축을 풀어
`data/data/`가 되지 않도록 주의하세요.

```bash
# vlm-jailbreak-simulator/에서 실행: 기존 ZIP 다운로드는 필요한 경우에만 수행
curl --fail --location --retry 3 \
  'https://drive.usercontent.google.com/download?id=1z4Tw-D7GaQSGwhG86T0p2cJhsQxqCB-i&export=download&confirm=t' \
  --output ../data.zip

unzip -tq ../data.zip
unzip -n ../data.zip -d .
export VLM_JAILBREAK_DATA_ROOT="$PWD/data"

# 이전 ZIP을 두 디렉터리 구조로 정리: 기존 대상 파일은 덮어쓰지 않음
python -m core.migrate_data_layout --data-root "$VLM_JAILBREAK_DATA_ROOT" --dry-run
python -m core.migrate_data_layout --data-root "$VLM_JAILBREAK_DATA_ROOT"
export NLTK_DATA="$VLM_JAILBREAK_DATA_ROOT/method_assets/nltk_data"
```

다른 위치에 압축을 풀었다면 `VLM_JAILBREAK_DATA_ROOT`를 그 안의 `data/` 폴더로 지정하세요.
이전된 파일의 내용과 기존 실행 기록은 수정하지 않습니다. 이동 기록은
`method_assets/provenance/data_layout_migration.json`에 남깁니다.
새 설정은 두 디렉터리 경로를 사용하고, 공통 resolver는 이전 `data/HADES/...`,
`data/AdvBenchM/jood_subset/...` 같은 설정 경로도 계속 읽습니다.

### Validate the data

```bash
python -m core.validation.data --dataset SafeBench --dataset AdvBench --layout-only
python -m core.validation.data --dataset SafeBench --dataset AdvBench
```

`--dataset NAME`을 반복하여 검사할 데이터셋을 선택합니다. 전체 등록 데이터셋의
자산을 준비한 경우에는 `--dataset`을 생략하여 모두 검사할 수 있습니다.
공식 AdvBench-M 원본의 파일별 해시와 공개 범위는
`data/benchmarks/AdvBenchM/{manifest.json,availability.json}`에 기록되어 있습니다.
방법론별 필수 파일은 [4절](#4-jailbreak-attacks)과 [DATA_LAYOUT.md](DATA_LAYOUT.md)를 참고하세요.

[core/paths.py](core/paths.py)는 `data/benchmarks/...`, `data/method_assets/...`를
`$VLM_JAILBREAK_DATA_ROOT` 아래로 해석합니다. 기본 루트는 `<repo>/data`입니다.
데이터 루트 밖의 절대 경로는 그대로 사용하고, 그 밖의 상대 경로는 저장소 기준입니다.
생성 CLI는 필요한 입력과 방법론 자산을 검사하고, 누락 시 생성 전에 실패합니다.

### Registered loaders

등록 허브는 [core/datasets/__init__.py](core/datasets/__init__.py),
공통 인터페이스는 [core/dataset.py](core/dataset.py)입니다.
행 수는 [core/data.py](core/data.py)의 **원본 전체 로드 기대값**이며,
harmful-only 필터·학습 분리 후 생성할 수 있는 수와는 다릅니다.

| CLI 데이터셋 이름 | 로더 | 주요 입력 경로: 데이터 루트 기준 | 기대 행 수 |
| --- | --- | --- | ---: |
| `AdvBench` | [advbench.py](core/datasets/advbench.py) | `advbench/harmful_behaviors.csv` | 520 |
| `AdvBenchM` | [advbenchm.py](core/datasets/advbenchm.py) | `method_assets/JOOD/` (historical JOOD subset; original release is `benchmarks/AdvBenchM/`) | 170 |
| `CS-DJ` | [cs_dj.py](core/datasets/cs_dj.py) | `CS-DJ/instructions/*.json` | 750 |
| `HADES` | [hades.py](core/datasets/hades.py) | `HADES/hades.csv` 및 참조 이미지 | 750 |
| `HarmBench` | [harmbench.py](core/datasets/harmbench.py) | `HarmBench/harmbench_standard.json` | 200 |
| `MM-SafetyBench` | [mm_safetybench.py](core/datasets/mm_safetybench.py) | `MM-SafetyBench/` | 1,680 |
| `SafeBench` | [safebench.py](core/datasets/safebench.py) | `SafeBench/safebench.csv` | 500 |

`CS-DJ`는 공격 이름인 동시에 데이터셋 등록 이름입니다.
데이터셋 로딩에 필요한 자산과 공격 최적화에 필요한 추가 자산은 서로 다를 수 있습니다.
VisualAdv의 최적화 목표 코퍼스는 `data/method_assets/VisualAdv/derogatory_corpus.csv`에 두고,
기저 이미지는 `data/method_assets/FORCE/clean.jpeg`를 사용합니다.

### Schema and harmful-only policy

[core/schema.py](core/schema.py)의 세 레코드로 데이터와 실험을 연결합니다.

| 레코드 | 역할 |
| --- | --- |
| `Behavior` | 원본 지시문, 고유 ID, 데이터셋, 위험 유형, 이미지 및 보조 필드 |
| `TestCase` | 공격으로 만든 실제 입력 텍스트·이미지, `behavior_id`, 시도 번호 |
| `Completion` | 대상 모델 응답, 차단 여부, judge 라벨 및 계측 메타데이터 |

원본 `Behavior.instruction`을 평가 기준으로 유지하며, 공격 과정에서 바뀐 텍스트로
정답 기준을 바꾸지 않습니다. 원본 하나에서 여러 시도가 생성될 수 있으므로
Behavior 수와 TestCase 수는 같지 않을 수 있습니다.

Stage 1은 기본적으로 `eval_type=harmful`만 선택합니다.
SafeBench의 advice, MM-SafetyBench의 advice/opinion 등은 제외하고,
그다음 공격 학습용 행을 제외한 뒤 `--limit`, `--per_category`를 적용합니다.
허용 유형·제외 사유·실제 평가 행 수는 `run_manifest.json`에 남깁니다.

`--eval-types all`은 별도의 전체 벤치마크 평가용입니다.
그 결과를 harmful-instruction 탈옥 셋에 합치지 않습니다.
advice/opinion을 benign으로 재라벨링해서 FPR 대조군으로 사용하지도 않습니다.

## 4. Jailbreak attacks

[core/attacks/__init__.py](core/attacks/__init__.py)에 등록된 개별 공격은 9개입니다.
기본 설정 경로는 `configs/attacks/<CLI 이름>.yaml`이며,
실행 시 `--config`로 다른 프로필을 명시할 수 있습니다.

| CLI 공격 이름 | 구현 위치 | 생성 방식 및 주요 의존성 | 실행 스크립트 |
| --- | --- | --- | --- |
| `FigStep` | [multi_attacks/figstep/](multi_attacks/figstep/) | 지시문을 이미지 목록으로 렌더링하고 고정 프롬프트 결합. 폰트 차이 기록 | [run_figstep.sh](attack_scripts/run_figstep.sh) |
| `CS-DJ` | [multi_attacks/cs_dj/](multi_attacks/cs_dj/) | Qwen 분해 → CLIP 대조 이미지 검색 → 3×4 콜라주. 외부 이미지 풀·폰트 필요 | [run_cs_dj.sh](attack_scripts/run_cs_dj.sh) |
| `MML` | [multi_attacks/mml/](multi_attacks/mml/) | 공개 이미지·프롬프트 replay. 공통 Stage 2의 MML 프로토콜로 CLAS 조기 종료 | [run_mml.sh](attack_scripts/run_mml.sh) |
| `HADES` | [multi_attacks/hades/](multi_attacks/hades/) | 기본 white-box 상단 패치 3,000 epochs. 별도 official 설정은 공개 이미지 6단계 replay | [run_hades.sh](attack_scripts/run_hades.sh) |
| `JOOD` | [multi_attacks/jood/](multi_attacks/jood/) | benign 항목과 Mixup/CutMix/ResizeMix/RandAug. 개별 runner는 AdvBenchM 대상 | [run_jood.sh](attack_scripts/run_jood.sh) |
| `JPS` | [multi_attacks/jps/](multi_attacks/jps/) | Qwen2 시각 최적화와 Qwen2.5-14B 에이전트의 피드백·수정. 공식 CSV·기저 이미지 별도 필요 | [run_jps.sh](attack_scripts/run_jps.sh) |
| `JailBound` | [multi_attacks/jailbound/](multi_attacks/jailbound/) | 안전 경계 probe 학습 후 이미지·20-token suffix 공동 최적화. 논문 기반 adapted 구현 | [run_jailbound.sh](attack_scripts/run_jailbound.sh) |
| `VisualAdv` | [image_attacks/visual_adv/](image_attacks/visual_adv/) | CE/PGD universal 이미지. 기저 이미지·목표 문장 코퍼스·미분 가능 surrogate | [run_visualadv.sh](attack_scripts/run_visualadv.sh) |
| `DeltaJP` | [image_attacks/delta_jp/](image_attacks/delta_jp/) | 원본 이미지에 universal delta 적용. 이미지 있는 데이터셋과 학습 분리 필요 | [run_deltajp.sh](attack_scripts/run_deltajp.sh) |

### Configs and reproduction scope

CS-DJ, JOOD, MML, JPS는 공식 설정을 지향하는 단일 기본 YAML을 사용합니다.
다른 공격에는 `*_official.yaml`, transfer 등 별도 프로필을 제공합니다.
특히 다음은 구분해야 합니다.

- JPS 기본 설정에 필요한 `advbench_subset.csv`와 `clean.jpeg`는 별도로 준비합니다.
  조합 모드의 자산 설정은 effective config에서 확인할 수 있습니다.
- CS-DJ는 CC3M 이미지 풀·공식 폰트가 필요합니다.
- MML replay에 필요한 공식 이미지·메타데이터와 hash-pinned upstream `const.py`는
  [준비 안내](multi_attacks/mml/README.md)에 따라 별도로 준비합니다.
- JailBound는 논문 기반 adapted 구현이며 LLaVA-1.5, pooling, suffix 목적함수의
  구체적인 설정을 manifest에 기록합니다.
- `algorithm_fidelity`는 코드의 충실도, `experiment_profile`은 실행 설정,
  `paper_reproduction`은 공격 생성 측 재현 조건을 기록합니다.

각 변경점·upstream revision은 [OFFICIAL_ALIGNMENT.md](OFFICIAL_ALIGNMENT.md)에 있습니다.
실행 시 `run_manifest.json`의 실제 resolved config와 lineage까지 확인하세요.

### Attack wrappers

일반 래퍼는 `PYTHON`, `MODEL`, `DATASETS`, `LIMIT`를 사용하지만
데이터셋 호환성, 기본 모델, judge와 추가 옵션은 스크립트별로 다릅니다.

| 스크립트 계열 | 추가 옵션 / 주의점 |
| --- | --- |
| FigStep·HADES·VisualAdv·DeltaJP | 기본 대상은 `llava_v1_5`. HADES는 기본 HADES 데이터셋, DeltaJP는 원본 이미지 필요 |
| `run_cs_dj.sh` | 기본 `MODEL=gpt-4o-mini`, `DATASETS=CS-DJ` |
| `run_jood.sh` | 기본 GPT-4 Turbo 대상, AdvBenchM, 네 `AUGS`를 각각 실행 |
| `run_jps.sh` | 기본 `MODEL=qwen2_vl_7b`, `CONFIG` 지정 가능 |
| `run_mml.sh` | 기본 GPT-4o snapshot 대상. `VARIANTS`, `JUDGE_MODEL`, `EVALUATE_LLAMA_GUARD` 지원 |
| `run_jailbound.sh` | **단수** `DATASET`, 기본 MM-SafetyBench·1행. `RUN_DIR`, `BOUNDARY` 지정 가능 |
| [run_transfer.sh](attack_scripts/run_transfer.sh) | VisualAdv/DeltaJP의 Qwen2 surrogate → 별도 target 전이 실행 |
| [run_jps_full.sh](attack_scripts/run_jps_full.sh), [run_visualadv_full.sh](attack_scripts/run_visualadv_full.sh) | 장시간 프로필과 lineage 검증 캐시 실행 |

환경 변수의 `LIMIT=0`은 보통 래퍼에서 전체 실행을 뜻합니다.
직접 Python CLI에는 필요한 양의 `--limit`을 지정하거나 옵션을 생략하세요.
`LIMIT=3` / `--limit 3`은 원본 행 수 제한이지 최적화 iteration 축소가 아닙니다.
생성만 필요한 경우 아래 Stage 1을 직접 실행하면 대상 모델 질의·평가를 분리할 수 있습니다.

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
| `gpt-4o-mini`, `gpt-4o`, `gpt-4o-2024-08-06`, `gpt-4-turbo-2024-04-09` | OpenAI API |

전체 alias는 [core/targets/__init__.py](core/targets/__init__.py)를 참고하세요.

```bash
# 평가 대상 변경: 기존 test cases를 그대로 사용
python run_generate_completions.py \
  --run_dir runs/example --model internvl3_2b --batch_size 1
```

### Surrogate models

VisualAdv, DeltaJP, HADES white-box의 공통 CE 최적화에는
`llava_v1_5`, `llava_v1_6`, `qwen2_vl_7b`,
`qwen3_vl_2b`, `qwen3_vl_8b`, `internvl3_2b`, `internvl3_8b`를 선택할 수 있습니다.
프로필 정의는 [multimodalmodels/surrogates.py](multimodalmodels/surrogates.py)에 있습니다.

```bash
# 최적화 surrogate 변경: VisualAdv / DeltaJP / HADES white_box 공통 CE 경로
python run_generate_test_cases.py \
  --attack VisualAdv --dataset SafeBench --limit 3 \
  --surrogate-profile qwen3_vl_2b --save_dir runs/visualadv_qwen3
```

공통 CE 프로필은 원본 RGB 텐서를 각 모델의 미분 가능한 전처리에 통과시킵니다.
JPS는 공통 Qwen2 adapter, JailBound는 LLaVA-1.5 전용 내부 adapter를 사용합니다.
CIDER는 LLaVA-1.5 target의 encoder를 사용합니다.

## 6. Single-attack pipeline

```text
Dataset loader → Behavior → Attack → TestCase
                                      ↓
                              Defense (선택) → Target → Completion → Judge → 지표
```

### Stage 1–3 quickstart

소규모 예제로 FigStep 3개 원본을 생성하고 방어 없이 응답·평가합니다.
대상 모델과 기본 HarmBench judge는 각각 가중치와 실행 자원이 필요합니다.

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

주요 출력은 다음과 같습니다. Stage 1만 끝났다면 ASR은 아직 측정되지 않은 상태입니다.

| 파일 | 의미 |
| --- | --- |
| `behaviors.jsonl` | 실제 생성에 포함된 원본 Behavior |
| `test_cases.jsonl`, `images/` | 공격 입력과 생성 이미지 |
| `run_manifest.json` | 설정·코드 출처·정책·학습/제외 행·lineage |
| `completions_<model>.jsonl` | Stage 2 모델 응답 |
| `results_<model>.jsonl` | Stage 3 라벨이 붙은 응답 |
| `asr_results.jsonl` | 기본적으로 **실행 작업 디렉터리**에 append되는 ASR 요약 |

ASR 로그 위치는 Stage 3의 `--asr_results_file`로 바꿀 수 있습니다.
공격·대상·방어·judge·seed가 다른 실험은 별도 출력 경로로 보존하세요.
기존 결과 경로를 재사용하면 일반 Stage 2/3 출력이 덮어써질 수 있습니다.

MML의 CLAS 판정이 다음 시도를 제어하는 공식 평가 루프는 일반 3단계 평가와 다릅니다.
공통 Stage 2의 `--protocol mml_official` 옵션이나
[run_mml.sh](attack_scripts/run_mml.sh)를 사용합니다.

```bash
python run_generate_completions.py --protocol mml_official \
  --run_dir runs/MML_word_replacement_HADES \
  --model gpt-4o-2024-08-06 --judge_model gpt-4o-mini-2024-07-18
```

이 모드는 최대 5회 예산·score-5 조기 종료를 유지하고, completions와 판정된 results,
ASR/RSR/query 요약을 한 번에 기록합니다. 작은 실험은 Stage 1에서 behavior 수를
줄인 뒤 `--allow_subset`을 지정합니다. Stage 2의 `--limit`으로 시도 행을 잘라내지 않습니다.
프로토콜 구현은 [core/protocols/mml.py](core/protocols/mml.py)에 있습니다.
공식 모드는 baseline 전용이며, 방어 비교는 양쪽 모두 기본 `--protocol standard`와
동일한 judge·시도 예산을 사용합니다.

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

필요한 방어의 setup만 실행하세요. setup은 자산을 다운로드하거나 패키지를 설치합니다.

```bash
bash defense_scripts/setup_asd.sh
bash defense_scripts/setup_cider.sh
bash defense_scripts/setup_jailguard.sh
```

앞 절의 baseline과 **동일 입력**에 AdaShield를 적용하는 예제입니다.

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

동일한 작업을 래퍼로 실행할 수도 있습니다.

```bash
RUN_DIR=runs/figstep_safebench MODEL=llava_v1_5 LIMIT=3 EVALUATE=1 \
  bash defense_scripts/run_adashield.sh
```

방어 래퍼의 공통 변수는 `RUN_DIR`, `MODEL`, `DEFENSE_CONFIG`, `LIMIT`,
`EVALUATE`, `PYTHON`입니다. 기본 `LIMIT=0`은 전체 입력, `EVALUATE=0`은 응답 생성까지만입니다.
방어별 결과는 `__defense-<이름>` 파일로 분리됩니다.
일부 입력·모델 제한이 있으므로 단일 이미지에서 먼저 확인하세요.

### Algorithm and compatibility notes

- **AdaShield:** S는 공식 프롬프트와 `query + shield + query` 배치를 유지합니다.
  A inference는 [예제 설정](configs/defense_configs/AdaShield-A.example.yaml)과
  upstream refinement의 `final_table.csv`가 필요합니다.
- **AsD:** 공식 패치 SHA-256을 확인하고 seed 42, 이미지 면적 25%, alpha 0.7을 사용합니다.
  입력 변환이지 차단 detector가 아니므로 `blocked=false`가 정상일 수 있습니다.
- **JailGuard:** PL 정책(RR/BL/RP)으로 변형 8개, 응답 앞 100자,
  spaCy 유사도/KL threshold 0.025를 사용합니다.
  통과한 입력은 원본에 대한 후속 질의 1회가 추가됩니다.
  setup으로 설치하는 `en_core_web_md`를 사용합니다.
- **CIDER:** LLaVA-1.5 target과 약 2.21 GB의 공식 guided-diffusion 체크포인트를 사용합니다.
  denoising 0/50/…/350, threshold `-0.003936767578125`를 사용합니다.
  주 위협 모델은 VisualAdv/DeltaJP 같은 최적화 perturbation입니다.

방어별 upstream revision, 모델·runtime 설정과 재현 범위는
[OFFICIAL_ALIGNMENT.md](OFFICIAL_ALIGNMENT.md)에 정리했습니다.

## 8. Composed jailbreak generation

최신 3,000개 생성 워크플로는 [FIVE_METHOD_DATASET.md](FIVE_METHOD_DATASET.md)를 참고하세요.
FigStep, CS-DJ, MML, JOOD, JPS의 18개 조합으로 구성합니다.

일반 조합 엔진의 [v3 설정](configs/compositions/jailbreak_dataset_v3.yaml)은
**8개 조합, 최종 1,450개 / 후보 1,740개**로 구성합니다.

| ID | 적용 순서 | 최종 수량 |
| --- | --- | ---: |
| P01 | CS-DJ-Split → FigStep | 200 |
| P02 | CS-DJ-Split → MML | 200 |
| P03 | FigStep → JOOD | 200 |
| P05 | HADES → JailBound | 200 |
| P08 | JPS → JailBound | 200 |
| T01 | CS-DJ-Split → FigStep → JailBound | 150 |
| T02 | CS-DJ-Split → MML → JailBound | 150 |
| T04 | JPS → JailBound → JOOD | 150 |

```bash
# 설정 검사: 데이터·모델 로딩 없음
python run_generate_composed_dataset.py --dry-run
# 데이터·수량·설정 사전 검사
python run_generate_composed_dataset.py --preflight
# 레시피당 3개, 총 24개 pilot. 최적화 단계는 GPU와 시간이 필요합니다.
python run_generate_composed_dataset.py --per-recipe 3 \
  --save-dir runs/composed_jailbreak_pruned_pilot
python -m core.validation.composed_dataset runs/composed_jailbreak_pruned_pilot
```

`--recipes P01,T01`로 범위를 제한할 수 있습니다. 단계별 구현은
[core/composition_components.py](core/composition_components.py)에 있습니다.
소수 입력에서도 설정된 HADES·JPS·JailBound 최적화 루프를 수행합니다.
JPS는 `configs/attacks/JPS.yaml`의 학습 CSV와 초기 이미지를 준비해야 합니다.

그룹 실행은 `attack_scripts/run_composed_dataset_3000.sh`,
백그라운드 실행은 `run_composed_dataset_3000_nohup.sh`를 사용합니다.
스크립트 이름과 별개로 현재 YAML의 수량을 따르며, 기본 출력은
`runs/composed_jailbreak_pruned_20261103`입니다. `CONFIG`, `OUTPUT_ROOT`,
`PYTHON`, `PREFLIGHT_ONLY`로 실행 조건을 지정할 수 있습니다.
최신 5개 방법·3,000개 생성기와 이 일반 조합 실행기는 별개의 워크플로입니다.

후보 체크포인트는 10회마다 저장하며 설정·입력·코드 해시가 일치해야 재개합니다.
`nohup` 실행은 SSH·VS Code 종료 후에도 유지되지만 서버 재시작을 견디지는 않습니다.

모든 조합 결과는 `adapted_composition`, `paper_reproduction: false`로 기록합니다.
원본 질문별 split을 유지하고, 구조·중복·결정적 순서로 최종 선별합니다.
대상 응답이나 ASR로 성공한 입력만 고르는 과정은 없습니다.
출력은 `test_cases.jsonl`, `behaviors.jsonl`, 이미지 및
`composition_manifest.json`/`composition_qc.json`입니다.
타깃 추론과 판정은 [6절](#6-single-attack-pipeline)의 별도 단계에서 실행합니다.

## 9. Evaluation and metrics

기본 judge는 HarmBench입니다. Stage 3의 `--judge`로
`harmbench`, `beaver_dam`, `llama_guard`, `mml_clas`를 선택할 수 있고,
`--judge_model`로 모델을 지정합니다. Llama Guard는 모델 이름 지정이 필수입니다.
논문 프로토콜별 평가와 공통 judge 비교 결과를 분리하세요.

지표 구현은 [core/metrics.py](core/metrics.py),
멀티모달 분석은 [core/multimodal_metrics.py](core/multimodal_metrics.py),
집계 CLI는 [run_extended_metrics.py](run_extended_metrics.py)입니다.

| 지표군 | 지원 내용 / 필요한 입력 |
| --- | --- |
| 공격 성공 | behavior-level ASR, attempt ASR, Pass@k, 예산별 ASR, Wilson 95% CI |
| 평가 커버리지 | 요청·생성·판정 건수, 미판정·누락, category/eval_type별 ASR, macro·safety-worst ASR |
| 방어 비교 | 같은 입력의 baseline/defended 결과로 ASR 감소, 조건부 mitigation·regression, exact McNemar |
| 이진 탐지 | TP/TN/FP/FN, accuracy, precision/recall/F1, FPR/FNR, balanced accuracy, MCC |
| 연속 score | score가 있을 때 AUROC/AUPRC, FPR@95%TPR, TPR@1%FPR, EER, Brier, ECE |
| 효율·비용 | 시도·질의·시간·첫 성공까지의 비용, 방어 overhead. 토큰·API 비용·GPU 값은 기록된 경우만 |
| 추가 비교 | 전이 ASR, judge 일치율·kappa, seed 안정성, text/image/joint ablation, 변형·JPEG/resize 강건성 |
| 이미지·응답 품질 | L-inf/L2/PSNR/SSIM, 선택적 LPIPS/CLIP. 거절·부분 이행·severity·benign utility는 명시적 주석 필요 |

### Definitions and prerequisites

ASR은 기본적으로 **판정된 원본 Behavior 중 하나 이상의 시도가 성공한 비율**입니다.
미판정은 성공/실패로 임의 처리하지 않고 coverage로 함께 보고합니다.
행 단위 attempt ASR과 behavior-level ASR을 섞지 않으며,
비교 시 같은 대상·judge·입력·시도 예산을 맞춰야 합니다.
safety-worst category ASR은 가장 높은 공격 성공률입니다.

탐지에서는 **악성 입력을 positive**로 정의합니다.
`FPR = FP / (FP + TN)`, `FNR = FN / (TP + FN)`이며 분모가 없으면 `null`입니다.
FPR 및 양쪽 클래스를 포괄하는 탐지 성능 해석에는 독립적으로 검토한 benign 입력이 필요합니다.
AsD/AdaShield 같은 입력 변환 방어의 차단 여부를 탐지 정답처럼 사용하지 않습니다.
탐지 decision metadata가 없는 경우 해당 지표는 unavailable로 남습니다.
공격 응답의 judge 라벨과 입력의 악성/정상 ground truth는 서로 다른 개념입니다.

### Extended report

앞의 baseline과 AdaShield 결과를 비교하는 예입니다.

```bash
python run_extended_metrics.py \
  --results runs/figstep_safebench/results_llava_v1_5__defense-AdaShield.jsonl \
  --baseline-results runs/figstep_safebench/results_llava_v1_5.jsonl \
  --behaviors runs/figstep_safebench/behaviors.jsonl \
  --output runs/figstep_safebench/extended_metrics.json
```

추가 입력은 실제 생성·평가된 JSONL을 준비한 경우에만 지정합니다.

- `--benign-results`, `--benign-baseline-results`: FPR·benign utility 비교.
- `--source-results`, `--second-judge-results`: 전이·judge 비교.
- `--seed-results`, `--variant NAME=PATH`, `--transformed NAME=PATH`: 여러 번 지정 가능.
- `--text-only-results`, `--image-only-results`: 동일 사례의 modality ablation.
- `--image-pairs`: reference/candidate 이미지 쌍. LPIPS/CLIP은 별도 패키지·가중치와 opt-in 필요.

공통 방향은 “score가 높을수록 악성”이며 CIDER/JailGuard score를 변환해 집계합니다.
누락된 계측·주석은 추정하지 않고 `null`/unavailable로 유지합니다.
배치 target latency는 행별 실측값이 아니라 배치 시간의 균등 분배 추정치입니다.
입력 형식 전체는 `python run_extended_metrics.py --help`를 참고하세요.

## 10. Validation

[core/validation/](core/validation/)에서 데이터 경로, 공식 프로필의 준비 조건,
생성된 조합 결과와 캐시 계보를 검증합니다.

```bash
# 데이터셋 필수 경로 검사
python -m core.validation.data --layout-only

# 조합 결과의 harmful-only, 수량, 중복, 아티팩트·단계 연결·split 검사
python -m core.validation.composed_dataset runs/composed_jailbreak_v3_pilot
```

캐시 계보는 `python -m core.validation.artifact_lineage`로 별도 검사할 수 있습니다
(`--attack`, `--config`, `--artifact` 필요).
이 검사는 자산·설정·산출물의 무결성을 확인합니다.
실제 공격 성능은 [6절](#6-single-attack-pipeline)의 응답 생성·평가와
[9절](#9-evaluation-and-metrics)의 지표 집계로 측정합니다.

## 11. Related documentation

| 문서 | 내용 |
| --- | --- |
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
  - the **OpenAI Terms of Use** — govern the `gpt-3.5-turbo` and
    `gpt-4-turbo-2024-04-09` API calls made by the Hades and JOOD
    pipelines;
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

