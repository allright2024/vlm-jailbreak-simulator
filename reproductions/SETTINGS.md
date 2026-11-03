# 재현 실험 설정 보존

`dev_norep`은 전용 재현 실행기를 제외한 브랜치다. 방법론 구현과 기존 설정 파일
29개는 유지했고, 33개 실험 기록의 설정을 아래 데이터 파일로 정리했다.

- [실험별 설정 목록](../configs/experiments/reproduction_settings/index.json)
- [실행기에 정의돼 있던 기본값](../configs/experiments/reproduction_settings/runner_defaults.json)
- [FigStep](../configs/experiments/reproduction_settings/figstep.json)
- [JOOD MiniGPT-4](../configs/experiments/reproduction_settings/jood_minigpt4.json)
- [JPS 직접 최적화](../configs/experiments/reproduction_settings/jps_trained.json)
- [MML 변형](../configs/experiments/reproduction_settings/mml_variants.json)
- [AdaShield](../configs/experiments/reproduction_settings/adashield_figstep.json)
- [JailGuard int8](../configs/experiments/reproduction_settings/jailguard_8bit.json)

각 실험 JSON의 `recorded_settings`는 보존된 manifest·plan·runtime에서 가져온 값이다.
`evidence`는 원본 기록과 해시를, `runner_default_sources`는 삭제 전 실행기 기본값의
출처를 연결한다. 생성·판정 모델, sampling, 토큰 제한, seed, 배치 크기,
최적화 옵션과 데이터 범위를 가능한 원본 기록대로 유지했다.
FigStep의 생성 설정과 후속 판정 설정처럼 서로 다른 단계의 기록은 분리했다.

실제 실행 기록이 기본값보다 우선한다. 명령행에서 바꾼 값이나 외부 모델 설정을
기본값만으로 추정하지 않는다. 기본값 파일에는 정적으로 확인한 값만 담았으며,
`runtime_resolved_fields`는 해당 실행 기록에서 확인해야 하는 항목이다.
이 스냅샷은 실행 코드가 없고 공통 CLI가 자동으로 읽는 설정 형식도 아니다.

`configs/attacks/`, `configs/defense_configs/`와 기존 실험 YAML은 이번 정리에서
바이트를 변경하지 않았다. 기본 adapter와 원본 모델 구현·판정기·프로토콜이 다른
실험은 설정 JSON의 scope와 근거를 함께 확인해야 한다.
예를 들어 JOOD MiniGPT-4 실험의 HF 판정기는 GPT-4.1이고,
일반 JOOD 기본 YAML의 역사적 모델 이름과 다르다.

JailGuard reference의 원본 runtime YAML은
[별도 설정 파일](../configs/experiments/reproduction_settings/jailguard_reference_minigpt4.yaml)로 보존했다.
데이터·가중치·최적화 산출물의 로컬 경로는 당시 기록이며 자산 자체는 포함하지 않는다.
