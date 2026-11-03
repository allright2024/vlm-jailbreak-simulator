# 보존된 실험 설정

기존 재현 실험 33건의 설정을 데이터 파일로 보존했다.

- [실험별 설정 목록](index.json)
- [삭제 전 실행기의 기본값](runner_defaults.json)
- [JailGuard reference runtime 설정](jailguard_reference_minigpt4.yaml)

각 실험 JSON의 `recorded_settings`에는 당시 manifest·plan·runtime에서 추출한
설정이 들어 있다. 실제 실행 기록이 실행기 기본값보다 우선한다.
`runtime_resolved_fields`는 기본값만으로 확정할 수 없었던 항목이다.
이 스냅샷은 실행 코드가 아니며 공통 CLI가 자동으로 읽는 설정 형식도 아니다.

`reproductions/` 결과 모음과 전용 실행기는 삭제했다. JSON에 남아 있는
`reproductions/...` 경로와 `evidence` 해시는 과거 출처 식별자이며 현재 파일
링크나 실행 의존성이 아니다. 해당 원본은 Git 커밋 `426029a`에서 확인할 수 있다.
설정값과 출처 식별자는 그대로 보존했다.

`configs/attacks/`, `configs/defense_configs/`와 기존 실험 설정도 유지한다.
일반 adapter와 원본 모델 구현·판정기·프로토콜이 다른 실험은 각 JSON의
`scope`와 설정을 함께 확인해야 한다. 데이터·가중치·최적화 산출물의 로컬
경로는 당시 기록이며 해당 자산 자체는 포함하지 않는다.
