from typing import Any, Dict, Optional


class BaseDefense:

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def apply(self, image_filename: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Args:
            image_filename: images/ 디렉토리 기준 파일명
            prompt: 원본 텍스트 프롬프트

        Returns:
            {
                "blocked": bool,
                "image_filename": str,   # 방어 적용 후 이미지 파일명 (변형 없으면 원본 그대로)
                "prompt": str,           # 방어 적용 후 프롬프트
                "default_response": Optional[str],  # blocked=True일 때 target model 대신 사용할 응답
                "response": Optional[str],  # 방어가 자체 생성한 최종 응답
            }
        """
        raise NotImplementedError
