"""Gemini 클라이언트 인증 분기 (Vertex AI 서비스 계정 우선, API Key는 fallback)."""

from __future__ import annotations

import json
import os
import tempfile

from google import genai

from translation_web_app.paths import PROJECT_ROOT

DEFAULT_SERVICE_ACCOUNT_FILE = "smartthings-explore-6d1922d106bd.json"
DEFAULT_PROJECT_ID = "smartthings-explore"
# Gemini 3.x/3.5 프리뷰 모델은 us-central1 등 리전 엔드포인트에선 404 — global에서만 서빙됨.
# 2.5 계열은 global/리전 둘 다 동작하므로 global을 공통 기본값으로 둔다.
DEFAULT_LOCATION = "global"


def _vertex_client(project_id: str) -> genai.Client:
    location = os.getenv("GCP_LOCATION", DEFAULT_LOCATION)
    return genai.Client(vertexai=True, project=project_id, location=location)


def build_gemini_client(raw_key: str | None) -> tuple[genai.Client, bool]:
    """Vertex AI(서비스 계정) 우선 순위로 Gemini 클라이언트를 만든다.

    Returns (client, is_vertex_ai). 아무 인증 수단도 없으면 ValueError.
    """
    key = raw_key.strip() if raw_key else None

    # 분기 1: 키 값 자체가 서비스 계정 JSON 문자열인 경우
    # (Hugging Face Spaces/Docker처럼 파일 마운트 없이 문자열 시크릿만 줄 수 있는 배포 환경 지원)
    if key and key.startswith("{"):
        try:
            credentials_info = json.loads(key)
        except json.JSONDecodeError:
            credentials_info = None
        if credentials_info is not None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8")
            temp_file.write(key)
            temp_file.close()
            os.chmod(temp_file.name, 0o600)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file.name
            project_id = credentials_info.get("project_id", DEFAULT_PROJECT_ID)
            print(f"[Gemini] 서비스 계정 JSON 환경 변수 로드 완료 (Project: {project_id})")
            return _vertex_client(project_id), True

    # 분기 2: 키 값이 존재하는 .json 파일 경로인 경우
    if key and key.endswith(".json") and os.path.exists(key):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(key)
        try:
            with open(key, encoding="utf-8") as f:
                project_id = json.load(f).get("project_id", DEFAULT_PROJECT_ID)
        except Exception:
            project_id = DEFAULT_PROJECT_ID
        print(f"[Gemini] 서비스 계정 JSON 파일 로드 완료: {key}")
        return _vertex_client(project_id), True

    # 분기 3: 프로젝트 루트의 기본 서비스 계정 파일 (raw_key 유무와 무관하게 항상 우선 시도)
    local_path = PROJECT_ROOT / DEFAULT_SERVICE_ACCOUNT_FILE
    if local_path.exists():
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(local_path)
        try:
            with open(local_path, encoding="utf-8") as f:
                project_id = json.load(f).get("project_id", DEFAULT_PROJECT_ID)
        except Exception:
            project_id = DEFAULT_PROJECT_ID
        print(f"[Gemini] 서비스 계정 JSON 파일 로드 완료: {local_path}")
        return _vertex_client(project_id), True

    # 분기 4: OS 환경에 이미 GOOGLE_APPLICATION_CREDENTIALS가 설정된 경우
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        project_id = os.getenv("GCP_PROJECT_ID", DEFAULT_PROJECT_ID)
        print("[Gemini] 시스템 GOOGLE_APPLICATION_CREDENTIALS 기반 로드 완료")
        return _vertex_client(project_id), True

    # 분기 5: 평문 API Key (fallback)
    if key:
        print("[Gemini] 일반 API Key 기반 AI Studio 클라이언트 로드 완료")
        return genai.Client(api_key=key), False

    raise ValueError("Gemini 인증 수단이 없습니다 (GEMINI_API_KEY 또는 서비스 계정 JSON 필요).")


def embedding_model_name(is_vertex_ai: bool, model: str) -> str:
    return model if is_vertex_ai else f"models/{model}"


def has_gemini_auth(raw_key: str | None) -> bool:
    """build_gemini_client와 동일한 우선순위로 인증 가능 여부만 판단 (클라이언트 생성/부작용 없음)."""
    key = raw_key.strip() if raw_key else None
    if key and key.startswith("{"):
        try:
            json.loads(key)
            return True
        except json.JSONDecodeError:
            pass
    if key and key.endswith(".json") and os.path.exists(key):
        return True
    if (PROJECT_ROOT / DEFAULT_SERVICE_ACCOUNT_FILE).exists():
        return True
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        return True
    return bool(key)
