#!/bin/zsh
# 현재 파일이 있는 위치로 이동
cd "$(dirname "$0")"

echo "🚀 안티그래비티 검수 웹앱을 시작합니다..."

# 가상환경이 없으면 생성
if [ ! -d "venv" ]; then
    echo "📦 가상환경 생성 중..."
    python3 -m venv venv
fi

# 가상환경 활성화 및 라이브러리 설치
source venv/bin/activate
pip install -r requirements.txt

# 브라우저 실행
open "http://127.0.0.1:8000"

# 서버 실행
PYTHONPATH=src python3 -m translation_web_app.main
