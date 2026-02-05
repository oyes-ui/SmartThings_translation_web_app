@echo off
setlocal
echo [Antigravity] 환경 체크 중...

:: 1. 파이썬 설치 여부 확인
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] 시스템에 파이썬이 설치되어 있지 않습니다.
    echo [!] 포터블 파이썬 환경을 구성하거나 설치 페이지를 엽니다.
    start https://www.python.org/downloads/
    echo 파이썬 설치 후 다시 실행해 주세요. (3.10 이상 추천)
    pause
    exit
)

:: 2. 가상환경 구성 (있으면 건너뜀)
if not exist "venv" (
    echo [+] 가상환경을 생성하고 라이브러리를 설치합니다...
    python -m venv venv
    call venv\Scripts\activate
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate
)

:: 3. 앱 실행
start http://127.0.0.1:8000
python main.py

pause