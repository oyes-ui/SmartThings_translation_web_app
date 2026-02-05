# 1. 경량화된 파이썬 이미지 사용
FROM python:3.10-slim

# 2. 사용자 환경 설정 (Hugging Face Spaces 권장: UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# 3. 라이브러리 설치를 위해 requirements.txt를 먼저 복사합니다.
COPY --chown=user requirements.txt .

# 4. 필요한 라이브러리를 설치합니다.
RUN pip install --no-cache-dir --user -r requirements.txt

# 5. 나머지 소스 코드와 static 폴더를 모두 복사합니다.
COPY --chown=user . .

# 6. Hugging Face Spaces는 기본적으로 7860 포트를 사용합니다.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]