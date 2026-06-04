# Docker 구성 문서

> 생성일: 2026-06-05
> Streamlit 대시보드(`app.py`) 컨테이너화 구성. 모델 산출물(pkl/csv)은 이미지에 포함하지 않고 런타임에 S3에서 다운로드.

## 설계 요약
- **베이스 이미지**: `python:3.11-slim`
- **시스템 패키지**: `libgomp1` (xgboost OpenMP 런타임)
- **레이어 캐싱**: `stage1/requirements.txt` 먼저 복사·설치 후 소스 복사
- **공통 의존성**: stage1/stage2 모두 `stage1/requirements.txt` 사용
- **데이터**: `wsts_historical.xlsx`는 이미지에 포함, 모델 산출물은 S3에서 런타임 다운로드
- **포트**: 8501 (Streamlit)
- **볼륨**: 없음

---

## Dockerfile

```dockerfile
# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────
# Streamlit 대시보드용 이미지
# 모델 산출물(pkl/csv)은 이미지에 포함하지 않고 런타임에 S3에서 받음
# ─────────────────────────────────────────────────────────────

# 가볍고 호환성 좋은 공식 슬림 이미지 사용
FROM python:3.11-slim

# xgboost 등 OpenMP 의존 라이브러리 구동에 필요한 libgomp1 설치
# slim 이미지에는 빠져 있어 미설치 시 import 단계에서 OSError 발생
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── 레이어 캐싱 최적화 ──
# 의존성 목록만 먼저 복사·설치 → 소스만 바뀌면 pip 설치 캐시 재사용
# stage1/requirements.txt를 stage1·stage2 공통 의존성으로 사용
COPY stage1/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ── 소스 복사 ──
# 의존성 설치 이후 단계라 소스 변경이 위 캐시를 무효화하지 않음
COPY stage1/ ./stage1/
COPY stage2/ ./stage2/
COPY app.py ./app.py

# WSTS 원본 데이터는 이미지에 포함 (S3 다운로드 대상 아님)
COPY wsts_historical.xlsx ./wsts_historical.xlsx

# Streamlit 기본 포트
EXPOSE 8501

# 0.0.0.0 바인딩으로 컨테이너 외부에서 접근 가능하게 실행
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## docker-compose.yml

```yaml
# ─────────────────────────────────────────────────────────────
# 대시보드 단일 서비스 구성
# 모델 산출물은 런타임에 S3에서 받으므로 영속 볼륨 불필요
# ─────────────────────────────────────────────────────────────
services:
  dashboard:
    # 현재 디렉터리의 Dockerfile로 이미지 빌드
    build:
      context: .
      dockerfile: Dockerfile
    # 호스트 8501 → 컨테이너 8501 (Streamlit)
    ports:
      - "8501:8501"
    # .env 파일의 키들을 컨테이너 환경변수로 주입
    # (FRED API 키 + S3 접근 자격증명/버킷명)
    env_file:
      - .env
    # env_file로 들어온 값을 명시적으로 매핑
    # 값을 비워 두면 .env / 호스트 환경의 동일 키 값을 그대로 전달
    environment:
      FRED_API_KEY: ${FRED_API_KEY}
      AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID}
      AWS_SECRET_ACCESS_KEY: ${AWS_SECRET_ACCESS_KEY}
      S3_BUCKET_NAME: ${S3_BUCKET_NAME}
    # 비정상 종료 시 자동 재시작
    restart: unless-stopped
    # 볼륨 마운트 없음: 산출물은 S3에서 받고 이미지 자체로 완결
```

---

## .env.example

```dotenv
# ─────────────────────────────────────────────────────────────
# 환경변수 템플릿
# 이 파일을 .env로 복사한 뒤 실제 값을 채워 사용하세요.
#   cp .env.example .env
# .env는 자격증명을 포함하므로 절대 커밋하지 마세요(.gitignore 권장).
# ─────────────────────────────────────────────────────────────

# FRED(경제지표) API 키
FRED_API_KEY=

# S3 접근용 AWS 자격증명 (모델 산출물 다운로드)
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=

# 모델 산출물(pkl/csv)이 저장된 S3 버킷 이름
S3_BUCKET_NAME=
```

---

## 사용 방법

```bash
# 1) 환경변수 파일 준비
cp .env.example .env      # 이후 .env에 실제 값 입력

# 2) 빌드 & 실행
docker compose up --build

# 3) 접속
# http://localhost:8501
```

## 참고 / 주의
- **`app.py` 미존재**: 현재 프로젝트 루트에 `app.py`가 아직 없습니다. 빌드 전 생성이 필요합니다.
- **`boto3` 누락**: S3 다운로드를 하려면 `stage1/requirements.txt`에 `boto3`(또는 `awscli`) 추가가 필요합니다. 현재 목록에는 포함되어 있지 않습니다.
- **`.env` 커밋 금지**: 자격증명 노출 방지를 위해 `.gitignore`에 `.env` 추가를 권장합니다.
