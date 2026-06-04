# ✅ FINAL STATUS — E2E 파이프라인 & 배포 준비 현황

> 작성일: 2026-06-05
> 작업: FRED 키 분리 → 파이프라인 실행 → S3 업로드 스크립트 → .gitignore → Docker 로컬 테스트

---

## 1. 완료된 작업 목록

| STEP | 작업 | 상태 |
|------|------|------|
| 1 | FRED API 키 환경변수 분리 (`os.getenv`) | ✅ 완료 |
| 2-1 | Stage 1 파이프라인 실행 | ✅ 완료 (exit 0) |
| 2-2 | Stage 2 파이프라인 실행 | ✅ 완료 (exit 0) |
| 3-1 | `upload_to_s3.py` 작성 | ✅ 완료 |
| 3-2 | S3 환경변수 확인 | ⚠️ 미설정 → 업로드 **수동 실행 필요** |
| 4 | `.gitignore` 보완 | ✅ 완료 |
| 5 | Docker 로컬 빌드 + 기동 | ✅ 완료 (컨테이너 Up) |
| 6 | 본 문서 작성 | ✅ 완료 |

### STEP 1 상세
- `stage1/config.py`, `stage2/config.py`의 하드코딩된 FRED 키 제거 → `FRED_API_KEY = os.getenv("FRED_API_KEY")`.
- 파이프라인 실행 시에는 환경변수로 키를 주입해 정상 동작 확인.

---

## 2. 생성된 파일 목록

### 신규 작성/수정한 코드·설정
| 파일 | 설명 |
|------|------|
| `app.py` | Streamlit 대시보드 (이전 세션 작성) |
| `Dockerfile`, `docker-compose.yml`, `.env.example` | 배포 구성 (이전 세션) |
| `upload_to_s3.py` | **신규** — 산출물 5개 S3 업로드 스크립트 |
| `stage1/config.py`, `stage2/config.py` | FRED 키 환경변수화 |
| `stage1/requirements.txt` | `boto3>=1.34`, `streamlit>=1.30` 추가 |
| `.gitignore` | `*.pkl`, `*.csv`, `*.pyc`, `stage1/outputs/`, `stage2/outputs/` 보강 |
| `.env` | 빈 템플릿 (Docker 기동 테스트용, 값 미입력) |

### 파이프라인 산출물 (S3 업로드 대상 5개 ★)
| 파일 | 크기 |
|------|------|
| ★ `stage1/outputs/models/best_xgboost_final.pkl` | 757 KB |
| ★ `stage1/outputs/data/features_dataset.csv` | 1039 KB |
| ★ `stage2/outputs/models/skh_xgb_final.pkl` | 189 KB |
| ★ `stage2/outputs/data/stage2_features.csv` | 46 KB |
| ★ `stage2/outputs/data/stage1_predictions.csv` | 5.3 KB |
| (그 외) merged_dataset.csv, figures/*.png, metrics/*.csv 등 | — |

### 모델 성능 요약 (CV 평균)
- **Stage 1** (반도체 매출 YoY): 방향정확도 **93.3%** (Bear 90.5%), RMSE 5.79, AsymLoss 5.76
- **Stage 2** (SK하이닉스 6M 수익률): 방향정확도 **80.0%** (Bear 66.7%), RMSE 15.92, IC 0.48
- ⚠️ 참고: 두 모델 모두 Hold-out 구간(2023~2025) RMSE가 CV 대비 크게 상승 — 최근 변동성 확대 구간에서 오차가 커짐. 발표 시 한계점으로 언급 권장.

---

## 3. 남은 작업

### 🔴 필수 — S3 업로드 수동 실행
S3 환경변수(`S3_BUCKET_NAME`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)가 미설정 상태라 업로드를 건너뛰었습니다. 아래로 수동 실행하세요.

```bash
# .env에 실제 값 입력 후
$env:S3_BUCKET_NAME="your-bucket"
$env:AWS_ACCESS_KEY_ID="..."
$env:AWS_SECRET_ACCESS_KEY="..."
python upload_to_s3.py
```
산출물 5개 존재 여부 확인 → 업로드 → 파일별 성공/실패를 출력합니다.

### 🟡 권장
- `.env`에 실제 값 입력 (현재는 빈 템플릿). **`.env`는 커밋 금지** (`.gitignore`에 포함됨).
- FRED 키도 `.env`로 주입 (config는 이미 `os.getenv` 사용).

---

## 4. Docker 로컬 테스트 결과

- `docker compose build` → **성공** (이미지 `8_1_bapo_copy-dashboard:latest`)
  - `libgomp1` 설치, `stage1/requirements.txt` 설치(streamlit·boto3 포함), 소스/엑셀 복사 정상.
- `docker compose up -d` → **컨테이너 기동 성공**
  - `docker compose ps`: `8_1_bapo_copy-dashboard-1` **Up**, 포트 `0.0.0.0:8501->8501`
  - Streamlit 로그: `Uvicorn server started on 0.0.0.0:8501`, `http://localhost:8501`
  - 현재 S3 env가 비어 있어 앱은 **"S3_BUCKET_NAME 미설정" 가드 화면**을 표시 (의도된 정상 동작). 실제 데이터 화면을 보려면 `.env`에 값 입력 후 재기동 필요.

```bash
# 컨테이너 종료/재기동
docker compose down
docker compose up -d
```

---

## 5. Railway 배포를 위한 다음 단계

### (A) Docker Hub에 이미지 push
```bash
docker login
docker tag 8_1_bapo_copy-dashboard:latest <DOCKERHUB_USER>/bapo-dashboard:latest
docker push <DOCKERHUB_USER>/bapo-dashboard:latest
```
> 또는 Railway가 GitHub 저장소의 `Dockerfile`을 직접 빌드하게 할 수도 있음(이미지 push 생략 가능).

### (B) Railway 프로젝트 연결
1. Railway → **New Project**
2. 방법 ①: **Deploy from GitHub repo** (저장소 연결 → Dockerfile 자동 감지)
   방법 ②: **Deploy from Docker Image** (위에서 push한 이미지 지정)
3. 포트: Railway는 `$PORT`를 주입하므로, 실행 명령을 `--server.port=$PORT`로 바꾸거나 Railway 설정에서 8501 노출 확인 필요.

### (C) 환경변수 설정 (Railway → Variables)
| 변수 | 값 |
|------|-----|
| `FRED_API_KEY` | FRED 발급 키 |
| `AWS_ACCESS_KEY_ID` | AWS 키 |
| `AWS_SECRET_ACCESS_KEY` | AWS 시크릿 |
| `S3_BUCKET_NAME` | 산출물 업로드한 버킷명 |

### (D) 사전 조건
- 배포 전 **STEP 3 S3 업로드를 반드시 먼저 실행**해야 함. 안 그러면 컨테이너가 기동돼도 "모델 학습이 필요합니다" 화면만 표시됨.

### ⚠️ Railway 배포 시 점검 포인트
- **포트**: Dockerfile은 8501 고정. Railway의 `$PORT` 동적 포트와 맞추려면 CMD 수정 또는 Railway 포트 설정 필요.
- **헬스체크**: Streamlit 기동까지 수십 초 소요 — Railway 헬스체크 타임아웃 여유 설정.
- **이미지 크기/빌드 시간**: xgboost·streamlit 포함으로 빌드가 1~2분 소요됨.
