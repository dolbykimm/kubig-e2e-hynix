# 🚀 DEPLOY STATUS — S3 업로드 · Docker Hub · Railway 준비

> 작성일: 2026-06-05
> 작업자: 석우
> 이미지: `dolbykimm/bapo-dashboard:latest`

---

## 1. S3 업로드 결과 — ✅ 5/5 성공

버킷: **`kubig-e2e-hynix-models`** (S3 key = 로컬 상대경로)

| 파일 | 크기 | 결과 |
|------|------|------|
| `stage1/outputs/models/best_xgboost_final.pkl` | 0.74 MB | ✅ 성공 |
| `stage1/outputs/data/features_dataset.csv` | 1.01 MB | ✅ 성공 |
| `stage2/outputs/models/skh_xgb_final.pkl` | 0.18 MB | ✅ 성공 |
| `stage2/outputs/data/stage2_features.csv` | 0.04 MB | ✅ 성공 |
| `stage2/outputs/data/stage1_predictions.csv` | 0.01 MB | ✅ 성공 |

> 참고: 로컬에 `boto3`가 없어 1회 실패 → `pip install boto3` 후 재실행하여 전부 성공.

---

## 2. Docker 빌드 결과 — ✅ 성공

- **Dockerfile 포트 수정 반영**: `CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]`
  - Railway가 주입하는 `$PORT`를 우선 사용, 없으면 8501 폴백.
- `docker compose build` → 성공 (이미지 `8_1_bapo_copy-dashboard:latest`)
- `libgomp1` + `stage1/requirements.txt`(streamlit·boto3 포함) 설치 정상.

---

## 3. Docker Hub Push 결과 — ✅ 성공

| 항목 | 값 |
|------|-----|
| Docker Hub 계정 | `dolbykimm` |
| 로컬 이미지 | `8_1_bapo_copy-dashboard:latest` |
| 태그 | `dolbykimm/bapo-dashboard:latest` |
| **전체 경로** | **`docker.io/dolbykimm/bapo-dashboard:latest`** |
| digest | `sha256:60da15004510e9b17d131c92d6a7a47fa5c1c1d1924f1435473d0c3078caf863` |

Pull 명령:
```bash
docker pull dolbykimm/bapo-dashboard:latest
```

---

## 4. Railway 배포 — 석우가 직접 해야 할 남은 작업

### (A) Railway 프로젝트 생성 & 이미지 연결
1. https://railway.app 로그인 → **New Project**
2. **Deploy from Docker Image** 선택
3. 이미지 입력: `dolbykimm/bapo-dashboard:latest`
   - (대안) **Deploy from GitHub repo**로 저장소를 연결하면 Railway가 `Dockerfile`을 직접 빌드함. 이 경우 위 push 이미지는 백업용.

### (B) 환경변수 4개 설정 (Railway → 서비스 → **Variables**)
| 변수명 | 값 | 비고 |
|--------|-----|------|
| `FRED_API_KEY` | `611878a66228a152fc523aeefc78bd67` | 파이프라인 재실행 안 하면 사실상 미사용 |
| `AWS_ACCESS_KEY_ID` | (본인 AWS 키) | S3 다운로드용 |
| `AWS_SECRET_ACCESS_KEY` | (본인 AWS 시크릿) | S3 다운로드용 |
| `S3_BUCKET_NAME` | `kubig-e2e-hynix-models` | 산출물 버킷 |

> ⚠️ `.env` 파일은 Railway에 올라가지 않습니다(gitignore). **반드시 Railway Variables에 직접 입력**해야 앱이 S3에서 모델을 받아옵니다. 미설정 시 "S3_BUCKET_NAME 미설정" 화면만 표시됨.

### (C) 포트 설정
- Dockerfile이 `$PORT`를 자동으로 받도록 수정됨 → **Railway가 주입하는 포트를 그대로 사용**하므로 보통 추가 설정 불필요.
- 만약 Railway가 포트를 자동 감지하지 못하면, Settings → Networking에서 **포트 8501** 또는 Railway 제공 도메인의 포트 노출을 확인.

### (D) 배포 후 점검
- 첫 기동 시 Streamlit 부팅 + S3 다운로드로 수십 초 소요 → 헬스체크 타임아웃 여유 있게.
- 배포 도메인 접속 → 사이드바 **E2E / Stage1 / Stage2** 정상 표시되면 성공.

---

## 5. 보안 메모 (권장 조치)
- 이번 작업 중 노출된 자격증명은 **재발급(rotate)** 권장:
  - **AWS 액세스 키**(`AKIA...`): IAM에서 비활성화 → 신규 발급
  - **Docker Hub Access Token**(`dckr_pat_...`): Docker Hub → Security에서 revoke 후 재발급
- `.env`는 gitignore됨(안전), `.env.example`은 빈 템플릿으로 복원 완료.

---

## ✅ 완료 요약
| 단계 | 상태 |
|------|------|
| S3 업로드 (5개) | ✅ |
| Dockerfile 포트 수정 | ✅ |
| Docker 재빌드 | ✅ |
| Docker Hub push | ✅ |
| **남은 일: Railway 연결 + 환경변수 4개 입력** | ⏳ 석우 직접 |
