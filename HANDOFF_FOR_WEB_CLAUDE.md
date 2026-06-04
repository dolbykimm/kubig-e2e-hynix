# 📨 Web Claude에게 — 현재 작업 상황 공유

> 이 문서는 로컬 Claude Code 세션에서 진행한 작업을 web Claude에게 전달하기 위한 인수인계 메모입니다.
> 날짜: 2026-06-05

---

## 1. 프로젝트 한 줄 요약

**반도체 사이클 → SK하이닉스 수익률을 예측하는 2단계(E2E) ML 파이프라인**을 Docker로 배포하고, 결과를 Streamlit 대시보드로 보여주려는 프로젝트입니다.

- **Stage 1**: 전 세계 반도체 매출 YoY(6개월 선행) 예측 — XGBoost
  - 타깃: `TARGET_Worldwide_YoY_T6`
- **Stage 2**: SK하이닉스 6개월 수익률 방향 예측 — XGBoost
  - 타깃: `TARGET_SKH_6M_RET`
  - Stage 1의 예측값(`v2_pred_ww_yoy`)을 입력 피처로 사용 (= 두 단계가 연결되는 핵심 고리)

두 모델 모두 **Bear(하락) 구간 오예측에 더 큰 페널티**를 주는 Asymmetric Loss + sample weight 전략을 씁니다.

---

## 2. 폴더 구조

```
8_1_bapo_copy/
├── app.py                      # ★ 이번 세션에서 신규 작성한 Streamlit 대시보드
├── Dockerfile                  # ★ 신규
├── docker-compose.yml          # ★ 신규
├── .env.example                # ★ 신규 (실제 값은 빈칸)
├── docker_config.md            # Docker 관련 3개 파일 내용 정리 문서
├── wsts_historical.xlsx        # 입력 원본 데이터 (이미지에 포함)
├── stage1/
│   ├── config.py               # 경로·상수·CV·Loss 가중치
│   ├── pipeline.py
│   ├── requirements.txt        # ★ boto3>=1.34, streamlit>=1.30 추가됨 (공통 의존성)
│   ├── s1_data.py ~ s5_evaluate.py
│   └── outputs/{data,models,figures,metrics}/   # 산출물 (현재 비어 있음)
└── stage2/
    ├── config.py
    ├── pipeline.py
    ├── s1_dates.py ~ s6_evaluate.py
    └── outputs/{data,models,figures,metrics}/   # 산출물 (현재 비어 있음)
```

> ⚠️ **중요**: `outputs/` 하위는 현재 전부 비어 있습니다(.gitkeep만 존재). 즉 **모델/데이터 산출물(.pkl, .csv)이 아직 생성되지 않았습니다.** 파이프라인을 실행해야 만들어집니다.

---

## 3. 이번 세션에서 한 일

### (1) Docker 구성 3종 + 예시 env 생성
- `Dockerfile`
  - 베이스 `python:3.11-slim`, `libgomp1` 설치(xgboost OpenMP 의존성)
  - `stage1/requirements.txt` 먼저 복사·설치 후 소스 복사(레이어 캐싱)
  - `wsts_historical.xlsx` 포함, 포트 8501, `streamlit run app.py ...`로 실행
- `docker-compose.yml`
  - 서비스명 `dashboard`, 포트 `8501:8501`
  - `.env`에서 `FRED_API_KEY / AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / S3_BUCKET_NAME` 로드
  - 볼륨 마운트 없음 (산출물은 런타임에 S3에서 다운로드)
- `.env.example` — 위 4개 키를 빈 값으로 템플릿화

### (2) `app.py` (Streamlit 대시보드) 신규 작성
- **앱 시작 시 S3에서 산출물 5개를 `/app`으로 다운로드** (`boto3`):
  - `stage1/outputs/models/best_xgboost_final.pkl`
  - `stage1/outputs/data/features_dataset.csv`
  - `stage2/outputs/models/skh_xgb_final.pkl`
  - `stage2/outputs/data/stage2_features.csv`
  - `stage2/outputs/data/stage1_predictions.csv`
- 버킷명은 `os.getenv("S3_BUCKET_NAME")`. 파일이 없으면 **"모델 학습이 필요합니다" 안내 후 `st.stop()`**.
- S3 미설정/접근 실패/파일 없음 4가지 상태를 분기 처리.
- 사이드바에서 **E2E 전체 / Stage 1 / Stage 2** 선택.
  - Stage 1·2: 성능 지표 KPI 카드(방향정확도 전체/Bull/Bear, RMSE, Asym Loss, Stage2는 IC 추가) + 예측 vs 실제 라인차트 (+ Stage2는 혼동행렬)
  - E2E: `Stage1 → Bridge(v2_pred_ww_yoy) → Stage2` 흐름 다이어그램 + Stage1 예측 시계열 + Stage2 피처 결합 검증 + 두 단계 성능 비교표
- 캐싱: 모델 `@st.cache_resource`, 데이터 `@st.cache_data`.
- **성능 지표는 metrics CSV를 받지 않고**, evaluate 스크립트 로직(hold-out 재학습→예측)을 재현해 **런타임 계산**.

### (3) `stage1/requirements.txt`에 의존성 추가
- `boto3>=1.34` (S3 연동)
- `streamlit>=1.30` (대시보드 실행) ← 원래 빠져 있어 컨테이너 실행이 불가했던 것을 보완

---

## 4. 주요 설계 결정 (왜 이렇게 했는지)

1. **시각화는 matplotlib이 아니라 Streamlit 네이티브 차트로 구현.**
   - 이유: `s5_evaluate.py`/`s6_evaluate.py`는 matplotlib 폰트로 `AppleGothic`을 쓰는데, Linux 컨테이너엔 이 폰트가 없어 **한글이 깨집니다.** 네이티브 차트(st.line_chart 등)는 폰트 의존이 없어 안전합니다.
2. **metrics CSV를 S3 다운로드 목록에 넣지 않음.**
   - 이유: 사용자가 지정한 다운로드 목록에 metrics가 없었고, 모델+피처 데이터만 있으면 hold-out 지표를 그대로 재계산할 수 있어 의존성을 줄였습니다.
3. **`pkl` 번들 포맷 가정**: `{"model", "feature_names", "best_params"}` — 실제 evaluate 스크립트가 이 구조로 로드하므로 동일하게 맞췄습니다.

---

## 5. 아직 안 된 것 / 다음 할 일 (web Claude가 알아야 할 열린 항목)

- [ ] **파이프라인 미실행** → S3에 올릴 산출물(.pkl, .csv)이 아직 없음. `stage1/pipeline.py`, `stage2/pipeline.py`를 돌려 산출물을 만들고 S3 버킷에 업로드해야 대시보드가 동작함.
- [ ] **S3 업로드 스크립트/절차 미정** — 산출물을 어떤 키 구조로 올릴지 합의 필요(현재 app.py는 위 4-(2)의 상대경로 그대로를 S3 key로 가정).
- [ ] **`.env` 실제 값 미입력** — `.env.example`을 복사해 `FRED_API_KEY`, AWS 자격증명, `S3_BUCKET_NAME`을 채워야 함. `.env`는 커밋 금지(.gitignore 권장).
- [ ] **FRED API 키가 config.py에 하드코딩되어 있음** (`stage1/config.py`, `stage2/config.py`) — 보안상 환경변수로 빼는 게 좋음.
- [ ] (선택) 평가 스크립트의 한글 matplotlib 폰트도 컨테이너 대응이 필요하면 별도 처리.

---

## 6. 실행 방법 (참고)

```bash
cp .env.example .env      # 값 채우기
docker compose up --build # http://localhost:8501
```

---

### web Claude에게 부탁
위 **5번 열린 항목** 위주로 이어서 도와주면 됩니다. 특히 (a) 파이프라인 실행 → 산출물 생성, (b) S3 업로드 키 구조 확정, (c) 비밀값을 env로 분리하는 부분이 다음 우선순위입니다.
