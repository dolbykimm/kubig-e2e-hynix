# 반도체 업황 기반 SK하이닉스 주가 예측 — 2-Stage 파이프라인

2단계 XGBoost 파이프라인으로 **반도체 업황 YoY%를 6개월 선행 예측(Stage 1)**하고,
그 결과를 피처로 활용해 **SK하이닉스 6개월 주가 수익률을 예측(Stage 2)**합니다.

---

## 결과 요약

### Stage 1 — 반도체 업황 YoY% 6개월 선행 예측 (월간)

| 지표 | 값 |
|------|----|
| CV RMSE (5-fold) | **6.06** |
| CV DirAcc | **95.0%** (Bull 100.0% / Bear 87.1%) |
| CV AsymLoss | **6.01** |
| Hold-out RMSE (최근 24개월) | 17.41 |
| Hold-out DirAcc | 95.8% |
| 최종 선택 피처 수 | 30개 / 165개 |

> **AsymLoss**: Bear 국면(YoY% ≤ 0) 오예측에 ×3.0 페널티를 부여한 커스텀 손실함수.

### Stage 2 — SK하이닉스 6개월 주가 수익률 예측 (분기)

| 지표 | 값 |
|------|----|
| CV RMSE (5-fold) | **16.77** |
| CV DirAcc | **70.0%** (Bull 66.7% / Bear 71.7%) |
| CV AsymLoss | **17.48** |
| CV IC (Spearman) | 0.12 |
| Hold-out DirAcc (최근 12분기) | 58.3% |
| 최종 선택 피처 수 | 25개 |

---

## 디렉토리 구조

```
conference/
├── wsts_historical.xlsx        WSTS 원본 데이터
│
├── stage1/                     Stage 1 — 반도체 업황 YoY% 예측
│   ├── config.py               공통 경로·상수·하이퍼파라미터
│   ├── pipeline.py             전체 파이프라인 마스터 실행기
│   ├── s1_data.py              Step 1: 데이터 수집 + 피처 엔지니어링
│   ├── s2_tune.py              Step 2: Optuna RMSE 하이퍼파라미터 튜닝
│   ├── s3_select.py            Step 3: 다중공선성 제거 + RFE 피처 선택
│   ├── s4_optimize.py          Step 4: AsymLoss Bear 최적화
│   ├── s5_evaluate.py          Step 5: 최종 평가 + 시각화
│   ├── requirements.txt
│   └── outputs/                실행 후 자동 생성
│       ├── data/               CSV 데이터 파일
│       ├── models/             학습된 모델 pkl 파일
│       ├── figures/            시각화 png 파일
│       └── metrics/            평가 지표 CSV 파일
│
└── stage2/                     Stage 2 — SK하이닉스 주가 수익률 예측
    ├── config.py               공통 경로·상수·하이퍼파라미터
    ├── pipeline.py             전체 파이프라인 마스터 실행기
    ├── s1_dates.py             Step 1: 분기 날짜 생성 + 타겟 수익률 산출
    ├── s2_data.py              Step 2: A~E 피처 수집 (yfinance · FRED · WSTS)
    ├── s3_stage1_feat.py       Step 3: Stage 1 Expanding Window Pseudo 예측
    ├── s4_features.py          Step 4: 피처 엔지니어링 + 피처 선택
    ├── s5_tune.py              Step 5: XGBoost Optuna 튜닝
    ├── s6_evaluate.py          Step 6: 최종 평가 + 시각화
    └── outputs/                실행 후 자동 생성
        ├── data/               CSV 데이터 파일
        ├── models/             학습된 모델 pkl 파일
        ├── figures/            시각화 png 파일
        └── metrics/            평가 지표 CSV 파일
```

---

## 환경 설정

### 1. Python 버전

Python **3.9 이상**을 권장합니다.

```bash
python --version  # 3.9+
```

### 2. 패키지 설치

```bash
pip install -r stage1/requirements.txt
```

### 3. FRED API 키 설정

```bash
# Mac / Linux
export FRED_API_KEY=your_api_key_here

# Windows (PowerShell)
$env:FRED_API_KEY="your_api_key_here"

# Windows (cmd)
set FRED_API_KEY=your_api_key_here
```

또는 `stage1/config.py`, `stage2/config.py`의 `FRED_API_KEY` 값을 직접 수정합니다.

---

## 실행 방법

**Stage 1을 먼저 실행해야 합니다.** Stage 2는 Stage 1의 피처셋과 학습 모델을 입력으로 사용합니다.

### Stage 1 — 전체 파이프라인 한 번에 실행 (권장)

```bash
python stage1/pipeline.py
```

예상 소요시간: **약 25~35분** (Optuna 2회 × 50 trial)

### Stage 1 — 단계별 개별 실행

```bash
python stage1/s1_data.py      # Step 1: 데이터 수집 + 피처 생성     (~2분)
python stage1/s2_tune.py      # Step 2: RMSE 기준 Optuna 튜닝       (~10분)
python stage1/s3_select.py    # Step 3: 피처 선택 (RFE)             (~5분)
python stage1/s4_optimize.py  # Step 4: AsymLoss Bear 최적화        (~10분)
python stage1/s5_evaluate.py  # Step 5: 최종 평가 + 그래프 저장      (~1분)
```

### Stage 2 — 전체 파이프라인 한 번에 실행

```bash
python stage2/pipeline.py
```

예상 소요시간: **약 30~50분** (expanding window 재훈련 ~100회 + Optuna 2회 × 50 trial)

### Stage 2 — 단계별 개별 실행

```bash
python stage2/s1_dates.py       # Step 1: 분기 날짜 생성 + 타겟 수익률 산출  (~1분)
python stage2/s2_data.py        # Step 2: A~E 피처 수집                     (~5분)
python stage2/s3_stage1_feat.py # Step 3: Stage 1 Expanding Window 예측     (~20분)
python stage2/s4_features.py    # Step 4: 피처 엔지니어링 + 피처 선택        (~3분)
python stage2/s5_tune.py        # Step 5: XGBoost Optuna 튜닝               (~20분)
python stage2/s6_evaluate.py    # Step 6: 최종 평가 + 그래프 저장            (~1분)
```

> **주의**: 각 스텝은 이전 스텝의 출력 파일에 의존합니다. 반드시 순서대로 실행하세요.

---

## 파이프라인 단계별 설명

### Stage 1

#### Step 1 — 데이터 수집 + 피처 엔지니어링 (`s1_data.py`)

| 데이터 소스 | 내용 |
|------------|------|
| `wsts_historical.xlsx` | WSTS 월별 반도체 출하량 (Americas / Europe / Japan / Asia Pacific / Worldwide) |
| FRED API | 산업생산지수, 장단기 금리차(T10Y2Y·T10Y3M), 소비자심리지수, 신규수주, 제조업 고용, 연방기금금리, 재고/매출 비율 등 |
| yfinance | 필라델피아 반도체지수(SOX), NVDA, TSM, ASML, 삼성전자, SK하이닉스 주가 |

생성되는 피처 카테고리:

- **YoY% 기본값** / **Lag 피처** (lag6, lag12) / **이동평균** (ma3, ma6, ma12)
- **변동성** (vol3, vol6) / **모멘텀** / **가속도** / **사이클 위치 Percentile**
- **Bear 선행 지표**: T10Y3M 역전 여부·지속 기간, 재고/매출 비율(ISRATIO), 연방기금금리 변화

타겟 변수:

```
TARGET_Worldwide_YoY_T6   = Worldwide 매출 YoY%  (shift(-6), 6개월 후 예측)
TARGET_Asia_Pacific_YoY_T6 = Asia Pacific 매출 YoY% (보조 타겟)
```

#### Step 2 — Optuna 하이퍼파라미터 튜닝 (`s2_tune.py`)

- 목적함수: TimeSeriesSplit 5-fold CV **RMSE 최소화**
- Sampler: TPE / Pruner: MedianPruner
- 탐색 공간: `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`, `min_child_weight`

#### Step 3 — 피처 선택 (`s3_select.py`)

1. **다중공선성 제거**: |Pearson r| ≥ 0.9인 피처 쌍에서 하나 제거
2. **SHAP 중요도 계산**: 전체 피처 중요도 순위 산출
3. **RFE 커브**: n = 10, 15, 18, 20, 22, 25, 30, 35, 40, 50, 70개 피처로 CV AsymLoss 측정
4. **최적 피처 수 선택**: 가장 낮은 AsymLoss를 달성하는 피처 수 선택 (이번 실행: **30개**)

#### Step 4 — Bear 최적화 (`s4_optimize.py`)

- 목적함수: **AsymLoss** (Bear 오예측 ×3.0 페널티) 최소화
- 선택된 30개 피처 + Bear 월 `sample_weight=2.0` 적용
- 최종 모델: `stage1/outputs/models/best_xgboost_final.pkl`

#### Step 5 — 최종 평가 + 시각화 (`s5_evaluate.py`)

- Hold-out 평가 (최근 24개월: 2023-08 ~ 2025-07)
- 그래프 저장: 예측 타임라인, CV fold별 지표, Bear/Bull 개선 비교, SHAP 요약
- 지표 저장: `stage1/outputs/metrics/final_cv_metrics.csv`

---

### Stage 2

#### Step 1 — 분기 날짜 생성 + 타겟 수익률 산출 (`s1_dates.py`)

- SK하이닉스 **분기 실적발표일**: 매년 1·4·7·10월 넷째 주 목요일
- **관찰일**: 실적발표일 정확히 6개월 전 같은 요일
- 타겟: `TARGET_SKH_6M_RET = (P_earnings / P_obs − 1) × 100 (%)`

#### Step 2 — 피처 데이터 수집 (`s2_data.py`)

관찰일 기준 5개 피처 그룹 수집 (lookahead 없음):

| 그룹 | 내용 |
|------|------|
| A. SK하이닉스 기술적 지표 | 가격, 1·3·6·12개월 수익률, 변동성(60d), RSI(14), MA 괴리율, 52주 고저 위치 |
| B. 시장 센티먼트 | VIX, SOX, NVDA, TSM, ASML, Samsung, S&P500의 1·3·6개월 수익률 |
| C. WSTS 실제 역사 데이터 | Worldwide·Asia Pacific YoY%, 이동평균, 모멘텀, 사이클 위치 |
| D. FRED 거시지표 | 장단기 금리차(T10Y2Y·T10Y3M), 기준금리, 산업생산, PCE, 소비자심리 |
| E. 환율·원자재 | USD/KRW, WTI 유가의 3·6개월 수익률 |

#### Step 3 — Stage 1 Expanding Window Pseudo 예측 (`s3_stage1_feat.py`)

- 각 관찰일 기준 **expanding window**로 Stage 1 모델을 재훈련하여 lookahead 없는 OOS pseudo-prediction `v2_pred_ww_yoy` (6개월 선행 WW YoY%)를 생성
- Stage 2의 핵심 피처로 활용

#### Step 4 — 피처 엔지니어링 + 피처 선택 (`s4_features.py`)

- 달력·사이클 피처 추가: 실적 발표 분기, 반도체 슈퍼사이클 위치, 장기 추세 proxy
- `v2_pred_ww_yoy` 파생 피처: 예측 vs 현재 WSTS YoY% 괴리, Bull/Bear 신호
- **3단계 피처 선택**: NaN 비율 필터 → VIF 다중공선성 제거(임계 10) → XGBoost importance 상위 60% → RFE 최종 **25개**

#### Step 5 — XGBoost Optuna 튜닝 (`s5_tune.py`)

- **Phase A**: RMSE 최소화 → `skh_xgb_tuned.pkl`
- **Phase B**: AsymLoss (Bear 오예측 ×3.0 페널티) 최소화 → `skh_xgb_final.pkl`
- CV 구조: TimeSeriesSplit 5-fold (test_size=4분기, min_train=20분기)

#### Step 6 — 최종 평가 + 시각화 (`s6_evaluate.py`)

평가 지표: RMSE (전체·Bull·Bear), DirAcc (전체·Bull·Bear), AsymLoss, IC (Spearman)

시각화:

| 파일 | 내용 |
|------|------|
| `01_return_timeline.png` | 예측 vs 실제 수익률 전 기간 + Hold-out 확대 |
| `02_cv_metrics.png` | CV 평균 vs Hold-out 지표 바차트 |
| `03_direction_analysis.png` | Bull/Bear 방향 정확도 + 혼동행렬 |
| `04_simulation.png` | Long-only 전략 vs Buy & Hold 누적 수익률 |

---

## 주요 설정값

### Stage 1 (`stage1/config.py`)

| 상수 | 기본값 | 설명 |
|------|--------|------|
| `START_DATE` | `"1993-01-01"` | 데이터 수집 시작일 |
| `END_DATE` | `"2026-03-31"` | 데이터 수집 종료일 |
| `TEST_EVAL_SIZE` | `24` | Hold-out 개월 수 |
| `N_SPLITS` | `5` | TimeSeriesSplit fold 수 |
| `TEST_SIZE` | `12` | fold당 test 개월 수 |
| `N_TRIALS` | `50` | Optuna trial 수 |
| `BEAR_SAMPLE_W` | `2.0` | Bear 월 sample_weight |
| `W_BEAR_WRONG` | `3.0` | AsymLoss Bear 오예측 페널티 |

### Stage 2 (`stage2/config.py`)

| 상수 | 기본값 | 설명 |
|------|--------|------|
| `START_YEAR` | `2000` | 데이터 수집 시작 연도 |
| `END_YEAR` | `2026` | 데이터 수집 종료 연도 |
| `TEST_EVAL_SIZE` | `12` | Hold-out 분기 수 (3년) |
| `N_SPLITS` | `5` | TimeSeriesSplit fold 수 |
| `TEST_SIZE` | `4` | fold당 test 분기 수 |
| `N_TRIALS` | `50` | Optuna trial 수 |
| `BEAR_SAMPLE_W` | `2.0` | Bear 분기 sample_weight |
| `W_BEAR_WRONG` | `3.0` | AsymLoss Bear 오예측 페널티 |

---

## 출력 결과물

### Stage 1 (`stage1/outputs/`)

```
stage1/outputs/
├── data/
│   ├── wsts_monthly.csv            WSTS 월별 원본 (지역별 매출)
│   ├── merged_dataset.csv          병합 원본 데이터
│   └── features_dataset.csv        엔지니어링된 피처셋 (165피처 + 타겟)  ← Stage 2 입력
│
├── models/
│   ├── best_xgboost.pkl            Step 2 RMSE 최적 파라미터
│   ├── best_xgboost_selected.pkl   Step 3 선택 피처 30개로 재학습
│   └── best_xgboost_final.pkl      Step 4 AsymLoss 최적화 최종 모델 ★  ← Stage 2 입력
│
├── metrics/
│   ├── selected_features.csv       피처별 중요도·선택 여부
│   └── final_cv_metrics.csv        fold별 RMSE / DirAcc / AsymLoss
│
└── figures/
    ├── 01_prediction_timeline.png  Hold-out 구간 예측 vs 실제
    ├── 02_cv_metrics.png           CV fold별 성능 비교
    ├── 03_bear_improvement.png     Bear 국면 성능 개선 비교
    ├── rfe_curve.png               피처 수 vs CV AsymLoss 커브
    └── shap_summary.png            SHAP 피처 중요도 요약
```

### Stage 2 (`stage2/outputs/`)

```
stage2/outputs/
├── data/
│   ├── quarterly_dates.csv         분기 관찰일·실적발표일·타겟 수익률
│   ├── raw_quarterly.csv           관찰일별 원시 피처 (A~E 그룹)
│   ├── stage1_predictions.csv      Stage 1 expanding window OOS 예측값
│   └── stage2_features.csv         최종 피처셋 (25피처 + 타겟)
│
├── models/
│   ├── skh_xgb_tuned.pkl           Step 5 RMSE 최적 파라미터
│   └── skh_xgb_final.pkl           Step 5 AsymLoss 최적화 최종 모델 ★
│
├── metrics/
│   └── final_cv_metrics.csv        fold별 RMSE / DirAcc / AsymLoss / IC
│
└── figures/
    ├── 01_return_timeline.png      예측 vs 실제 수익률 전 기간
    ├── 02_cv_metrics.png           CV vs Hold-out 지표 바차트
    ├── 03_direction_analysis.png   Bull/Bear 방향 정확도 + 혼동행렬
    └── 04_simulation.png           Long-only 전략 vs Buy & Hold 시뮬레이션
```
