# 반도체 업황 기반 SK하이닉스 주가 예측 — 2-Stage 파이프라인

2단계 XGBoost 파이프라인으로 **반도체 업황 YoY%를 6개월 선행 예측(Stage 1)** 하고,
그 결과를 피처로 활용해 **SK하이닉스 6개월 주가 수익률을 예측(Stage 2)** 합니다.

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

평가 환경: Tune 86분기 / Hold-out 20분기 (2021-01-28 ~ 2025-10-23)

| 지표 | 베이스라인 | 최종 모델 (Dynamic Weight) |
|------|-----------|--------------------------|
| CV RMSE (5-fold) | 20.882% | **18.974%** (개선 ✅) |
| CV DirAcc | 80.0% (Bull 85.0% / Bear 50.0%) | **90.0%** (Bull 88.3% / Bear 100.0%) (개선 ✅) |
| CV AsymLoss | 21.099 | **19.320** (개선 ✅) |
| CV IC (Spearman) | +0.440 | **+0.600** (개선 ✅) |
| Hold-out RMSE | 68.946% | **62.389%** (개선 ✅) |
| Hold-out DirAcc (최근 20분기) | 55.0% (Bull 66.7% / Bear 37.5%) | **60.0%** (Bull 91.7% / ⚠️ Bear 12.5%) (개선 ✅) |
| Hold-out IC | -0.218 | **+0.033** (개선 ✅) |
| 최적 recency_scale | — | 0.4087 |
| 최종 선택 피처 수 | 25개 | 25개 |

> ⚠️ **Hold-out Bear DirAcc**: Dynamic Weight은 CV에서 Bear DirAcc를 50.0% → 100.0%로 개선했으나,
> Hold-out에서는 37.5% → 12.5%로 악화됨. recency 가중치가 "최근 Bull 우세 패턴"을 강화하는 특성에서 비롯된 트레이드오프라고 판단.

> **AsymLoss**: Bear 국면(수익률 ≤ 0) 오예측에 ×3.0 페널티를 부여한 커스텀 손실함수.

---

## 평가 설계 원칙 (Stage 2)

### 왜 RMSE가 아닌 DirAcc를 최적화 기준으로 삼았는가

SK하이닉스 6개월 수익률은 분기에 따라 -150% ~ +180%에 달하는 극단적인 변동폭을 보인다.
Hold-out 구간 타겟의 표준편차 자체가 약 60%p를 넘는 상황에서, RMSE를 직접 최소화하는 접근은 구조적으로 한계가 있다고 판단했다.
특히 2025년의 AI 반도체 수요 급증으로 인한 +150%대 수익률은 어떤 과거 패턴으로도 정량 예측이 불가능한 외생적 충격이다.

실질적인 투자 의사결정 관점에서도 "정확히 몇 % 오르는가"보다 **"오를 것인가 내릴 것인가"** 가 우선적인 정보이다. 
따라서 본 프로젝트는 DirAcc를 핵심 성능 지표로, 방향 오예측에 대한 비대칭 페널티(Asymmetric Loss)를 학습 목적함수로 채택했다. 
특히 하락(Bear) 오예측에 가장 큰 페널티(×3.0)를 부여했는데, 
이는 "상승을 놓치는 기회비용"보다 "하락을 못 보고 매수해 발생하는 원금 손실"이 더 치명적이기 때문이다.

### 왜 Hold-out 구간을 20분기로 설정했는가

| TEST_EVAL_SIZE | 기간 | Bull | Bear | 비고 |
|---|---|---|---|---|
| 12분기 | 2023-01 ~ 2025-10 | 11 | 1 | Bear 표본 1개 → 평가 지표가 0%/100%로만 산출, 통계적으로 무의미 |
| **20분기** | **2021-01 ~ 2025-10** | **12** | **8** | **Bull:Bear ≈ 60:40, 가장 균형적** ✅ |
| 24분기 | 2020-01 ~ 2025-10 | 15 | 9 | 학습 데이터 추가 축소, COVID 구간 포함 |

12분기 설정에서는 Bear 표본이 단 1개뿐이라, Bear DirAcc가 0% 또는 100%만 나온다는 한계가 있다. 
따라서 Bull/Bear 분기의 비율이 가장 균형적이면서 학습 데이터(86분기)를 충분히 확보할 수 있는 **20분기**를 최종 평가 기준으로 채택했다.

---

## 디렉토리 구조

```
conference/
├── wsts_historical.xlsx        WSTS 원본 데이터
├── requirements.txt            패키지 버전 고정 (xgboost==3.2.0 등)
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
    ├── s5_tune.py              Step 5: XGBoost Optuna 튜닝 (USE_DYNAMIC_WEIGHTS 플래그)
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
pip install -r requirements.txt
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

#### Step 5 — XGBoost Optuna 튜닝 + Dynamic Sample Weight (`s5_tune.py`)

- **Phase A**: RMSE 최소화 → `skh_xgb_tuned.pkl`
- **Phase B**: AsymLoss (Bear 오예측 ×3.0 페널티) 최소화 → `skh_xgb_final.pkl`
- CV 구조: TimeSeriesSplit 5-fold (test_size=4분기, min_train=20분기)

> **Dynamic Sample Weight** — 베이스라인 대비 최종 모델의 핵심 개선
>
> 베이스라인은 Bear 여부에 따른 고정 가중치만 사용한다(`Bull=1.0 / Bear=2.0`). 최종 모델은 여기에 시간 가중치(Recency Weight)를 결합해, 최근 분기일수록 더 높은 가중치로 학습하도록 설계했다. 
> Hold-out 구간(2021~2025)이 AI 반도체 수요 급증으로 학습 구간 전체의 패턴과 분포가 달라지는 것(distribution shift)에 대응하기 위함이다.
>
> ```python
> def dynamic_weights(y, recency_scale):
>     recency_w = np.exp(np.linspace(0, recency_scale, len(y)))
>     recency_w = recency_w / recency_w.mean()          # 평균 1.0 정규화
>     bear_w    = np.where(y > 0, 1.0, BEAR_SAMPLE_W)  # 기존 Bear 가중치
>     return recency_w * bear_w
> ```
>
> `recency_scale`은 Optuna로 XGBoost 하이퍼파라미터와 함께 공동 탐색한다 (탐색 범위: 0.0~0.5, 최적값: 0.4087). `exp(0.4087) ≈ 1.50`으로, 가장 오래된 샘플 대비 최근 샘플의 가중치가 약 1.5배 상향된다. 
> `USE_DYNAMIC_WEIGHTS` 플래그(`True`/`False`)로 베이스라인과 최종 모델을 전환할 수 있다.

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
| `TEST_EVAL_SIZE` | `20` | Hold-out 분기 수 (5년, Bull/Bear 균형 기준) |
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
│   └── skh_xgb_final.pkl           Step 5 AsymLoss + Dynamic Weight 최종 모델 ★
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