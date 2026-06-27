"""
app.py — E2E 반도체 사이클 → SK하이닉스 수익률 예측 대시보드 (Streamlit)
================================================================================
Stage 1 (반도체 출하량 YoY 예측) → Stage 2 (SK하이닉스 6개월 수익률 방향 예측)
2단계 파이프라인의 학습 결과를 발표용으로 시각화한다.
"""

import os
import pickle
import re
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

KST = timezone(timedelta(hours=9))

# ── 페이지 설정 (반드시 첫 Streamlit 호출) ──────────────────────────
st.set_page_config(
    page_title="반도체 사이클 → SK하이닉스 수익률 예측",
    page_icon="📈",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────
# 상수 정의
# ──────────────────────────────────────────────────────────────────

APP_ROOT = os.getenv("APP_ROOT", os.path.dirname(os.path.abspath(__file__)))

ARTIFACTS = [
    "stage1/outputs/models/best_xgboost_final.pkl",
    "stage1/outputs/data/features_dataset.csv",
    "stage2/outputs/models/skh_xgb_final.pkl",
    "stage2/outputs/data/stage2_features.csv",
    "stage2/outputs/data/stage1_predictions.csv",
]

W_BULL_CORRECT, W_BULL_WRONG = 1.0, 2.0
W_BEAR_CORRECT, W_BEAR_WRONG = 1.5, 3.0
BEAR_SAMPLE_W = 2.0

CLR_BLUE  = "#2a78d6"
CLR_TEAL  = "#1D9E75"
CLR_RED   = "#E24B4A"
CLR_AMBER = "#EF9F27"
CLR_GRAY  = "#888780"

BG_BLUE  = "#e6f1fb"
BG_GREEN = "#eaf3de"
BG_TEAL  = "#e1f5ee"
BG_RED   = "#fcebeb"
BG_AMBER = "#faeeda"

STAGE1 = {
    "name": "Stage 1",
    "title": "반도체 출하량 YoY 예측",
    "features_path": os.path.join(APP_ROOT, "stage1/outputs/data/features_dataset.csv"),
    "model_path":    os.path.join(APP_ROOT, "stage1/outputs/models/best_xgboost_final.pkl"),
    "target":        "TARGET_Worldwide_YoY_T6",
    "test_eval":     24,
    "value_label":   "예측 YoY",
    "freq_label":    "개월",
}
STAGE2 = {
    "name": "Stage 2",
    "title": "SK하이닉스 6개월 수익률 방향 예측",
    "features_path": os.path.join(APP_ROOT, "stage2/outputs/data/stage2_features.csv"),
    "model_path":    os.path.join(APP_ROOT, "stage2/outputs/models/skh_xgb_final.pkl"),
    "target":        "TARGET_SKH_6M_RET",
    "test_eval":     12,
    "value_label":   "예측 수익률",
    "freq_label":    "분기",
}

STAGE1_PRED_PATH = os.path.join(APP_ROOT, "stage2/outputs/data/stage1_predictions.csv")
BRIDGE_COL = "v2_pred_ww_yoy"


# ──────────────────────────────────────────────────────────────────
# 1. S3 산출물 다운로드
# ──────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="S3에서 모델/데이터 산출물을 내려받는 중...")
def download_artifacts():
    bucket = os.getenv("S3_BUCKET_NAME")
    if not bucket:
        return {"status": "no_bucket", "missing": [], "error": None}

    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
    except ImportError as e:
        return {"status": "s3_error", "missing": [], "error": f"boto3 미설치: {e}"}

    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
    except (BotoCoreError, NoCredentialsError) as e:
        return {"status": "s3_error", "missing": [], "error": f"S3 클라이언트 생성 실패: {e}"}

    missing = []
    for key in ARTIFACTS:
        local_path = os.path.join(APP_ROOT, key)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            s3.download_file(bucket, key, local_path)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey", "NoSuchBucket"):
                missing.append(key)
            else:
                return {"status": "s3_error", "missing": [], "error": str(e)}
        except (BotoCoreError, NoCredentialsError) as e:
            return {"status": "s3_error", "missing": [], "error": str(e)}

    if missing:
        return {"status": "missing", "missing": missing, "error": None}
    return {"status": "ok", "missing": [], "error": None}


def guard_artifacts():
    result = download_artifacts()
    status = result["status"]

    if status == "ok":
        return

    if status == "no_bucket":
        st.error("⚙️ 환경변수 `S3_BUCKET_NAME`이 설정되지 않았습니다.")
        st.info("`.env`에 S3 버킷명과 AWS 자격증명을 설정한 뒤 다시 실행해 주세요.")
        st.stop()

    if status == "s3_error":
        st.error("❌ S3 접근에 실패했습니다. 자격증명 또는 네트워크를 확인해 주세요.")
        st.code(str(result["error"]), language="text")
        st.stop()

    if status == "missing":
        st.error("🛠️ 모델 학습이 필요합니다.")
        st.warning(
            "S3 버킷에서 아래 산출물을 찾을 수 없습니다. "
            "Stage 1·2 파이프라인을 먼저 실행해 산출물을 업로드해 주세요."
        )
        for key in result["missing"]:
            st.markdown(f"- `{key}`")
        st.stop()


# ──────────────────────────────────────────────────────────────────
# 2. 데이터 / 모델 로딩 (캐싱)
# ──────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


# ──────────────────────────────────────────────────────────────────
# 3. 지표 계산 (hold-out 평가 재현)
# ──────────────────────────────────────────────────────────────────

def _safe_rmse(y_true, y_pred, mask):
    if not mask.any():
        return None
    err = y_true[mask] - y_pred[mask]
    return float(np.sqrt(np.mean(err ** 2)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, with_ic: bool = False) -> dict:
    bull    = y_true > 0
    bear    = ~bull
    correct = (y_true > 0) == (y_pred > 0)
    w = np.where(bull & correct,  W_BULL_CORRECT,
        np.where(bull & ~correct, W_BULL_WRONG,
        np.where(bear & correct,  W_BEAR_CORRECT, W_BEAR_WRONG)))

    metrics = {
        "rmse":      float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "rmse_bull": _safe_rmse(y_true, y_pred, bull),
        "rmse_bear": _safe_rmse(y_true, y_pred, bear),
        "dir_acc":   float(correct.mean() * 100),
        "dir_bull":  float(correct[bull].mean() * 100) if bull.any() else None,
        "dir_bear":  float(correct[bear].mean() * 100) if bear.any() else None,
        "asym_loss": float(np.sqrt((w * (y_true - y_pred) ** 2).sum() / w.sum())),
    }
    if with_ic:
        ic = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
        metrics["ic"] = float(ic) if pd.notna(ic) else None
    return metrics


@st.cache_data(show_spinner="모델 성능을 평가하는 중...")
def evaluate_stage(features_path: str, model_path: str, target: str,
                   test_eval: int, with_ic: bool = False):
    """백테스트용: hold-out 구간에서 re-train → predict → 성능 지표 반환."""
    import xgboost as xgb

    bundle  = load_model(model_path)
    model   = bundle["model"]
    feats   = bundle["feature_names"]
    params  = model.get_params()

    df = load_csv(features_path)
    use_feats = [f for f in feats if f in df.columns]
    df_clean  = df.dropna(subset=[target])      # 타겟 있는 행만 (미래 행 제외)
    X = df_clean[use_feats].ffill().fillna(0)
    y = df_clean[target]

    split = len(X) - test_eval
    X_tune, y_tune = X.iloc[:split], y.iloc[:split]
    X_ho,   y_ho   = X.iloc[split:], y.iloc[split:]

    w_tune = np.where(y_tune.values > 0, 1.0, BEAR_SAMPLE_W)
    m = xgb.XGBRegressor(**params)
    m.fit(X_tune, y_tune, sample_weight=w_tune)
    preds = m.predict(X_ho)

    metrics = compute_metrics(y_ho.values, preds, with_ic=with_ic)
    metrics["period"] = f"{y_ho.index[0].date()} ~ {y_ho.index[-1].date()}"
    metrics["n_holdout"] = len(y_ho)
    metrics["n_features"] = len(use_feats)

    out = pd.DataFrame({"실제값": y_ho.values, "예측값": preds}, index=y_ho.index)
    return metrics, out


@st.cache_data(show_spinner=False)
def get_latest_forecast(features_path: str, model_path: str) -> dict:
    """
    진짜 현재 예측: 최종 학습 모델(bundle['model'])로
    타겟 NaN 여부와 무관하게 가장 최신 피처 행을 추론한다.

    기존 evaluate_stage()는 hold-out 재학습용이므로 미래 행(타겟 NaN)을 dropna로 버린다.
    여기서는 버리지 않고 가장 최신 행을 그대로 입력해 미래 방향을 예측한다.
    """
    bundle = load_model(model_path)
    model  = bundle["model"]
    feats  = bundle["feature_names"]

    df = load_csv(features_path)
    use_feats = [f for f in feats if f in df.columns]
    X = df[use_feats].ffill().fillna(0)     # dropna 없음 — 최신 행 포함

    latest_X = X.iloc[[-1]]
    pred     = float(model.predict(latest_X)[0])
    date     = X.index[-1]
    target_date = date + pd.DateOffset(months=6)

    return {"pred": pred, "date": date, "target_date": target_date}


@st.cache_data(show_spinner="SHAP 피처 중요도 계산 중...")
def compute_shap_importance(model_path: str, features_path: str, target: str,
                            top_n: int = 10) -> pd.DataFrame:
    import shap

    bundle = load_model(model_path)
    model  = bundle["model"]
    feats  = bundle["feature_names"]

    df = load_csv(features_path)
    use_feats = [f for f in feats if f in df.columns]
    df_clean  = df.dropna(subset=[target])
    X = df_clean[use_feats].ffill().fillna(0)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    mean_abs    = np.abs(shap_values).mean(axis=0)

    s = (pd.Series(mean_abs, index=use_feats)
         .sort_values(ascending=False)
         .head(top_n))
    return s.rename("평균 |SHAP|").to_frame()


@st.cache_data(ttl=3600, show_spinner="시장 신호(yfinance) 수집 중...")
def get_market_momentum() -> dict:
    import yfinance as yf

    result = {}
    for label, ticker in [("KOSPI", "^KS11"), ("SOX", "^SOX")]:
        try:
            data  = yf.download(ticker, period="3mo", interval="1d",
                                progress=False, auto_adjust=True)
            close = np.asarray(data["Close"]).reshape(-1)
            close = close[~np.isnan(close)]
            if len(close) >= 2:
                result[label] = float(close[-1] / close[0] - 1.0) * 100
            else:
                result[label] = None
        except Exception:
            result[label] = None
    return result


# ──────────────────────────────────────────────────────────────────
# 4. UI 헬퍼
# ──────────────────────────────────────────────────────────────────

def _fmt(v, pct=False):
    if v is None:
        return "N/A"
    return f"{v:.1f}%" if pct else f"{v:.3f}"


def _fmt_bear(v):
    if v is None:
        return "해당 기간 하락 구간 없음"
    return f"{v:.1f}%"


def _pill(text: str, bg: str, fg: str) -> str:
    return f"<span class='pill' style='background:{bg};color:{fg}'>{text}</span>"


def _chart_legend(*items) -> str:
    badges = " ".join(_pill(label, bg, fg) for label, bg, fg in items)
    return f"<div class='chart-legend'>{badges}</div>"


def _inject_styles():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ─── Base typography ─── */
html, body, [class*="css"], .stMarkdown, .element-container {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
.block-container { padding-top: 1.6rem !important; }

/* ─── Pill / badge ─── */
.pill {
  display: inline-block;
  font-size: 11px; font-weight: 500;
  padding: 4px 12px; border-radius: 20px;
  white-space: nowrap; line-height: 1.4;
}

/* ─── Chart legend ─── */
.chart-legend {
  display: flex; gap: 8px; flex-wrap: wrap;
  margin: 8px 0 4px 2px;
}

/* ─── Page title block ─── */
.page-title {
  font-size: 1.45rem; font-weight: 700;
  color: var(--text-color, #1a1a2e);
  letter-spacing: -0.025em; line-height: 1.2;
  margin-bottom: 2px;
}
.page-sub {
  font-size: 13px; color: #999; margin-bottom: 20px;
}

/* ─── Hero card ─── */
.hero-card {
  border-radius: 20px;
  padding: 40px 28px 32px;
  text-align: center;
  margin: 4px 0 12px;
}
.hero-card.up {
  background: linear-gradient(145deg, rgba(29,158,117,0.10) 0%, rgba(20,140,100,0.05) 100%);
  border: 1.5px solid rgba(29,158,117,0.25);
}
.hero-card.dn {
  background: linear-gradient(145deg, rgba(226,75,74,0.10) 0%, rgba(200,50,50,0.05) 100%);
  border: 1.5px solid rgba(226,75,74,0.25);
}
.hero-emoji { font-size: 2.6rem; line-height: 1; margin-bottom: 8px; }
.hero-direction {
  font-size: 2.8rem; font-weight: 700;
  letter-spacing: -0.03em; line-height: 1.1; margin-bottom: 14px;
}
.hero-direction.up { color: #0a6e45; }
.hero-direction.dn { color: #b02020; }
.hero-badges { display: flex; gap: 8px; justify-content: center; }

/* ─── Forecast note ─── */
.forecast-note {
  font-size: 11px; color: #aaa; text-align: center;
  margin-top: 6px; margin-bottom: 2px;
}

/* ─── Step flow ─── */
.step-flow {
  display: flex; align-items: center;
  gap: 0; margin: 10px 0 20px;
}
.step-card {
  flex: 1;
  background: var(--secondary-background-color, #f8f9fb);
  border: 1px solid rgba(0,0,0,0.07);
  border-radius: 12px; padding: 20px 14px; text-align: center;
}
.step-num {
  width: 26px; height: 26px; border-radius: 50%;
  background: #2a78d6; color: #fff;
  font-size: 12px; font-weight: 700;
  display: flex; align-items: center; justify-content: center;
  margin: 0 auto 8px;
}
.step-icon { font-size: 1.4rem; margin-bottom: 6px; }
.step-title { font-size: 13px; font-weight: 600; color: var(--text-color, #1a1a2e); margin-bottom: 4px; }
.step-desc { font-size: 11px; color: #888; line-height: 1.5; }
.step-arrow { font-size: 1.4rem; color: #ccc; padding: 0 8px; flex-shrink: 0; }

/* ─── Info box ─── */
.info-box {
  background: rgba(42,120,214,0.07);
  border: 1px solid rgba(42,120,214,0.18);
  border-radius: 10px; padding: 14px 18px; margin: 12px 0;
  font-size: 13px; color: #185FA5; line-height: 1.75;
}

/* ─── Signal cards ─── */
.sig-grid {
  display: grid; grid-template-columns: repeat(3, 1fr);
  gap: 14px; margin: 10px 0 6px;
}
.sig-card {
  background: var(--background-color, #ffffff);
  border: 1px solid rgba(0,0,0,0.07);
  border-radius: 12px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.05);
  padding: 18px 16px 14px;
}
.sig-icon { font-size: 1.2rem; margin-bottom: 4px; }
.sig-name { font-size: 12px; font-weight: 500; color: var(--text-color, #1a1a2e); margin-bottom: 1px; }
.sig-period { font-size: 10px; color: #bbb; margin-bottom: 10px; }
.sig-val {
  font-size: 1.6rem; font-weight: 700; line-height: 1;
  margin-bottom: 6px; font-variant-numeric: tabular-nums;
}
.sig-val.up  { color: #1D9E75; }
.sig-val.dn  { color: #E24B4A; }
.sig-val.neu { color: #EF9F27; }
.sig-val.na  { color: #bbb; }
.sig-sub { font-size: 10px; color: #bbb; }

/* ─── KPI cards ─── */
.kpi-card {
  background: var(--secondary-background-color, #f8f9fb);
  border-radius: 10px;
  border: 1px solid rgba(0,0,0,0.06);
  padding: 16px 18px; height: 100%;
}
.kpi-label {
  font-size: 10.5px; font-weight: 600; color: #aaa;
  text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;
}
.kpi-value {
  font-size: 22px; font-weight: 600;
  color: var(--text-color, #111); margin-bottom: 6px;
  font-variant-numeric: tabular-nums;
}

/* ─── Confidence bar ─── */
.cb-label { display:flex; justify-content:space-between; font-size:12px; color:#888; margin-bottom:5px; }
.cb-track { height:6px; border-radius:3px; background:rgba(0,0,0,0.08); overflow:hidden; margin-bottom:10px; }
.cb-fill  { height:100%; border-radius:3px; }

/* ─── Signal rows (inside expanders) ─── */
.signal-row {
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 0; border-bottom: 1px solid rgba(0,0,0,0.05);
}
.signal-row:last-child { border-bottom: none; }
.signal-label { font-size: 13px; color: #666; }
.signal-val   { font-size: 13px; font-weight: 600; }
.signal-val.up  { color: #1D9E75; }
.signal-val.dn  { color: #E24B4A; }
.signal-val.neu { color: #EF9F27; }

/* ─── Caution box ─── */
.caution-box {
  background: #fffbf2;
  border: 1px solid rgba(239,159,39,0.25);
  border-left: 3px solid #EF9F27;
  border-radius: 8px; padding: 14px 16px; margin-top: 12px;
}
.caution-box .c-title { font-size: 12px; font-weight: 600; color: #854F0B; margin-bottom: 6px; }
.caution-box .c-body  { font-size: 12px; color: #633806; line-height: 1.7; }

/* ─── Expert banner ─── */
.expert-banner {
  background: linear-gradient(135deg, #e6f1fb 0%, #d8eaf8 100%);
  border-radius: 10px; padding: 12px 16px; margin-bottom: 16px;
  border-left: 3px solid #2a78d6;
}
.eb-title { font-size: 13px; font-weight: 600; color: #0C447C; margin-bottom: 2px; }
.eb-body  { font-size: 12px; color: #185FA5; line-height: 1.6; }

/* ─── Flow cards (E2E view) ─── */
.flow-card {
  background: var(--secondary-background-color, #f8f9fb);
  border: 1px solid rgba(0,0,0,0.07);
  border-radius: 12px; padding: 22px 16px; text-align: center;
}
.fc-step {
  font-size: 10px; font-weight: 700; color: #2a78d6;
  text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;
}
.fc-icon { font-size: 1.6rem; margin-bottom: 6px; }
.fc-title { font-size: 14px; font-weight: 600; color: var(--text-color, #1a1a2e); }
.fc-code {
  font-size: 10px; color: #aaa; margin-top: 8px;
  font-family: 'SFMono-Regular', Consolas, monospace;
}

/* ─── Sidebar ─── */
[data-testid="stSidebar"] {
  border-right: 1px solid rgba(0,0,0,0.06) !important;
}
.sidebar-footer {
  font-size: 11px; color: #bbb; line-height: 2; padding: 4px 0;
}
.sidebar-footer b { color: #999; font-weight: 500; }

/* ─── Streamlit element tweaks ─── */
.stExpander > details > summary { font-size: 13px !important; }
</style>
""", unsafe_allow_html=True)


def _expert_banner():
    st.markdown("""
<div class="expert-banner">
  <div class="eb-title">🔬 전문가 모드 켜짐</div>
  <div class="eb-body">판단 근거, 모델 수치, 주의사항을 상세하게 보여줘요.</div>
</div>
""", unsafe_allow_html=True)


def _confidence_bar(pct: float, label: str, color: str = CLR_BLUE):
    st.markdown(f"""
<div class="cb-label">
  <span>{label}</span>
  <span style="color:{color};font-weight:600">{pct:.0f}%</span>
</div>
<div class="cb-track">
  <div class="cb-fill" style="width:{pct:.0f}%;background:{color}"></div>
</div>
""", unsafe_allow_html=True)


def _signal_rows(rows: list):
    html = ""
    for label, val, direction in rows:
        cls = {"up": "up", "dn": "dn", "neu": "neu"}.get(direction, "")
        html += (
            f"<div class='signal-row'>"
            f"<span class='signal-label'>{label}</span>"
            f"<span class='signal-val {cls}'>{val}</span>"
            f"</div>"
        )
    st.markdown(html, unsafe_allow_html=True)


def _caution_box(text: str):
    st.markdown(f"""
<div class="caution-box">
  <div class="c-title">⚠️ 주의사항</div>
  <div class="c-body">{text}</div>
</div>
""", unsafe_allow_html=True)


def render_ribbon_chart(out_df: pd.DataFrame, rmse: float, height: int = 380):
    """Plotly 리본 차트: 80%/95% 신뢰구간 밴드 + 예측 파선 + 실제값 실선."""
    dates   = out_df.index.tolist()
    actual  = out_df["실제값"].tolist()
    pred    = out_df["예측값"].tolist()
    sigma   = rmse

    upper95 = [p + 1.96 * sigma for p in pred]
    lower95 = [p - 1.96 * sigma for p in pred]
    upper80 = [p + 1.28 * sigma for p in pred]
    lower80 = [p - 1.28 * sigma for p in pred]

    all_vals = actual + pred + upper95 + lower95
    y_min = min(all_vals)
    y_max = max(all_vals)
    pad   = (y_max - y_min) * 0.08
    y_lo  = y_min - pad
    y_hi  = y_max + pad

    fig = go.Figure()

    fig.add_shape(type="rect", xref="paper", yref="y",
        x0=0, x1=1, y0=0, y1=y_hi,
        fillcolor="rgba(29,158,117,0.04)", line_width=0, layer="below")
    fig.add_shape(type="rect", xref="paper", yref="y",
        x0=0, x1=1, y0=y_lo, y1=0,
        fillcolor="rgba(226,75,74,0.04)", line_width=0, layer="below")

    fig.add_trace(go.Scatter(
        x=dates + dates[::-1], y=upper95 + lower95[::-1],
        fill="toself", fillcolor="rgba(150,150,150,0.12)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, name="95% CI",
    ))
    fig.add_trace(go.Scatter(
        x=dates + dates[::-1], y=upper80 + lower80[::-1],
        fill="toself", fillcolor="rgba(120,120,120,0.22)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, name="80% CI",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=pred,
        line=dict(color=CLR_BLUE, width=2, dash="dash"),
        showlegend=False, name="예측 중앙값",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=actual,
        line=dict(color=CLR_TEAL, width=2),
        mode="lines+markers", marker=dict(size=5, color=CLR_TEAL),
        showlegend=False, name="실제값",
    ))

    fig.update_layout(
        height=height,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        margin=dict(l=0, r=0, t=8, b=0),
        yaxis=dict(
            range=[y_lo, y_hi],
            gridcolor="rgba(136,135,128,0.12)",
            zeroline=True,
            zerolinecolor="rgba(100,100,100,0.45)",
            zerolinewidth=1,
            tickfont=dict(size=11, family="Inter, sans-serif"),
        ),
        xaxis=dict(showgrid=False, tickfont=dict(size=11, family="Inter, sans-serif")),
        font=dict(family="Inter, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(_chart_legend(
        ("실제값", "#d4f5e7", "#0a6e48"),
        ("예측 중앙값", BG_BLUE, "#185FA5"),
        ("80% 신뢰구간", "#ebebeb", "#555"),
        ("95% 신뢰구간", "#f2f2f2", "#888"),
    ), unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# 5. 공통 섹션 렌더러
# ──────────────────────────────────────────────────────────────────

def render_direction_headline(pred: float, date: pd.Timestamp,
                               target_date: pd.Timestamp, value_label: str):
    """현재 예측 헤드라인 — get_latest_forecast() 결과를 받아 표시."""
    up = pred > 0

    emoji    = "📈" if up else "📉"
    label    = "상승 전망" if up else "하락 전망"
    card_cls = "up" if up else "dn"

    direction_pill = _pill(
        "▲ 상승 전망" if up else "▼ 하락 전망",
        BG_TEAL if up else BG_RED,
        CLR_TEAL if up else CLR_RED,
    )
    ai_pill = _pill("AI 예측", BG_BLUE, "#185FA5")

    st.markdown(f"""
<div class="hero-card {card_cls}">
  <div class="hero-emoji">{emoji}</div>
  <div class="hero-direction {card_cls}">{label}</div>
  <div class="hero-badges">{ai_pill} {direction_pill}</div>
</div>
""", unsafe_allow_html=True)
    st.caption(
        f"📅 **{date.strftime('%Y년 %m월')}까지의 데이터** 기준 → "
        f"**{target_date.strftime('%Y년 %m월')} 방향** 예측 · "
        f"{value_label} {pred:+.2f}%"
    )


def _kpi(label: str, value: str, pill_text: str = None,
         pill_bg: str = BG_BLUE, pill_fg: str = "#185FA5"):
    pill_html = _pill(pill_text, pill_bg, pill_fg) if pill_text else ""
    st.markdown(
        f"<div class='kpi-card'>"
        f"<div class='kpi-label'>{label}</div>"
        f"<div class='kpi-value'>{value}</div>"
        f"{pill_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


def _pill_grade(v):
    if v is None:
        return None
    if v >= 80:
        return f"{v:.0f}%", BG_GREEN, "#3B6D11"
    if v >= 60:
        return f"{v:.0f}%", BG_AMBER, "#854F0B"
    return f"{v:.0f}%", BG_RED, "#A32D2D"


def render_metric_cards(metrics: dict, with_ic: bool = False):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        p = _pill_grade(metrics["dir_acc"])
        _kpi("방향 정확도 (전체)", _fmt(metrics["dir_acc"], pct=True),
             p[0] if p else None, p[1] if p else BG_BLUE, p[2] if p else "#185FA5")
    with c2:
        p = _pill_grade(metrics.get("dir_bull"))
        _kpi("방향 정확도 (Bull)", _fmt(metrics.get("dir_bull"), pct=True),
             p[0] if p else None, p[1] if p else BG_BLUE, p[2] if p else "#185FA5")
    with c3:
        p = _pill_grade(metrics.get("dir_bear"))
        _kpi("방향 정확도 (Bear)", _fmt_bear(metrics.get("dir_bear")),
             p[0] if p else None, p[1] if p else BG_BLUE, p[2] if p else "#185FA5")
    with c4:
        _kpi("RMSE (전체)", _fmt(metrics["rmse"]), "오차 지표", BG_BLUE, "#185FA5")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        _kpi("RMSE (Bull)", _fmt(metrics.get("rmse_bull")))
    with c6:
        _kpi("RMSE (Bear)", _fmt(metrics.get("rmse_bear")))
    with c7:
        _kpi("Asymmetric Loss", _fmt(metrics.get("asym_loss")))
    with c8:
        if with_ic:
            _kpi("IC (Spearman)", _fmt(metrics.get("ic")))
        else:
            _kpi("Hold-out 구간", f"{metrics['n_holdout']}개")


def _build_feat_map() -> dict:
    m = {}

    _regions = {
        "Americas": "미주", "Europe": "유럽", "Japan": "일본",
        "Asia_Pacific": "아태지역", "Worldwide": "전세계",
    }
    for r, ko in _regions.items():
        b = f"{r}_YoY"
        m[b]                       = f"{ko} 반도체 YoY"
        m[f"{b}_lag6"]             = f"{ko} 반도체 YoY (6개월 전)"
        m[f"{b}_lag12"]            = f"{ko} 반도체 YoY (12개월 전)"
        m[f"{b}_ma3"]              = f"{ko} 반도체 YoY (3개월 평균)"
        m[f"{b}_ma6"]              = f"{ko} 반도체 YoY (6개월 평균)"
        m[f"{b}_ma12"]             = f"{ko} 반도체 YoY (12개월 평균)"
        m[f"{b}_vol3"]             = f"{ko} 반도체 YoY (3개월 변동성)"
        m[f"{b}_vol6"]             = f"{ko} 반도체 YoY (6개월 변동성)"
        m[f"{b}_momentum_3_12"]    = f"{ko} 반도체 YoY 모멘텀"
        m[f"{b}_accel"]            = f"{ko} 반도체 YoY 가속도"
        m[f"{b}_vs_ma24"]          = f"{ko} 반도체 YoY (24개월 내 상대 위치)"
        m[f"wsts_{r}_YoY"]         = f"{ko} 반도체 매출 YoY"

    _tickers = {
        "SOX": "반도체지수 SOX", "NVDA": "NVIDIA", "TSM": "TSMC",
        "ASML": "ASML", "Samsung": "삼성전자", "SKHynix": "SK하이닉스",
    }
    for t, ko in _tickers.items():
        b = f"Ret_{t}"
        m[b]            = f"{ko} 수익률"
        m[f"{b}_lag6"]  = f"{ko} 수익률 (6개월 전)"
        m[f"{b}_lag12"] = f"{ko} 수익률 (12개월 전)"
        m[f"{b}_ma3"]   = f"{ko} 수익률 (3개월 평균)"
        m[f"{b}_ma6"]   = f"{ko} 수익률 (6개월 평균)"
        m[f"{b}_vol3"]  = f"{ko} 수익률 (3개월 변동성)"
        m[f"{b}_vol6"]  = f"{ko} 수익률 (6개월 변동성)"
    m["Eq_AvgRet"]       = "반도체 기업 평균 수익률"
    m["Eq_AvgRet_lag6"]  = "반도체 기업 평균 수익률 (6개월 전)"
    m["Eq_AvgRet_lag12"] = "반도체 기업 평균 수익률 (12개월 전)"

    _fred = {
        "FRED_SemiProd":  "반도체 생산지수 (미국)",
        "FRED_ISM_Mfg":   "ISM 제조업 지수",
        "FRED_IndProd":   "산업생산지수 (미국)",
        "FRED_PCE_Core":  "근원 PCE 물가",
        "FRED_MfgEmp":    "제조업 고용",
        "FRED_ConsSenti": "소비자 심리지수",
        "FRED_NewOrder":  "제조업 신규 주문",
        "FRED_InvSales":  "재고/매출 비율",
        "FRED_FedFunds":  "연방기금금리",
    }
    for key, ko in _fred.items():
        b = f"{key}_YoY"
        m[b]                    = f"{ko} YoY"
        m[f"{b}_lag6"]          = f"{ko} YoY (6개월 전)"
        m[f"{b}_lag12"]         = f"{ko} YoY (12개월 전)"
        m[f"{b}_ma3"]           = f"{ko} YoY (3개월 평균)"
        m[f"{b}_ma6"]           = f"{ko} YoY (6개월 평균)"
        m[f"{b}_momentum_3_12"] = f"{ko} YoY 모멘텀"
        m[f"{b}_accel"]         = f"{ko} YoY 가속도"
    m["FRED_T10Y2Y"]       = "장단기 금리차 (10년-2년)"
    m["FRED_T10Y2Y_lag6"]  = "장단기 금리차 (6개월 전)"
    m["FRED_T10Y2Y_lag12"] = "장단기 금리차 (12개월 전)"
    m["FRED_T10Y2Y_chg3"]  = "장단기 금리차 변화 (3개월)"

    m["ISM_above50"] = "ISM 50 초과 여부 (제조업 확장)"
    m["ISM_mom3"]    = "ISM 3개월 모멘텀"

    m["month_sin"] = "계절성 (사인)"
    m["month_cos"] = "계절성 (코사인)"

    m["T10Y3M"]              = "장단기 금리차 (10년-3개월)"
    m["T10Y3M_chg3"]         = "금리차 3개월 변화"
    m["T10Y3M_chg6"]         = "금리차 6개월 변화"
    m["T10Y3M_inverted"]     = "금리 역전 여부"
    m["T10Y3M_inv_streak"]   = "금리 역전 연속 기간"
    m["T10Y3M_lag6"]         = "장단기 금리차 (6개월 전)"
    m["T10Y3M_lag12"]        = "장단기 금리차 (12개월 전)"
    m["InvSales"]            = "재고/매출 비율"
    m["InvSales_diff3"]      = "재고/매출 변화 (3개월)"
    m["InvSales_diff6"]      = "재고/매출 변화 (6개월)"
    m["InvSales_lag6"]       = "재고/매출 비율 (6개월 전)"
    m["InvSales_lag12"]      = "재고/매출 비율 (12개월 전)"
    m["FedFunds"]            = "연방기금금리"
    m["FedFunds_diff6"]      = "금리 변화 (6개월)"
    m["FedFunds_diff12"]     = "금리 변화 (12개월)"
    m["FedFunds_lag6"]       = "연방기금금리 (6개월 전)"
    m["FedFunds_lag12"]      = "연방기금금리 (12개월 전)"

    m["v2_pred_ww_yoy"] = "AI 반도체 경기 예측 (1단계 출력)"

    return m


_FEAT_MAP       = _build_feat_map()
_FEAT_MAP_LOWER = {k.lower(): v for k, v in _FEAT_MAP.items()}


def _translate_feat(name: str) -> str:
    return _FEAT_MAP.get(name) or _FEAT_MAP_LOWER.get(name.lower(), name)


def render_shap_section(cfg: dict):
    st.markdown("#### 🔬 예측에 영향을 준 주요 지표 (상위 10)")
    st.caption("막대가 길수록 해당 지표가 이번 예측에 더 크게 영향을 미쳤어요.")
    try:
        shap_df = compute_shap_importance(cfg["model_path"], cfg["features_path"], cfg["target"])
        labels = [_translate_feat(n) for n in shap_df.index.tolist()]
        fig = go.Figure(go.Bar(
            x=shap_df["평균 |SHAP|"].values,
            y=labels,
            orientation="h",
            marker_color=CLR_BLUE,
            marker_line_width=0,
        ))
        fig.update_layout(
            height=400,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=4, r=16, t=8, b=0),
            yaxis=dict(
                autorange="reversed",
                automargin=True,
                gridcolor="rgba(0,0,0,0)",
                tickfont=dict(size=12, family="Inter, sans-serif"),
            ),
            xaxis=dict(
                gridcolor="rgba(136,135,128,0.15)", title="영향도",
                tickfont=dict(size=11, family="Inter, sans-serif"),
            ),
            font=dict(family="Inter, sans-serif"),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"SHAP 계산을 수행하지 못했습니다: {e}")


def render_detail_sections(metrics: dict, is_up: bool, expert_mode: bool):
    """📊 한 줄 요약 + 🔍 세부 분석 아코디언."""
    dir_acc  = metrics.get("dir_acc", 0)
    dir_bear = metrics.get("dir_bear")
    rmse     = metrics.get("rmse", 0)
    asym     = metrics.get("asym_loss", 0)
    n_ho     = metrics.get("n_holdout", 0)

    st.markdown("#### 📊 한 줄 요약")
    if expert_mode:
        st.markdown(
            f"방향 정확도 **{_fmt(dir_acc, pct=True)}** (Bear {_fmt(dir_bear, pct=True)}) · "
            f"RMSE **{rmse:.2f}** · AsymLoss **{asym:.2f}** · Hold-out {n_ho}개 기준."
        )
    else:
        if is_up:
            st.markdown(
                "반도체 사이클 지표가 상승 구간을 가리키고 있어요 📈 "
                "**→ HBM 수요**가 시그널을 이끌고 있어요."
            )
        else:
            st.markdown(
                "현재 사이클 지표는 하락 구간을 시사하고 있어요 📉 "
                "**→ 재고 조정 국면** 에 주의가 필요해요."
            )

    st.markdown("---")
    st.markdown("#### 🔍 세부 분석")

    with st.expander("📡 모델 성능 분석", expanded=False):
        if expert_mode:
            st.markdown(
                f"**Hold-out 평가 결과** (`{metrics.get('period', '')}`)\n\n"
                f"- 방향 정확도(전체): **{_fmt(dir_acc, pct=True)}**\n"
                f"- 방향 정확도(Bear): **{_fmt_bear(dir_bear)}** ← 핵심 지표\n"
                f"- RMSE: **{rmse:.3f}** · AsymLoss: **{asym:.3f}**\n"
                f"- 평가 샘플 수: {n_ho}개"
            )
            bear_dir = dir_bear or 0
            _signal_rows([
                ("방향 정확도 (전체)", _fmt(dir_acc, pct=True),
                 "up" if dir_acc >= 70 else "dn"),
                ("방향 정확도 (Bear)", _fmt_bear(dir_bear),
                 "up" if bear_dir >= 60 else ("neu" if bear_dir >= 40 else "dn")),
                ("RMSE", f"{rmse:.3f}", "neu"),
                ("AsymLoss", f"{asym:.3f}", "neu"),
            ])
        else:
            st.markdown(
                f"모델이 방향을 **{_fmt(dir_acc, pct=True)}** 정확도로 맞혔어요. "
                f"Bear(하락) 구간 정확도는 **{_fmt_bear(dir_bear)}** 이에요."
            )

    with st.expander("⚠️ 리스크 & 주의사항", expanded=False):
        if expert_mode:
            bear_dir = dir_bear or 0
            _signal_rows([
                ("Bear DirAcc 안정성", _fmt_bear(dir_bear),
                 "dn" if bear_dir < 60 else "up"),
                ("RMSE 대비 예측 신뢰", f"{rmse:.2f}", "neu"),
            ])
            _caution_box(
                "이 예측은 과거 데이터 패턴 기반의 통계 모델 출력값입니다. "
                "규제 리스크, 지정학적 이벤트, 기업 내부 정보 등 구조적 변화는 "
                "반영되지 않습니다. 투자 결정 시 이 수치만 단독으로 활용하지 마세요."
            )
        else:
            st.markdown(
                "이 예측은 참고용이에요. "
                "실제 투자 결정에는 다양한 요소를 종합적으로 고려해주세요. 🙏"
            )

    with st.expander("🎯 모델 신뢰도"):
        conf_color = CLR_TEAL if dir_acc >= 75 else (CLR_AMBER if dir_acc >= 60 else CLR_RED)
        _confidence_bar(dir_acc, "방향 정확도 기반 신뢰도", conf_color)
        if dir_bear is not None:
            bear_color = (CLR_TEAL if dir_bear >= 60
                          else (CLR_AMBER if dir_bear >= 40 else CLR_RED))
            st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
            _confidence_bar(dir_bear, "Bear 정확도 (하락 예측 신뢰도)", bear_color)


def render_confusion(df: pd.DataFrame):
    yt, yp = df["실제값"].values, df["예측값"].values
    tp = int(((yt > 0) & (yp > 0)).sum())
    fp = int(((yt <= 0) & (yp > 0)).sum())
    fn = int(((yt > 0) & (yp <= 0)).sum())
    tn = int(((yt <= 0) & (yp <= 0)).sum())
    cm = pd.DataFrame(
        [[tp, fn], [fp, tn]],
        index=["실제 상승", "실제 하락"],
        columns=["예측 상승", "예측 하락"],
    )
    st.dataframe(cm.style.background_gradient(cmap="Blues"), use_container_width=True)


def render_hit_history(out_df: pd.DataFrame, freq_label: str):
    st.markdown("#### 🎯 과거 적중 히스토리 (Hold-out)")
    d = out_df.copy()
    d["적중"] = (d["실제값"] > 0) == (d["예측값"] > 0)
    n, hit = len(d), int(d["적중"].sum())
    acc = d["적중"].mean() * 100 if n else 0.0

    cA, cB = st.columns([1, 3])
    with cA:
        st.metric("적중률", f"{acc:.1f}%", f"{hit}/{n} {freq_label} 적중")
    with cB:
        timeline = " ".join("✅" if v else "❌" for v in d["적중"])
        st.markdown("**적중 타임라인** (왼쪽=과거 → 오른쪽=최근)")
        st.markdown(
            f"<div style='font-size:1.5rem;letter-spacing:2px'>{timeline}</div>",
            unsafe_allow_html=True,
        )

    table = pd.DataFrame({
        "예측 방향": ["📈 상승" if v > 0 else "📉 하락" for v in d["예측값"]],
        "실제 방향": ["📈 상승" if v > 0 else "📉 하락" for v in d["실제값"]],
        "예측값": [f"{v:+.2f}%" for v in d["예측값"]],
        "실제값": [f"{v:+.2f}%" for v in d["실제값"]],
        "결과": ["✅" if v else "❌" for v in d["적중"]],
    }, index=d.index.strftime("%Y-%m"))
    with st.expander("적중 히스토리 상세"):
        st.dataframe(table, use_container_width=True)


# ──────────────────────────────────────────────────────────────────
# 6. Stage별 화면
# ──────────────────────────────────────────────────────────────────

def view_stage1(expert_mode: bool = False):
    cfg = STAGE1
    st.markdown(
        "<div class='page-title'>📦 반도체 경기 예측 (6개월 뒤)</div>"
        "<div class='page-sub'>전 세계 반도체 매출이 1년 전보다 얼마나 늘지 예측해요. SK하이닉스 전망의 출발점이에요.</div>",
        unsafe_allow_html=True,
    )

    # ── 진짜 현재 예측 (최신 피처 행 → 미래 방향) ──
    fc = None
    try:
        fc = get_latest_forecast(cfg["features_path"], cfg["model_path"])
        render_direction_headline(fc["pred"], fc["date"], fc["target_date"], cfg["value_label"])
    except Exception as e:
        st.error(f"현재 예측을 불러오지 못했습니다: {e}")

    # ── 백테스트 성능 (별도 expander) ──
    try:
        metrics, df = evaluate_stage(
            cfg["features_path"], cfg["model_path"], cfg["target"], cfg["test_eval"]
        )
    except Exception as e:
        st.error(f"Stage 1 평가 중 오류가 발생했습니다: {e}")
        return

    is_up = fc["pred"] > 0 if fc else True
    render_detail_sections(metrics, is_up, expert_mode)

    with st.expander("📉 백테스트 결과 — 과거 예측이 얼마나 맞았나요?"):
        st.caption(f"모델이 학습에 쓰지 않은 구간({metrics['period']})에서 예측값과 실제값을 비교한 검증 차트예요. 현재 예측과는 별개예요.")
        render_ribbon_chart(df, metrics["rmse"])

    with st.expander("📊 상세 성능 지표"):
        st.caption(f"평가 구간: {metrics['period']}  ·  피처 {metrics['n_features']}개")
        render_metric_cards(metrics)

    with st.expander("🔬 예측에 영향을 준 주요 지표"):
        render_shap_section(cfg)

    with st.expander("🎯 적중 히스토리"):
        render_hit_history(df, cfg["freq_label"])


def view_stage2(expert_mode: bool = False):
    cfg = STAGE2
    st.markdown(
        "<div class='page-title'>📈 SK하이닉스 주가 전망 (6개월)</div>"
        "<div class='page-sub'>반도체 경기 예측을 바탕으로 SK하이닉스 주가가 오를지 내릴지 판단해요.</div>",
        unsafe_allow_html=True,
    )

    # ── 진짜 현재 예측 ──
    fc = None
    try:
        fc = get_latest_forecast(cfg["features_path"], cfg["model_path"])
        render_direction_headline(fc["pred"], fc["date"], fc["target_date"], cfg["value_label"])
    except Exception as e:
        st.error(f"현재 예측을 불러오지 못했습니다: {e}")

    # ── 백테스트 성능 ──
    try:
        metrics, df = evaluate_stage(
            cfg["features_path"], cfg["model_path"], cfg["target"],
            cfg["test_eval"], with_ic=True
        )
    except Exception as e:
        st.error(f"Stage 2 평가 중 오류가 발생했습니다: {e}")
        return

    is_up = fc["pred"] > 0 if fc else True
    render_detail_sections(metrics, is_up, expert_mode)

    with st.expander("📉 백테스트 결과 — 과거 예측이 얼마나 맞았나요?"):
        st.caption(f"모델이 학습에 쓰지 않은 구간({metrics['period']})에서 예측값과 실제값을 비교한 검증 차트예요. 현재 예측과는 별개예요.")
        render_ribbon_chart(df, metrics["rmse"], height=340)

    with st.expander("📊 상세 성능 지표"):
        st.caption(f"평가 구간: {metrics['period']}  ·  피처 {metrics['n_features']}개")
        render_metric_cards(metrics, with_ic=True)

    with st.expander("🔀 방향 예측 혼동행렬"):
        render_confusion(df)

    with st.expander("🔬 SHAP 피처 중요도"):
        render_shap_section(cfg)

    with st.expander("🎯 적중 히스토리"):
        render_hit_history(df, cfg["freq_label"])


def _flow_box(title: str, subtitle: str, code: str = None):
    parts = title.split(' ', 1)
    icon  = parts[0] if parts else ''
    step  = parts[1] if len(parts) > 1 else title
    code_html = f'<div class="fc-code">{code}</div>' if code else ''
    st.markdown(f"""
<div class="flow-card">
  <div class="fc-step">{step}</div>
  <div class="fc-icon">{icon}</div>
  <div class="fc-title">{subtitle}</div>
  {code_html}
</div>
""", unsafe_allow_html=True)


def _flow_arrow():
    st.markdown(
        "<div style='text-align:center;font-size:1.6rem;color:#ccc;margin-top:1.4rem'>→</div>",
        unsafe_allow_html=True,
    )


def render_market_signals():
    st.markdown(
        "<div style='font-size:1rem;font-weight:600;margin:4px 0 6px'>📡 현재 시장 분위기</div>",
        unsafe_allow_html=True,
    )
    st.caption("코스피·미국 반도체지수는 **실시간** (최대 1시간 자동 갱신) · AI 예측 신호는 최신 모델 기준이에요.")

    mom = get_market_momentum()
    kospi_mom = mom.get("KOSPI")
    sox_mom   = mom.get("SOX")

    model_up = None
    try:
        fc2 = get_latest_forecast(STAGE2["features_path"], STAGE2["model_path"])
        model_up = bool(fc2["pred"] > 0)
    except Exception:
        pass

    def _val_html(v, is_bool: bool = False) -> str:
        if is_bool:
            if v is None:
                return '<div class="sig-val na">N/A</div>'
            cls = "up" if v else "dn"
            txt = "📈 상승" if v else "📉 하락"
            return f'<div class="sig-val {cls}">{txt}</div>'
        if v is None:
            return '<div class="sig-val na">N/A</div>'
        cls = "up" if v > 0 else "dn"
        return f'<div class="sig-val {cls}">{v:+.1f}%</div>'

    votes = []
    if kospi_mom is not None:
        votes.append(kospi_mom > 0)
    if sox_mom is not None:
        votes.append(sox_mom > 0)
    if model_up is not None:
        votes.append(model_up)

    pos = sum(votes) if votes else 0
    n   = len(votes) if votes else 0
    if not votes:
        combo_html = '<div class="sig-val na">N/A</div>'
    elif pos > n / 2:
        combo_html = '<div class="sig-val up">📈 상승 우세</div>'
    elif pos < n / 2:
        combo_html = '<div class="sig-val dn">📉 하락 우세</div>'
    else:
        combo_html = '<div class="sig-val neu">➖ 중립</div>'

    st.markdown(f"""
<div class="sig-grid">
  <div class="sig-card">
    <div class="sig-icon">🇰🇷</div>
    <div class="sig-name">코스피</div>
    <div class="sig-period">최근 3개월 변화</div>
    {_val_html(kospi_mom)}
    <div class="sig-sub">실시간 · 최대 1시간 전 데이터</div>
  </div>
  <div class="sig-card">
    <div class="sig-icon">💽</div>
    <div class="sig-name">미국 반도체지수 (SOX)</div>
    <div class="sig-period">최근 3개월 변화</div>
    {_val_html(sox_mom)}
    <div class="sig-sub">실시간 · 최대 1시간 전 데이터</div>
  </div>
  <div class="sig-card">
    <div class="sig-icon">🧭</div>
    <div class="sig-name">종합 신호</div>
    <div class="sig-period">코스피 + SOX + AI 예측</div>
    {combo_html}
    <div class="sig-sub">AI 신호는 최신 모델 기준</div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.caption(
        "※ 종합 신호 = 코스피·미국 반도체지수의 3개월 흐름 + AI 최신 예측을 합친 다수결이에요."
    )


def _plain_acc(dir_acc) -> str:
    n = round((dir_acc or 0) / 10)
    return f"과거 검증에서 **10번 중 약 {n}번** 방향을 맞혔어요."


def view_home(expert_mode: bool = False):
    """🏠 한눈에 보기 — 결론 · 시장 분위기 · 신뢰도를 한 페이지로."""
    # ── 진짜 현재 예측 (최신 피처 행 기반) ──
    fc2 = None
    try:
        fc2 = get_latest_forecast(STAGE2["features_path"], STAGE2["model_path"])
    except Exception as e:
        st.error(f"예측 결과를 불러오지 못했습니다: {e}")
        return

    up = fc2["pred"] > 0

    st.markdown("### 🔮 앞으로 6개월, SK하이닉스 주가는 오를까요?")
    render_direction_headline(fc2["pred"], fc2["date"], fc2["target_date"], STAGE2["value_label"])

    # 백테스트 정확도 참고용으로만 가져옴
    m2_metrics = None
    try:
        m2_metrics, _ = evaluate_stage(
            STAGE2["features_path"], STAGE2["model_path"],
            STAGE2["target"], STAGE2["test_eval"], with_ic=True,
        )
    except Exception:
        pass

    dir_acc = m2_metrics.get("dir_acc") if m2_metrics else None
    takeaway = (
        "AI는 향후 6개월 SK하이닉스 주가가 <b>오를 가능성</b>이 높다고 봐요."
        if up else
        "AI는 향후 6개월 SK하이닉스 주가가 <b>내릴 가능성</b>이 높다고 봐요."
    )
    acc_str = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', _plain_acc(dir_acc)) if dir_acc else ""
    st.markdown(
        f'<div class="info-box">{takeaway}<br>{acc_str}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### 🧭 이렇게 예측해요")
    st.markdown("""
<div class="step-flow">
  <div class="step-card">
    <div class="step-num">1</div>
    <div class="step-icon">🌐</div>
    <div class="step-title">반도체 경기</div>
    <div class="step-desc">전 세계 반도체가 6개월 뒤 얼마나 팔릴지 예측해요.</div>
  </div>
  <div class="step-arrow">›</div>
  <div class="step-card">
    <div class="step-num">2</div>
    <div class="step-icon">🔗</div>
    <div class="step-title">신호 연결</div>
    <div class="step-desc">반도체 경기 예측을 SK하이닉스 분석에 연결해요.</div>
  </div>
  <div class="step-arrow">›</div>
  <div class="step-card">
    <div class="step-num">3</div>
    <div class="step-icon">📈</div>
    <div class="step-title">주가 전망</div>
    <div class="step-desc">SK하이닉스 주가가 오를지 내릴지 최종 판단해요.</div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.divider()
    render_market_signals()

    st.divider()
    with st.expander("🎯 이 예측, 얼마나 믿을 수 있나요?"):
        if m2_metrics is not None and dir_acc is not None:
            dir_bear = m2_metrics.get("dir_bear")
            st.markdown(_plain_acc(dir_acc))
            conf_color = CLR_TEAL if dir_acc >= 75 else (CLR_AMBER if dir_acc >= 60 else CLR_RED)
            _confidence_bar(dir_acc, "전체 방향 정확도", conf_color)
            if dir_bear is not None:
                bear_color = (CLR_TEAL if dir_bear >= 60
                              else (CLR_AMBER if dir_bear >= 40 else CLR_RED))
                st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
                _confidence_bar(dir_bear, "하락장에서의 정확도", bear_color)
        st.caption("'검증'은 모델이 학습에 쓰지 않은 최근 데이터로 시험 본 결과예요. "
                   "참고용이며 투자 권유가 아니에요.")

    st.caption("👈 왼쪽 메뉴에서 단계별 상세 분석과 차트를 볼 수 있어요.")


def view_e2e(expert_mode: bool = False):
    st.markdown(
        "<div class='page-title'>🔗 예측은 어떻게 작동하나요?</div>"
        "<div class='page-sub'>반도체 경기 예측이 SK하이닉스 주가 전망으로 이어지는 전체 과정을 보여줘요.</div>",
        unsafe_allow_html=True,
    )

    f1, fa, f2, fb, f3 = st.columns([4, 1, 4, 1, 4])
    with f1:
        _flow_box("🌐 1단계", "반도체 경기 예측",
                  "best_xgboost_final.pkl" if expert_mode else None)
    with fa:
        _flow_arrow()
    with f2:
        _flow_box("🔗 연결", "예측 결과를 다음 단계로 전달",
                  BRIDGE_COL if expert_mode else None)
    with fb:
        _flow_arrow()
    with f3:
        _flow_box("📈 2단계", "SK하이닉스 주가 전망",
                  "skh_xgb_final.pkl" if expert_mode else None)

    st.divider()
    render_market_signals()

    with st.expander("① Stage 1 출력 시계열"):
        st.caption(f"lookahead 없이 재학습한 6개월 선행 반도체 매출 YoY 예측값(`{BRIDGE_COL}`)")
        try:
            s1pred = load_csv(STAGE1_PRED_PATH)
            if BRIDGE_COL in s1pred.columns:
                s1_data = s1pred[[BRIDGE_COL]].dropna()
                fig = go.Figure(go.Scatter(
                    x=s1_data.index, y=s1_data[BRIDGE_COL],
                    line=dict(color=CLR_BLUE, width=2),
                    mode="lines+markers", marker=dict(size=4),
                ))
                fig.update_layout(
                    height=280,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=False, margin=dict(l=0, r=0, t=8, b=0),
                    yaxis=dict(gridcolor="rgba(136,135,128,0.15)"),
                    xaxis=dict(showgrid=False),
                    font=dict(family="Inter, sans-serif"),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"`{BRIDGE_COL}` 컬럼을 찾을 수 없습니다.")
        except Exception as e:
            st.error(f"Stage 1 예측 데이터 로드 실패: {e}")

    with st.expander("② Bridge 피처 결합 확인"):
        try:
            s2feat = load_csv(STAGE2["features_path"])
            if BRIDGE_COL in s2feat.columns:
                st.success(f"Stage 2 피처셋에 `{BRIDGE_COL}` 포함 — 두 단계 정상 연결")
                n_total  = s2feat.shape[1]
                n_bridge = sum(1 for c in s2feat.columns if c.startswith("v2_pred"))
                m1, m2 = st.columns(2)
                m1.metric("전체 피처 수", f"{n_total}개")
                m2.metric("Bridge 피처", f"{n_bridge}개")
            else:
                st.warning(f"Stage 2 피처셋에서 `{BRIDGE_COL}`를 찾지 못했습니다.")
        except Exception as e:
            st.error(f"Stage 2 피처 데이터 로드 실패: {e}")

    with st.expander("③ 두 단계 성능 요약"):
        rows = []
        for cfg, with_ic in [(STAGE1, False), (STAGE2, True)]:
            try:
                m, _ = evaluate_stage(
                    cfg["features_path"], cfg["model_path"], cfg["target"],
                    cfg["test_eval"], with_ic=with_ic
                )
                rows.append({
                    "단계": f"{cfg['name']} · {cfg['title']}",
                    "방향정확도(전체)": _fmt(m["dir_acc"], pct=True),
                    "방향정확도(Bear)": _fmt_bear(m.get("dir_bear")),
                    "RMSE": _fmt(m["rmse"]),
                    "Asym Loss": _fmt(m["asym_loss"]),
                    "IC": _fmt(m.get("ic")) if with_ic else "—",
                })
            except Exception as e:
                rows.append({"단계": cfg["name"], "방향정확도(전체)": f"오류: {e}"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────
# 7. 메인
# ──────────────────────────────────────────────────────────────────

def main():
    _inject_styles()
    guard_artifacts()

    st.sidebar.title("📈 SK하이닉스 주가 전망")
    st.sidebar.caption("반도체 경기로 6개월 뒤 주가 방향을 예측해요")

    view = st.sidebar.radio(
        "메뉴",
        ["🏠 한눈에 보기", "A.  SK하이닉스 전망", "B.  반도체 경기", "작동 원리"],
        index=0,
    )

    st.sidebar.divider()
    expert_mode = st.sidebar.toggle("🔬 전문가 모드", value=False)
    st.sidebar.caption("SHAP·RMSE 등 전문 지표와 상세 수치를 함께 보여줘요.")

    st.sidebar.divider()
    with st.sidebar.expander("이 서비스는 어떻게 작동하나요?"):
        st.markdown(
            """
**1. 데이터 수집**
전 세계 반도체 출하량(WSTS), 미국 경제지표(FRED), 주요 반도체 기업 주가를 자동으로 모읍니다.

**2. 반도체 경기 예측 (B)**
수집한 데이터를 AI 모델에 넣어 6개월 뒤 반도체 시장이 성장할지 예측합니다.

**3. SK하이닉스 주가 전망 (A)**
반도체 경기 예측 결과를 포함한 신호들로 SK하이닉스 주가가 6개월 뒤 오를지 내릴지 판단합니다.

**4. 주기적 업데이트**
분기마다 (1·4·7·10월) 새 데이터로 모델을 다시 학습해 예측을 갱신합니다.
"""
        )

    st.sidebar.divider()
    now_kst = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")
    st.sidebar.markdown(
        f"<div class='sidebar-footer'>"
        f"📡 <b>코스피·SOX</b>: 실시간 (1시간 갱신)<br>"
        f"🤖 <b>AI 예측 기준</b>: 최신 모델 피처 기준<br>"
        f"🕐 <b>페이지 로드</b>: {now_kst}"
        f"</div>",
        unsafe_allow_html=True,
    )

    if expert_mode:
        _expert_banner()

    st.markdown(
        "<div class='page-title'>반도체 사이클 기반 SK하이닉스 수익률 예측</div>"
        "<div class='page-sub'>전 세계 반도체 경기를 분석해 SK하이닉스 6개월 주가 방향을 예측해요</div>",
        unsafe_allow_html=True,
    )

    if view.startswith("🏠"):
        view_home(expert_mode)
    elif view.startswith("A."):
        view_stage2(expert_mode)
    elif view.startswith("B."):
        view_stage1(expert_mode)
    else:
        view_e2e(expert_mode)


if __name__ == "__main__":
    main()
