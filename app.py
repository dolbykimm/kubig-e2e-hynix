"""
app.py — E2E 반도체 사이클 → SK하이닉스 수익률 예측 대시보드 (Streamlit)
================================================================================
Stage 1 (반도체 출하량 YoY 예측) → Stage 2 (SK하이닉스 6개월 수익률 방향 예측)
2단계 파이프라인의 학습 결과를 발표용으로 시각화한다.
"""

import os
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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

# ── 컬러 팔레트 ───────────────────────────────────────────────────
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
    """
    S3에서 ARTIFACTS를 APP_ROOT 하위로 다운로드한다.
    세션당 1회만 실행되도록 cache_resource로 캐싱.
    """
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
    """다운로드 결과를 검사하고, 문제가 있으면 안내 후 대시보드를 중단한다."""
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
    import xgboost as xgb

    bundle  = load_model(model_path)
    model   = bundle["model"]
    feats   = bundle["feature_names"]
    params  = model.get_params()

    df = load_csv(features_path)
    use_feats = [f for f in feats if f in df.columns]
    df_clean  = df.dropna(subset=[target])
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


def _pill(text: str, bg: str, fg: str) -> str:
    return (
        f"<span style='background:{bg};color:{fg};font-size:11px;font-weight:500;"
        f"padding:3px 10px;border-radius:20px;white-space:nowrap'>{text}</span>"
    )


def _chart_legend(*items) -> str:
    badges = " ".join(_pill(label, bg, fg) for label, bg, fg in items)
    return f"<div style='display:flex;gap:8px;margin-top:10px;flex-wrap:wrap'>{badges}</div>"


def _inject_styles():
    st.markdown("""
<style>
.signal-row {
  display: flex; align-items: center; justify-content: space-between;
  padding: 10px 0; border-bottom: 0.5px solid rgba(136,135,128,0.25);
}
.signal-row:last-child { border-bottom: none; }
.signal-label { font-size: 13px; color: var(--text-secondary, #555); }
.signal-val   { font-size: 13px; font-weight: 500; }
.signal-val.up  { color: #1D9E75; }
.signal-val.dn  { color: #E24B4A; }
.signal-val.neu { color: #EF9F27; }

.cb-label { display:flex; justify-content:space-between; font-size:12px;
            color:#888; margin-bottom:5px; }
.cb-track { height:8px; border-radius:4px; background:rgba(136,135,128,0.2); overflow:hidden; }
.cb-fill  { height:100%; border-radius:4px; }

.caution-box { background:#faeeda; border-radius:10px; padding:14px; margin-top:12px; }
.caution-box .c-title { font-size:12px; font-weight:500; color:#854F0B; margin-bottom:6px; }
.caution-box .c-body  { font-size:12px; color:#633806; line-height:1.7; }

.expert-banner { background:#e6f1fb; border-radius:10px; padding:12px 16px;
                 margin-bottom:16px; border-left:3px solid #2a78d6; }
.eb-title { font-size:13px; font-weight:500; color:#0C447C; margin-bottom:2px; }
.eb-body  { font-size:12px; color:#185FA5; line-height:1.6; }

.kpi-card { background:var(--secondary-background-color,#f8f9fa);
            border-radius:10px; border:0.5px solid rgba(136,135,128,0.25);
            padding:14px 16px; }
.kpi-label { font-size:12px; color:#888; margin-bottom:4px; }
.kpi-value { font-size:22px; font-weight:500; color:var(--text-color,#111);
             margin-bottom:6px; }
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
  <span style="color:{color};font-weight:500">{pct:.0f}%</span>
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

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates + dates[::-1],
        y=upper95 + lower95[::-1],
        fill="toself", fillcolor="rgba(29,158,117,0.10)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, name="95% CI",
    ))
    fig.add_trace(go.Scatter(
        x=dates + dates[::-1],
        y=upper80 + lower80[::-1],
        fill="toself", fillcolor="rgba(29,158,117,0.22)",
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
        yaxis=dict(gridcolor="rgba(136,135,128,0.15)", zeroline=False),
        xaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(_chart_legend(
        ("예측 중앙값", BG_BLUE, "#185FA5"),
        ("80% 신뢰구간", BG_GREEN, "#3B6D11"),
        ("95% 신뢰구간", BG_TEAL, "#085041"),
    ), unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# 5. 공통 섹션 렌더러
# ──────────────────────────────────────────────────────────────────

def render_direction_headline(out_df: pd.DataFrame, value_label: str):
    pred = float(out_df["예측값"].iloc[-1])
    date = out_df.index[-1]
    up   = pred > 0

    emoji  = "📈" if up else "📉"
    label  = "상승" if up else "하락"
    color  = CLR_TEAL if up else CLR_RED
    bg     = BG_TEAL  if up else BG_RED

    direction_pill = _pill("▲ 상승 전망" if up else "▼ 하락 전망", bg, color)
    ai_pill        = _pill("AI 예측", BG_BLUE, "#185FA5")

    st.markdown(
        f"<div style='text-align:center;padding:1.5rem;border-radius:16px;"
        f"background:{bg};margin:0.2rem 0 0.4rem 0;'>"
        f"<div style='font-size:3.4rem;font-weight:500;color:{color};line-height:1.15'>"
        f"{emoji} {label} 전망</div>"
        f"<div style='margin-top:10px;display:flex;gap:8px;justify-content:center'>"
        f"{ai_pill} {direction_pill}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"📅 기준 시점 **{date.date()}** · {value_label} **{pred:+.2f}%**")


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
    """정확도 수치 → (pill_text, bg, fg) 튜플."""
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
        _kpi("방향 정확도 (Bear)", _fmt(metrics.get("dir_bear"), pct=True),
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


def render_shap_section(cfg: dict):
    st.markdown("#### 🔬 SHAP 피처 중요도 (상위 10)")
    st.caption("평균 |SHAP| 값이 클수록 예측에 더 많이 기여한 피처")
    try:
        shap_df = compute_shap_importance(cfg["model_path"], cfg["features_path"], cfg["target"])
        fig = go.Figure(go.Bar(
            x=shap_df["평균 |SHAP|"].values,
            y=shap_df.index.tolist(),
            orientation="h",
            marker_color=CLR_BLUE,
        ))
        fig.update_layout(
            height=360,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=8, b=0),
            yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="rgba(136,135,128,0.15)"),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"SHAP 계산을 수행하지 못했습니다: {e}")


def render_detail_sections(metrics: dict, out_df: pd.DataFrame,
                           stage: str, expert_mode: bool):
    """📊 한 줄 요약 + 🔍 세부 분석 아코디언 + 🎯 신뢰도 바."""
    dir_acc  = metrics.get("dir_acc", 0)
    dir_bear = metrics.get("dir_bear")
    rmse     = metrics.get("rmse", 0)
    asym     = metrics.get("asym_loss", 0)
    n_ho     = metrics.get("n_holdout", 0)
    pred_last = float(out_df["예측값"].iloc[-1])
    is_up     = pred_last > 0

    # ── 📊 한 줄 요약 ──
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
                "HBM 수요가 시그널을 이끌고 있어요."
            )
        else:
            st.markdown(
                "현재 사이클 지표는 하락 구간을 시사하고 있어요 📉 "
                "재고 조정 국면에 주의가 필요해요."
            )

    st.markdown("---")

    # ── 🔍 세부 분석 아코디언 ──
    st.markdown("#### 🔍 세부 분석")

    with st.expander("📡 모델 성능 분석", expanded=False):
        if expert_mode:
            st.markdown(
                f"**Hold-out 평가 결과** (`{metrics.get('period', '')}`)\n\n"
                f"- 방향 정확도(전체): **{_fmt(dir_acc, pct=True)}**\n"
                f"- 방향 정확도(Bear): **{_fmt(dir_bear, pct=True)}** ← 핵심 지표\n"
                f"- RMSE: **{rmse:.3f}** · AsymLoss: **{asym:.3f}**\n"
                f"- 평가 샘플 수: {n_ho}개"
            )
            bear_dir = dir_bear or 0
            _signal_rows([
                ("방향 정확도 (전체)", _fmt(dir_acc, pct=True),
                 "up" if dir_acc >= 70 else "dn"),
                ("방향 정확도 (Bear)", _fmt(dir_bear, pct=True),
                 "up" if bear_dir >= 60 else ("neu" if bear_dir >= 40 else "dn")),
                ("RMSE", f"{rmse:.3f}", "neu"),
                ("AsymLoss", f"{asym:.3f}", "neu"),
            ])
        else:
            st.markdown(
                f"모델이 방향을 **{_fmt(dir_acc, pct=True)}** 정확도로 맞혔어요. "
                f"Bear(하락) 구간 정확도는 **{_fmt(dir_bear, pct=True)}** 이에요."
            )

    with st.expander("📈 예측 vs 실제 흐름", expanded=False):
        if expert_mode:
            st.markdown("**Hold-out 구간 예측/실제값 (수치)**")
            st.dataframe(out_df.style.format("{:.2f}"), use_container_width=True)
        else:
            st.markdown(
                "최근 예측과 실제 흐름을 보여줘요. "
                "초록 선이 실제, 파란 점선이 예측이에요."
            )

    with st.expander("⚠️ 리스크 & 주의사항", expanded=False):
        if expert_mode:
            bear_dir = dir_bear or 0
            _signal_rows([
                ("Bear DirAcc 안정성", _fmt(dir_bear, pct=True),
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
    st.header("📦 반도체 경기 예측 (6개월 뒤)")
    st.caption("전 세계 반도체 매출이 1년 전보다 얼마나 늘지 예측해요. SK하이닉스 전망의 출발점이에요.")

    try:
        metrics, df = evaluate_stage(
            cfg["features_path"], cfg["model_path"], cfg["target"], cfg["test_eval"]
        )
    except Exception as e:
        st.error(f"Stage 1 평가 중 오류가 발생했습니다: {e}")
        return

    # ── 1차 노출: 방향 예측 + 차트만 ──
    render_direction_headline(df, cfg["value_label"])
    render_ribbon_chart(df, metrics["rmse"])

    render_detail_sections(metrics, df, "stage1", expert_mode)

    # ── 2차: 토글로 숨긴 상세 정보 ──
    with st.expander("📊 상세 성능 지표"):
        st.caption(f"평가 구간: {metrics['period']}  ·  피처 {metrics['n_features']}개")
        render_metric_cards(metrics)

    with st.expander("🔬 SHAP 피처 중요도"):
        render_shap_section(cfg)

    with st.expander("🎯 적중 히스토리"):
        render_hit_history(df, cfg["freq_label"])


def view_stage2(expert_mode: bool = False):
    cfg = STAGE2
    st.header("📈 SK하이닉스 주가 전망 (6개월)")
    st.caption("반도체 경기 예측을 바탕으로 SK하이닉스 주가가 오를지 내릴지 판단해요.")

    try:
        metrics, df = evaluate_stage(
            cfg["features_path"], cfg["model_path"], cfg["target"],
            cfg["test_eval"], with_ic=True
        )
    except Exception as e:
        st.error(f"Stage 2 평가 중 오류가 발생했습니다: {e}")
        return

    # ── 1차 노출: 방향 예측 + 차트만 ──
    render_direction_headline(df, cfg["value_label"])
    render_ribbon_chart(df, metrics["rmse"], height=360)

    render_detail_sections(metrics, df, "stage2", expert_mode)

    # ── 2차: 토글로 숨긴 상세 정보 ──
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
    with st.container(border=True):
        st.markdown(f"### {title}")
        st.markdown(f"**{subtitle}**")
        if code:
            st.caption(f"`{code}`")


def _flow_arrow():
    st.markdown(
        "<div style='text-align:center;font-size:2.6rem;margin-top:1.6rem'>➡️</div>",
        unsafe_allow_html=True,
    )


def render_market_signals():
    st.subheader("📊 오늘의 시장 분위기")
    st.caption("실시간 주식시장 흐름과 AI 예측을 합친 현재 종합 신호예요.")

    mom = get_market_momentum()
    kospi_mom = mom.get("KOSPI")
    sox_mom   = mom.get("SOX")

    model_up = None
    try:
        _, out2 = evaluate_stage(
            STAGE2["features_path"], STAGE2["model_path"],
            STAGE2["target"], STAGE2["test_eval"], with_ic=True
        )
        model_up = bool(out2["예측값"].iloc[-1] > 0)
    except Exception:
        pass

    def _mom_str(v):
        return "N/A" if v is None else f"{v:+.1f}%"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("🇰🇷 코스피 (최근 3개월)", _mom_str(kospi_mom),
                  delta=None if kospi_mom is None else f"{kospi_mom:+.1f}%")
    with c2:
        st.metric("💽 미국 반도체지수 (최근 3개월)", _mom_str(sox_mom),
                  delta=None if sox_mom is None else f"{sox_mom:+.1f}%")

    votes = []
    if kospi_mom is not None:
        votes.append(kospi_mom > 0)
    if sox_mom is not None:
        votes.append(sox_mom > 0)
    if model_up is not None:
        votes.append(model_up)

    with c3:
        if not votes:
            st.metric("🧭 종합 신호", "N/A")
        else:
            pos = sum(votes)
            if pos > len(votes) / 2:
                st.metric("🧭 종합 신호", "📈 상승 우세",
                          delta=f"{pos}/{len(votes)} 신호 상승")
            elif pos < len(votes) / 2:
                st.metric("🧭 종합 신호", "📉 하락 우세",
                          delta=f"-{len(votes)-pos}/{len(votes)} 신호 하락")
            else:
                st.metric("🧭 종합 신호", "➖ 중립")

    st.caption(
        "※ 종합 신호 = 코스피·미국 반도체지수의 3개월 흐름 + AI 최신 예측을 합친 다수결이에요. "
        "주가지수는 실시간(최대 1시간 단위 갱신), AI 신호는 최신 예측 기준이에요."
    )


def _plain_acc(dir_acc) -> str:
    n = round((dir_acc or 0) / 10)
    return f"과거 검증에서 **10번 중 약 {n}번** 방향을 맞혔어요."


def view_home(expert_mode: bool = False):
    """🏠 한눈에 보기 — 결론 · 시장 분위기 · 신뢰도를 한 페이지로."""
    try:
        m2, df2 = evaluate_stage(
            STAGE2["features_path"], STAGE2["model_path"],
            STAGE2["target"], STAGE2["test_eval"], with_ic=True,
        )
    except Exception as e:
        st.error(f"예측 결과를 불러오지 못했습니다: {e}")
        return

    up = float(df2["예측값"].iloc[-1]) > 0

    # ── 결론 (가장 중요) ──
    st.markdown("### 🔮 앞으로 6개월, SK하이닉스 주가는 오를까요?")
    render_direction_headline(df2, STAGE2["value_label"])

    takeaway = ("AI는 향후 6개월 SK하이닉스 주가가 **오를 가능성**이 높다고 봐요."
                if up else
                "AI는 향후 6개월 SK하이닉스 주가가 **내릴 가능성**이 높다고 봐요.")
    st.info(f"{takeaway}  \n{_plain_acc(m2['dir_acc'])}")

    # ── 예측 과정 (쉬운 3단계) ──
    st.markdown("#### 🧭 이렇게 예측해요")
    s1, s2, s3 = st.columns(3)
    with s1:
        with st.container(border=True):
            st.markdown("##### 🌐 1. 반도체 경기")
            st.caption("전 세계 반도체가 6개월 뒤 얼마나 팔릴지 예측해요.")
    with s2:
        with st.container(border=True):
            st.markdown("##### 🔗 2. 신호 연결")
            st.caption("반도체 경기 예측을 SK하이닉스 분석에 연결해요.")
    with s3:
        with st.container(border=True):
            st.markdown("##### 📈 3. 주가 전망")
            st.caption("SK하이닉스 주가가 오를지 내릴지 최종 판단해요.")

    st.divider()
    render_market_signals()

    st.divider()
    with st.expander("🎯 이 예측, 얼마나 믿을 수 있나요?"):
        dir_acc  = m2.get("dir_acc", 0)
        dir_bear = m2.get("dir_bear")
        st.markdown(_plain_acc(dir_acc))
        conf_color = CLR_TEAL if dir_acc >= 75 else (CLR_AMBER if dir_acc >= 60 else CLR_RED)
        _confidence_bar(dir_acc, "전체 방향 정확도", conf_color)
        if dir_bear is not None:
            bear_color = (CLR_TEAL if dir_bear >= 60
                          else (CLR_AMBER if dir_bear >= 40 else CLR_RED))
            st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
            _confidence_bar(dir_bear, "하락장에서의 정확도", bear_color)
        st.caption("‘검증’은 모델이 학습에 쓰지 않은 최근 데이터로 시험 본 결과예요. "
                   "참고용이며 투자 권유가 아니에요.")

    st.caption("👈 왼쪽 메뉴에서 단계별 상세 분석과 차트를 볼 수 있어요.")


def view_e2e(expert_mode: bool = False):
    st.header("🔗 예측은 어떻게 작동하나요?")
    st.caption("반도체 경기 예측이 SK하이닉스 주가 전망으로 이어지는 전체 과정을 보여줘요.")

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

    # ── 1차 노출: 현재 시장 신호 ──
    render_market_signals()

    # ── 2차: 토글로 숨긴 기술 상세 ──
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
                    "방향정확도(Bear)": _fmt(m["dir_bear"], pct=True),
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
        ["🏠 한눈에 보기", "📈 SK하이닉스 전망", "📦 반도체 경기", "🔗 작동 원리"],
        index=0,
    )

    st.sidebar.divider()
    expert_mode = st.sidebar.toggle("🔬 전문가 모드", value=False)
    st.sidebar.caption("SHAP·RMSE 등 전문 지표와 상세 수치를 함께 보여줘요.")

    st.sidebar.divider()
    st.sidebar.markdown(
        "**예측 3단계**\n\n"
        "1. 🌐 반도체 경기 예측\n"
        "2. 🔗 신호 연결\n"
        "3. 📈 SK하이닉스 주가 전망"
    )

    if expert_mode:
        _expert_banner()

    st.title("반도체 사이클 기반 SK하이닉스 수익률 예측")

    if view.startswith("🏠"):
        view_home(expert_mode)
    elif view.startswith("📈"):
        view_stage2(expert_mode)
    elif view.startswith("📦"):
        view_stage1(expert_mode)
    else:
        view_e2e(expert_mode)


if __name__ == "__main__":
    main()
