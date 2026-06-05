"""
app.py — E2E 반도체 사이클 → SK하이닉스 수익률 예측 대시보드 (Streamlit)
================================================================================
Stage 1 (반도체 출하량 YoY 예측) → Stage 2 (SK하이닉스 6개월 수익률 방향 예측)
2단계 파이프라인의 학습 결과를 발표용으로 시각화한다.

[실행 흐름]
  1. 앱 시작 시 S3(S3_BUCKET_NAME)에서 모델/데이터 산출물을 /app 경로로 다운로드
  2. 산출물이 하나라도 없으면 "모델 학습이 필요합니다" 안내 후 대시보드 중단
  3. 사이드바에서 Stage1 / Stage2 / E2E 선택 → 해당 결과 시각화

[기술 노트]
  - 모델 로딩은 st.cache_resource, 데이터 로딩은 st.cache_data로 캐싱
  - 성능 지표(metrics)는 hold-out 평가를 재현해 런타임 계산
    (metrics CSV를 별도로 내려받지 않아도 동작하도록)
  - 한글 폰트가 없는 Linux 컨테이너에서도 깨지지 않도록
    matplotlib 대신 Streamlit 네이티브 차트(st.line_chart 등) 사용
"""

import os
import pickle

import numpy as np
import pandas as pd
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

# 컨테이너 작업 경로(/app). 로컬 실행 시에는 app.py가 위치한 폴더로 폴백.
APP_ROOT = os.getenv("APP_ROOT", os.path.dirname(os.path.abspath(__file__)))

# S3에서 받아와야 하는 산출물 (S3 key == 로컬 상대경로)
ARTIFACTS = [
    "stage1/outputs/models/best_xgboost_final.pkl",
    "stage1/outputs/data/features_dataset.csv",
    "stage2/outputs/models/skh_xgb_final.pkl",
    "stage2/outputs/data/stage2_features.csv",
    "stage2/outputs/data/stage1_predictions.csv",
]

# Asymmetric Loss 가중치 (stage1/2 config.py와 동일) — Bear 오예측 페널티 강화
W_BULL_CORRECT, W_BULL_WRONG = 1.0, 2.0
W_BEAR_CORRECT, W_BEAR_WRONG = 1.5, 3.0
BEAR_SAMPLE_W = 2.0

# Stage별 메타데이터
STAGE1 = {
    "name": "Stage 1",
    "title": "반도체 출하량 YoY 예측",
    "features_path": os.path.join(APP_ROOT, "stage1/outputs/data/features_dataset.csv"),
    "model_path":    os.path.join(APP_ROOT, "stage1/outputs/models/best_xgboost_final.pkl"),
    "target":        "TARGET_Worldwide_YoY_T6",
    "test_eval":     24,        # hold-out 개월 수
    "unit":          "%",
    "y_label":       "Worldwide 반도체 매출 YoY (%)",
    "freq_label":    "개월",
}
STAGE2 = {
    "name": "Stage 2",
    "title": "SK하이닉스 6개월 수익률 방향 예측",
    "features_path": os.path.join(APP_ROOT, "stage2/outputs/data/stage2_features.csv"),
    "model_path":    os.path.join(APP_ROOT, "stage2/outputs/models/skh_xgb_final.pkl"),
    "target":        "TARGET_SKH_6M_RET",
    "test_eval":     12,        # hold-out 분기 수
    "unit":          "%",
    "y_label":       "SK하이닉스 6개월 수익률 (%)",
    "freq_label":    "분기",
}

STAGE1_PRED_PATH = os.path.join(APP_ROOT, "stage2/outputs/data/stage1_predictions.csv")
# Stage 1 → Stage 2로 전달되는 핵심 피처(예측값) 컬럼명
BRIDGE_COL = "v2_pred_ww_yoy"


# ──────────────────────────────────────────────────────────────────
# 1. S3 산출물 다운로드
# ──────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="S3에서 모델/데이터 산출물을 내려받는 중...")
def download_artifacts():
    """
    S3에서 ARTIFACTS를 APP_ROOT 하위로 다운로드한다.
    세션당 1회만 실행되도록 cache_resource로 캐싱.

    반환: dict(status, missing, error)
      - status == "ok"          : 전부 성공
      - status == "missing"     : 일부 파일이 버킷에 없음 → 학습 필요
      - status == "no_bucket"   : S3_BUCKET_NAME 미설정
      - status == "s3_error"    : 자격증명/네트워크 등 S3 접근 실패
    """
    bucket = os.getenv("S3_BUCKET_NAME")
    if not bucket:
        return {"status": "no_bucket", "missing": [], "error": None}

    try:
        import boto3
        from botocore.exceptions import ClientError, BotoCoreError, NoCredentialsError
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
            # 객체가 없으면(404/NoSuchKey) "학습 필요" 신호로 수집
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
    """pkl 번들 로드: {'model', 'feature_names', 'best_params'}."""
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    """index_col=0(날짜) 기준 CSV 로드."""
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
    """7~8개 표준 지표 계산 (evaluate 스크립트와 동일 정의)."""
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
        # Spearman 순위상관(IC). scipy는 scikit-learn 의존성으로 항상 설치됨.
        ic = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
        metrics["ic"] = float(ic) if pd.notna(ic) else None
    return metrics


@st.cache_data(show_spinner="모델 성능을 평가하는 중...")
def evaluate_stage(features_path: str, model_path: str, target: str,
                   test_eval: int, with_ic: bool = False):
    """
    저장된 최종 모델의 하이퍼파라미터로 tune 구간 재학습 → hold-out 예측.
    반환: (metrics dict, 예측/실제 정렬 DataFrame)
    """
    import xgboost as xgb

    bundle  = load_model(model_path)
    model   = bundle["model"]
    feats   = bundle["feature_names"]
    params  = model.get_params()

    df = load_csv(features_path)
    # 모델이 학습한 피처만 사용 (없는 컬럼은 제외)
    use_feats = [f for f in feats if f in df.columns]
    df_clean  = df.dropna(subset=[target])
    X = df_clean[use_feats].ffill().fillna(0)
    y = df_clean[target]

    split = len(X) - test_eval
    X_tune, y_tune = X.iloc[:split], y.iloc[:split]
    X_ho,   y_ho   = X.iloc[split:], y.iloc[split:]

    # Bear(하락) 구간 sample_weight 강화 후 재학습
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


# ──────────────────────────────────────────────────────────────────
# 4. 공통 UI 헬퍼
# ──────────────────────────────────────────────────────────────────

def _fmt(v, pct=False):
    if v is None:
        return "N/A"
    return f"{v:.1f}%" if pct else f"{v:.3f}"


def render_metric_cards(metrics: dict, with_ic: bool = False):
    """성능 지표를 KPI 카드 형태로 표시."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("방향 정확도 (전체)", _fmt(metrics["dir_acc"], pct=True))
    c2.metric("방향 정확도 (Bull/상승)", _fmt(metrics["dir_bull"], pct=True))
    c3.metric("방향 정확도 (Bear/하락)", _fmt(metrics["dir_bear"], pct=True))
    c4.metric("RMSE (전체)", _fmt(metrics["rmse"]))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("RMSE (Bull)", _fmt(metrics["rmse_bull"]))
    c6.metric("RMSE (Bear)", _fmt(metrics["rmse_bear"]))
    c7.metric("Asymmetric Loss", _fmt(metrics["asym_loss"]))
    if with_ic:
        c8.metric("IC (Spearman)", _fmt(metrics.get("ic")))
    else:
        c8.metric("Hold-out 구간", f"{metrics['n_holdout']}개")


def render_confusion(df: pd.DataFrame):
    """상승/하락 방향 혼동행렬."""
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
    st.dataframe(cm.style.background_gradient(cmap="Blues"), width='stretch')


# ──────────────────────────────────────────────────────────────────
# 5. Stage별 화면
# ──────────────────────────────────────────────────────────────────

def view_stage1():
    cfg = STAGE1
    st.header("📦 Stage 1 — 반도체 출하량(WW 매출) YoY 예측")
    st.caption(
        "전 세계 반도체 매출의 6개월 선행 전년동월대비(YoY) 증감률을 XGBoost로 예측합니다. "
        "Bear(하락) 구간 오예측에 더 큰 페널티를 주도록 학습되었습니다."
    )

    try:
        metrics, df = evaluate_stage(
            cfg["features_path"], cfg["model_path"], cfg["target"], cfg["test_eval"]
        )
    except Exception as e:
        st.error(f"Stage 1 평가 중 오류가 발생했습니다: {e}")
        return

    st.subheader("모델 성능 지표 (Hold-out)")
    st.caption(f"평가 구간: {metrics['period']}  ·  선택 피처 {metrics['n_features']}개")
    render_metric_cards(metrics)

    st.subheader("예측 vs 실제 — Hold-out 타임라인")
    st.line_chart(df, height=380)

    with st.expander("Hold-out 예측 상세 데이터"):
        st.dataframe(df.style.format("{:.2f}"), width='stretch')


def view_stage2():
    cfg = STAGE2
    st.header("📈 Stage 2 — SK하이닉스 6개월 수익률 방향 예측")
    st.caption(
        "Stage 1의 반도체 사이클 예측을 입력 피처로 활용해 "
        "SK하이닉스 6개월 종가 수익률(방향)을 예측합니다."
    )

    try:
        metrics, df = evaluate_stage(
            cfg["features_path"], cfg["model_path"], cfg["target"],
            cfg["test_eval"], with_ic=True
        )
    except Exception as e:
        st.error(f"Stage 2 평가 중 오류가 발생했습니다: {e}")
        return

    st.subheader("모델 성능 지표 (Hold-out)")
    st.caption(f"평가 구간: {metrics['period']}  ·  사용 피처 {metrics['n_features']}개")
    render_metric_cards(metrics, with_ic=True)

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.subheader("예측 vs 실제 수익률 — Hold-out")
        st.line_chart(df, height=360)
    with col_b:
        st.subheader("방향 예측 혼동행렬")
        render_confusion(df)

    with st.expander("Hold-out 예측 상세 데이터"):
        st.dataframe(df.style.format("{:.2f}"), width='stretch')


def view_e2e():
    st.header("🔗 E2E — Stage 1 → Stage 2 파이프라인 흐름")
    st.caption(
        "Stage 1이 예측한 반도체 사이클 신호가 Stage 2의 입력 피처로 흘러들어가는 "
        "End-to-End 구조를 보여줍니다."
    )

    # ── 흐름 다이어그램 ──
    f1, fa, f2, fb, f3 = st.columns([3, 1, 3, 1, 3])
    with f1:
        st.markdown("#### 🏭 Stage 1")
        st.markdown("반도체 매출 YoY 예측\n\n`best_xgboost_final.pkl`")
    with fa:
        st.markdown("### ➡️")
    with f2:
        st.markdown(f"#### 🔌 Bridge 피처")
        st.markdown(f"Stage1 예측값\n\n`{BRIDGE_COL}`")
    with fb:
        st.markdown("### ➡️")
    with f3:
        st.markdown("#### 💹 Stage 2")
        st.markdown("SK하이닉스 수익률 예측\n\n`skh_xgb_final.pkl`")

    st.divider()

    # ── Bridge: Stage1 예측값 시계열 ──
    st.subheader("① Stage 1 출력 — Expanding Window 예측 시계열")
    st.caption(
        f"각 관찰일 시점에서 lookahead 없이 재학습해 생성한 6개월 선행 "
        f"반도체 매출 YoY 예측값(`{BRIDGE_COL}`)입니다."
    )
    try:
        s1pred = load_csv(STAGE1_PRED_PATH)
        if BRIDGE_COL in s1pred.columns:
            st.line_chart(s1pred[[BRIDGE_COL]].dropna(), height=300)
        else:
            st.warning(f"`{BRIDGE_COL}` 컬럼을 찾을 수 없습니다.")
            st.dataframe(s1pred.head(), width='stretch')
    except Exception as e:
        st.error(f"Stage 1 예측 데이터 로드 실패: {e}")
        s1pred = None

    st.divider()

    # ── 연결 검증: Stage2 피처에 Bridge 컬럼이 포함되어 있는지 ──
    st.subheader("② Stage 2 입력 — Bridge 피처 결합 확인")
    try:
        s2feat = load_csv(STAGE2["features_path"])
        if BRIDGE_COL in s2feat.columns:
            st.success(
                f"✅ Stage 2 피처셋에 Stage 1 예측 피처 `{BRIDGE_COL}`가 포함되어 있습니다. "
                "두 단계가 정상적으로 연결되었습니다."
            )
            n_total = s2feat.shape[1]
            n_bridge = sum(1 for c in s2feat.columns if c.startswith("v2_pred"))
            m1, m2 = st.columns(2)
            m1.metric("Stage 2 전체 피처 수", f"{n_total}개")
            m2.metric("Stage 1 유래 Bridge 피처", f"{n_bridge}개")
        else:
            st.warning(f"Stage 2 피처셋에서 `{BRIDGE_COL}`를 찾지 못했습니다.")
    except Exception as e:
        st.error(f"Stage 2 피처 데이터 로드 실패: {e}")

    st.divider()

    # ── 두 Stage 성능 요약 비교 ──
    st.subheader("③ 두 단계 성능 요약")
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
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)


# ──────────────────────────────────────────────────────────────────
# 6. 메인
# ──────────────────────────────────────────────────────────────────

def main():
    # 산출물 확보 (실패 시 내부에서 st.stop())
    guard_artifacts()

    st.sidebar.title("📊 대시보드")
    st.sidebar.caption("반도체 사이클 → SK하이닉스 수익률 예측")
    stage = st.sidebar.radio(
        "보기 선택",
        ["E2E 전체", "Stage 1", "Stage 2"],
        index=0,
    )
    st.sidebar.divider()
    st.sidebar.markdown(
        "**파이프라인 개요**\n\n"
        "1. **Stage 1** — 반도체 매출 YoY(6M 선행) 예측\n"
        "2. **Bridge** — 예측값을 Stage 2 피처로 전달\n"
        "3. **Stage 2** — SK하이닉스 6M 수익률 방향 예측"
    )

    st.title("반도체 사이클 기반 SK하이닉스 수익률 예측")

    if stage == "Stage 1":
        view_stage1()
    elif stage == "Stage 2":
        view_stage2()
    else:
        view_e2e()


if __name__ == "__main__":
    main()
