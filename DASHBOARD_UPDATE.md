# 🎨 대시보드 개편 요약 (DASHBOARD_UPDATE)

> 작성일: 2026-06-05
> 커밋: `feat: 대시보드 개편 - 방향예측 메인/SHAP/적중률 히스토리`

---

## 전체 방향
- **메인**: 방향 예측(📈 상승 / 📉 하락)을 크고 명확하게
- **부가**: 예측 수익률 수치는 작은 caption으로
- **알파**: SHAP 피처 중요도 + 과거 적중률 히스토리 + 현재 시장 신호

---

## 변경 내역

### 공통 / 신규 헬퍼
| 함수 | 역할 |
|------|------|
| `render_direction_headline()` | 최신 예측 방향을 색상 배경 박스로 **크게** 표시, 수익률 수치는 작은 caption |
| `compute_shap_importance()` | `shap.TreeExplainer`로 XGBoost 설명 → 평균 \|SHAP\| 상위 10 피처 (cache_data) |
| `render_shap_section()` | SHAP 중요도를 `st.bar_chart(horizontal=True)`로 시각화 (matplotlib 미사용) |
| `render_hit_history()` | hold-out 구간 예측방향 vs 실제방향 → ✅/❌ 타임라인 + 적중률 metric + 상세표 |
| `get_market_momentum()` | yfinance로 KOSPI(^KS11)/SOX(^SOX) 3개월 모멘텀 계산 (ttl=1h 캐시) |
| `get_trends_latest()` | stage2_features에서 Google Trends 류 컬럼 탐색 (없으면 N/A) |
| `render_market_signals()` | E2E ④ 시장 신호 요약 카드 + 종합 신호(다수결) |
| `_flow_box()` / `_flow_arrow()` | E2E 흐름 다이어그램용 테두리 컨테이너 + 화살표 |

### Stage 1 화면
- 상단에 **방향 예측 헤드라인**(📈/📉) 추가, 예측 YoY 수치는 작은 caption
- 기존(성능지표·타임라인·상세표) 유지
- **알파 섹션 추가**: ① SHAP 상위 10 피처 바차트, ② 과거 월별 적중 히스토리

### Stage 2 화면
- 상단에 **방향 예측 헤드라인** 추가, 예측 수익률 수치는 작은 caption
- 기존(성능지표·타임라인·혼동행렬·상세표) 유지
- **알파 섹션 추가**: ① SHAP 상위 10 피처 바차트, ② 과거 분기별 적중 히스토리

### E2E 전체 화면
- 흐름 다이어그램 개선: 단순 텍스트 컬럼 → **`st.container(border=True)` 박스 + 큰 화살표(➡️)**
- 기존 ①~③ (Stage1 출력 시계열 / Bridge 결합 검증 / 성능 요약) 유지
- **④ 현재 시장 신호 요약 카드 추가**:
  - 🔎 Google Trends 최신값 (데이터셋에 컬럼 있으면 표시, 없으면 N/A)
  - 🇰🇷 KOSPI 3개월 모멘텀
  - 💽 SOX 3개월 모멘텀
  - 🧭 종합 신호 = (KOSPI·SOX 모멘텀 방향 + Stage 2 모델 최신 예측 방향) 다수결

---

## 기술 조건 준수
- ✅ SHAP: `shap.TreeExplainer` + `st.bar_chart` (matplotlib 미사용 → 한글 폰트 깨짐 없음)
- ✅ 적중률: `evaluate_stage`의 hold-out 결과로 날짜별 방향 비교
- ✅ 시장 신호: yfinance `^KS11`, `^SOX` 3개월 모멘텀 = 현재가/3개월전가 − 1
- ✅ 캐싱 구조 유지: `cache_resource`(모델), `cache_data`(데이터·평가·SHAP·모멘텀)
- ✅ 한국어 UI 유지
- ✅ `use_container_width` 미사용 → 전부 `width='stretch'`

---

## ⚠️ 참고 사항
1. **Google Trends 피처 부재**: 현재 `stage2_features.csv`에는 Google Trends 컬럼이 없습니다.
   `get_trends_latest()`가 `trend`/`google_trends` 류 컬럼을 자동 탐색하지만, 없으면 카드에 **"N/A (데이터셋에 Trends 피처 없음)"** 으로 정직하게 표기됩니다.
   추후 파이프라인에 Trends 피처를 추가하면 자동으로 값이 표시됩니다.
2. **방향 헤드라인 기준**: hold-out 구간의 **가장 최근 예측 시점**을 "현재 신호"로 사용합니다.
3. **의존성 추가**: `stage1/requirements.txt`에 `shap>=0.44` 추가 (Docker 재빌드 시 자동 설치).

---

## 검증 결과 (로컬 스모크 테스트)
- `python -m py_compile app.py` → 통과
- SHAP 계산: Stage1(385×35), Stage2(106×25) 정상 — 피처 정렬 오류 없음
- yfinance 모멘텀: KOSPI/SOX 정상 수신
- (실제 배포 반영하려면 Docker 이미지 재빌드 + Docker Hub 재푸시 필요)
