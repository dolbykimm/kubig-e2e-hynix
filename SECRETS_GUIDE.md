# 🔐 GitHub Secrets 등록 가이드

> `.github/workflows/retrain.yml`(분기별 자동 재학습)이 동작하려면 아래 Secrets가 필요합니다.
> 등록 위치: **GitHub 레포 → Settings → Secrets and variables → Actions → New repository secret**

---

## 1. 등록해야 할 Secrets (필수 5개)

| Secret 이름 | 용도 | 값 예시 / 발급처 |
|-------------|------|------------------|
| `FRED_API_KEY` | 파이프라인의 FRED 거시지표 수집 | FRED 발급 키 (https://fredaccount.stlouisfed.org/apikeys) |
| `AWS_ACCESS_KEY_ID` | S3 업로드용 AWS 자격증명 | `AKIA...` (IAM 사용자 키) |
| `AWS_SECRET_ACCESS_KEY` | S3 업로드용 AWS 시크릿 | AWS 시크릿 |
| `S3_BUCKET_NAME` | 산출물 업로드 대상 버킷 | `kubig-e2e-hynix-models` |
| `RAILWAY_TOKEN` | 재학습 후 Railway 재배포 | Railway 토큰 (아래 2-③ 참고) |

---

## 2. 발급/등록 방법

### ① FRED_API_KEY
- https://fredaccount.stlouisfed.org/apikeys → 로그인 → **Request API Key** → 32자리 키 복사

### ② AWS 자격증명 (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)
- AWS Console → IAM → 사용자 → **보안 자격 증명** → **액세스 키 만들기**
- S3 `kubig-e2e-hynix-models` 버킷에 `s3:PutObject` 권한이 있어야 함
- (권장) 최소 권한 정책 예:
  ```json
  {
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Action": ["s3:PutObject"],
      "Resource": "arn:aws:s3:::kubig-e2e-hynix-models/*"
    }]
  }
  ```

### ③ RAILWAY_TOKEN
- Railway → 프로젝트 → **Settings → Tokens** 에서 **Project Token** 발급 (권장: 프로젝트 토큰)
- 또는 Account → **Create Token** (계정 토큰)
- 워크플로우의 `railway redeploy --service kubig-e2e-hynix` 가 이 토큰으로 인증함
- ⚠️ Railway 서비스 이름이 실제로 `kubig-e2e-hynix`인지 확인 (다르면 `retrain.yml`의 `--service` 값 수정 필요)

---

## 3. 참고: 선택적 Secret

| Secret 이름 | 필요 시점 |
|-------------|-----------|
| `AWS_DEFAULT_REGION` | S3 버킷 리전 문제로 업로드가 실패할 경우 추가 (예: `ap-northeast-2`). 현재 `upload_to_s3.py`는 리전 미지정으로 동작하지만, 리전 의존 오류가 나면 등록하고 워크플로우 `env`에 추가하세요. |

---

## 4. 등록 후 확인

1. 레포 → **Actions** 탭 → **분기별 자동 재학습** 워크플로우 선택
2. **Run workflow** (workflow_dispatch)로 수동 실행해 테스트
   - gate 잡이 "수동 트리거 — 실행합니다." 출력 후 retrain 잡 진행
3. 스케줄 실행은 매년 **1·4·7·10월 넷째 주 금요일 09:00 KST**에 자동 발화
   - (cron은 매월 22~28일 발화하지만 gate에서 금요일만 통과)

---

## 5. 보안 메모
- Secrets는 로그에 마스킹되어 출력됩니다(노출 방지).
- 키 유출이 의심되면 즉시 해당 서비스(AWS/Railway/FRED)에서 **재발급(rotate)** 후 Secret 값을 갱신하세요.
- `.env`(로컬용 실제 값)는 절대 커밋하지 마세요 — 이미 `.gitignore`에 포함되어 있습니다.
