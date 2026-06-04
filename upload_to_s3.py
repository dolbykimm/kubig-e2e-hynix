"""
upload_to_s3.py — 모델/데이터 산출물을 S3에 업로드
================================================================
Stage 1·2 파이프라인이 생성한 산출물 5개를 S3 버킷에 업로드한다.
S3 키 구조는 로컬 상대경로를 그대로 사용한다.
  예) stage1/outputs/models/best_xgboost_final.pkl

환경변수:
  S3_BUCKET_NAME        (필수) 업로드 대상 버킷명
  AWS_ACCESS_KEY_ID     (필수) AWS 자격증명
  AWS_SECRET_ACCESS_KEY (필수) AWS 자격증명

실행:
  python upload_to_s3.py
"""

import os
import sys

# 업로드 대상 (로컬 상대경로 == S3 key)
ARTIFACTS = [
    "stage1/outputs/models/best_xgboost_final.pkl",
    "stage1/outputs/data/features_dataset.csv",
    "stage2/outputs/models/skh_xgb_final.pkl",
    "stage2/outputs/data/stage2_features.csv",
    "stage2/outputs/data/stage1_predictions.csv",
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    bucket = os.getenv("S3_BUCKET_NAME")
    if not bucket:
        print("[오류] 환경변수 S3_BUCKET_NAME이 설정되지 않았습니다.")
        sys.exit(1)

    if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
        print("[오류] AWS 자격증명(AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)이 없습니다.")
        sys.exit(1)

    try:
        import boto3
        from botocore.exceptions import ClientError, BotoCoreError, NoCredentialsError
    except ImportError:
        print("[오류] boto3가 설치되어 있지 않습니다. pip install boto3")
        sys.exit(1)

    # ── 업로드 전 파일 존재 여부 확인 ──
    missing = [k for k in ARTIFACTS if not os.path.exists(os.path.join(BASE_DIR, k))]
    if missing:
        print("[오류] 업로드할 산출물이 없습니다. 파이프라인을 먼저 실행하세요:")
        for k in missing:
            print(f"  ✗ {k}")
        sys.exit(1)

    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    print("=" * 64)
    print(f"  S3 업로드 → 버킷: {bucket}")
    print("=" * 64)

    results = []
    for key in ARTIFACTS:
        local_path = os.path.join(BASE_DIR, key)
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        try:
            s3.upload_file(local_path, bucket, key)
            print(f"  ✓ 성공  {key}  ({size_mb:.2f} MB)")
            results.append((key, True, None))
        except (ClientError, BotoCoreError, NoCredentialsError) as e:
            print(f"  ✗ 실패  {key}  — {e}")
            results.append((key, False, str(e)))

    ok = sum(1 for _, success, _ in results if success)
    print("=" * 64)
    print(f"  결과: {ok}/{len(ARTIFACTS)} 성공")
    print("=" * 64)

    if ok != len(ARTIFACTS):
        sys.exit(1)


if __name__ == "__main__":
    main()
