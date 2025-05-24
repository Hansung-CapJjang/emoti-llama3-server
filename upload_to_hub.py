from huggingface_hub import HfApi, HfFolder, create_repo, upload_folder

# ✅ Hugging Face 로그인 (토큰 저장됨)
token = "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"

# ✅ repo_id = "사용자명/리포지토리명"
repo_id = "sseyeonn/emoti-lora-ko-8b"

api = HfApi()

# ✅ (선택) 리포지토리 생성: 이미 존재하면 오류 없이 넘어감
api.create_repo(repo_id=repo_id, exist_ok=True, token=token)

# ✅ 모델 폴더 업로드
upload_folder(
    folder_path="./emoti-lora-ko-8b",  # 병합 모델 경로
    repo_id=repo_id,
    repo_type="model",
    token=token
)
