from huggingface_hub import HfApi
api = HfApi()
REPO = "salam0z/Real-Time-Sign-Language-Recognition"        # your HF username
api.create_repo(REPO, repo_type="space", space_sdk="static", exist_ok=True)
api.upload_folder(folder_path=r"C:\Users\ahmed\OneDrive\Documents\project4\hf_static",
                  repo_id=REPO, repo_type="space")
print("→ https://huggingface.co/spaces/" + REPO)