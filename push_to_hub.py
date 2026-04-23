from huggingface_hub import HfApi, login
from dotenv import load_dotenv
import os

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Login
login(token=HF_TOKEN)

api = HfApi()

# Create repo
api.create_repo(
    repo_id="harsh2five/mistral-fastapi-finetuned",
    repo_type="model",
    private=False,
    exist_ok=True
)

# Upload adapter files
api.upload_folder(
    folder_path="./model",
    repo_id="harsh2five/mistral-fastapi-finetuned",
    repo_type="model"
)

print("Model pushed to: huggingface.co/harsh2five/mistral-fastapi-finetuned")