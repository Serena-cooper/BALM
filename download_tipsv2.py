from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ID = "google/tipsv2-so400m14"
SAVE_DIR = Path("./tipsv2-so400m14")

SAVE_DIR.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=MODEL_ID,
    local_dir=str(SAVE_DIR),
    local_dir_use_symlinks=False,
)

print(f"Downloaded {MODEL_ID} to {SAVE_DIR}")