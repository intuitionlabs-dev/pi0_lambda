from pathlib import Path
from huggingface_hub import list_repo_files, hf_hub_download

REPO_ID   = "intuitioncore/piper_pick_and_place"
REPO_TYPE = "dataset"
OUT_DIR   = Path("/home/intuition/pi0_test/pi0_lambda/pi0_replicate/data/piper_pick_and_place")

# Helper: list and keep only files that start with a given prefix
def first_n(prefix: str, n: int):
    files = [f for f in repo_files if f.startswith(prefix)]
    return sorted(files)[:n]

# 1) Ask the Hub for every filename in the repo (cheap – just metadata)
repo_files = list_repo_files(REPO_ID, repo_type=REPO_TYPE)

# 2) Build the exact subset you want
wanted = set()

# a) first 3 files in data/chunk-000/
wanted.update(first_n("data/chunk-000/", 3))

# b) everything under meta/
wanted.update([f for f in repo_files if f.startswith("meta/")])

# c) first 3 workspace images
wanted.update(first_n("videos/chunk-000/workspace_image", 3))

# d) first 3 wrist images
wanted.update(first_n("videos/chunk-000/wrist_image", 3))

# 3) Download them one-by-one, preserving paths
for rel_path in wanted:
    hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=rel_path,        # remote path
        local_dir=OUT_DIR,         # local root
        local_dir_use_symlinks=False,
        force_download=False,      # skip if it already exists
    )
print(f"Done – {len(wanted)} files now under {OUT_DIR}")