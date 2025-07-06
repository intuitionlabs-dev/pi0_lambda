from huggingface_hub import snapshot_download

# Set your repo_id (e.g., "username/dataset")
repo_id = "intuitioncore/piper_pick_and_place_full_fine-tune_5_new_recorder"

# Set the local directory where you want to download the files
local_dir = "/workspace/pi0_lambda/pi0_replicate/checkpoints/pi0_single_agilex_full"  # <-- CHANGE THIS

# Number of parallel download workers (increase for more speed)
num_threads = 8

# Download the entire repo (or specify a subfolder with 'repo_dir')
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    repo_type="model",  # or "model" if downloading a model
    max_workers=num_threads,
    allow_patterns=["piper_single_pick_place_full_finetune/3000/*"],  # Only download this folder and its contents
    resume_download=True,  # Resume partial downloads
    local_dir_use_symlinks=False  # Set True to save space, False to copy files
)

print(f"Download complete! Files are in {local_dir}")