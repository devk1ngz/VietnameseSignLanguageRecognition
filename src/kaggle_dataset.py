import os
import shutil
from pathlib import Path
import kagglehub

# Define project root and destination data directory
root_dir = Path(__file__).resolve().parent.parent
dest_dir = root_dir / "data_more"

# Use a temporary cache directory inside data_more to avoid storing files in ~/.cache (home directory)
temp_cache = dest_dir / ".cache"
temp_cache.mkdir(parents=True, exist_ok=True)
os.environ["KAGGLEHUB_CACHE"] = str(temp_cache)

# Download the latest version of the dataset
print("Downloading dataset via kagglehub...")
cache_path = kagglehub.dataset_download("aresusayhi/vsl-vietnamese-sign-languages")
print("Downloaded to cache at:", cache_path)

# Move files from cache_path directly to dest_dir
print(f"Moving files directly to {dest_dir}...")
for item in os.listdir(cache_path):
    s = os.path.join(cache_path, item)
    d = os.path.join(dest_dir, item)
    # If the destination already exists, remove it first
    if os.path.exists(d):
        if os.path.isdir(d):
            shutil.rmtree(d)
        else:
            os.remove(d)
    shutil.move(s, d)

# Clean up the temporary cache directory
try:
    shutil.rmtree(temp_cache)
except Exception as e:
    print(f"Note: Could not fully remove temporary cache directory: {e}")

print(f"Dataset successfully downloaded and saved directly to: {dest_dir}")