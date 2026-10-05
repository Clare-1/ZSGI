"""Download Qwen3-4B model files via proxy using requests."""
import os
import sys
import requests
from tqdm import tqdm

PROXIES = {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}

MODEL_ID = "Qwen/Qwen3-4B"
CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")
MODEL_DIR = os.path.join(CACHE_DIR, f"models--{MODEL_ID.replace('/', '--')}")
SNAPSHOT_DIR = os.path.join(MODEL_DIR, "snapshots")

# Files to download
FILES = [
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
    "model.safetensors.index.json",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
]

def download_file(url, dest_path, proxies):
    """Download a file with progress bar."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    # Check if file already exists and is complete
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"  Already exists: {os.path.basename(dest_path)} ({os.path.getsize(dest_path)//1024//1024} MB)")
        return
    
    resp = requests.get(url, proxies=proxies, stream=True, verify=False, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    
    with open(dest_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=os.path.basename(dest_path)
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

def main():
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # First get the snapshot commit hash
    api_url = f"https://huggingface.co/api/models/{MODEL_ID}"
    print("Fetching model info...")
    resp = requests.get(api_url, proxies=PROXIES, verify=False, timeout=30)
    resp.raise_for_status()
    info = resp.json()
    commit_hash = info["sha"]
    print(f"Commit: {commit_hash}")
    
    snapshot_path = os.path.join(SNAPSHOT_DIR, commit_hash)
    os.makedirs(snapshot_path, exist_ok=True)
    
    base_url = f"https://huggingface.co/{MODEL_ID}/resolve/{commit_hash}"
    
    for fname in FILES:
        url = f"{base_url}/{fname}"
        dest = os.path.join(snapshot_path, fname)
        print(f"\nDownloading {fname}...")
        download_file(url, dest, PROXIES)
    
    print(f"\nAll files downloaded to: {snapshot_path}")

if __name__ == "__main__":
    main()
