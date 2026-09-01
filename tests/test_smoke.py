import pytest
import subprocess
import os
import sys
import pandas as pd
from pathlib import Path

def test_end_to_end_smoke():
    # 1. Create a tiny synthetic manifest
    manifest_path = "smoke_manifest.csv"
    os.makedirs("smoke_data", exist_ok=True)
    
    # Create dummy images
    from PIL import Image
    for i in range(10):
        img = Image.new("RGB", (32, 32), color="red")
        img.save(f"smoke_data/img_{i}.jpg")
        
    df = pd.DataFrame({
        "image_path": [f"smoke_data/img_{i}.jpg" for i in range(10)],
        "video_id": [f"vid_{i//2}" for i in range(10)],
        "label": [i % 2 for i in range(10)],
        "group_id": [f"group_{i//4}" for i in range(10)]
    })
    
    # Assign splits manually to avoid leakage issues with tiny dataset
    df["split"] = ["train"]*6 + ["val"]*2 + ["test"]*2
    # Ensure splits have different groups
    df.loc[0:5, "group_id"] = "group_train"
    df.loc[6:7, "group_id"] = "group_val"
    df.loc[8:9, "group_id"] = "group_test"
    df.to_csv(manifest_path, index=False)
    
    # 2. Run train.py with --limit_batches 1 --epochs 1
    result = subprocess.run([
        sys.executable, "train.py",
        "--limit_batches", "1",
        "--epochs", "1",
        "--dataset", manifest_path
    ], capture_output=True, text=True)
    
    # Cleanup
    import shutil
    if os.path.exists("smoke_data"):
        shutil.rmtree("smoke_data")
    if os.path.exists(manifest_path):
        os.remove(manifest_path)
        
    assert result.returncode == 0, f"Smoke test failed. Output:\n{result.stdout}\nError:\n{result.stderr}"
    assert "Training completed" in result.stdout
