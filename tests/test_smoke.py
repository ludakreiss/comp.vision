"""
End-to-end smoke test: invokes the real train.py CLI as a subprocess against a fully
isolated, synthetic manifest - exercising the actual training entry point (argument
parsing, manifest loading, dataloaders, model build, one real optimizer step, checkpoint
save) without touching any real dataset or default config paths.
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402
from model import build_model  # noqa: E402
from train import set_seed  # noqa: E402


def test_end_to_end_smoke(tmp_path):
    # 1. Create a tiny synthetic manifest with consistent per-video labels and disjoint train/val/test groups
    data_dir = tmp_path / "smoke_data"
    data_dir.mkdir()

    video_specs = [
        ("vid_0", 0, "group_train"), ("vid_1", 1, "group_train"),
        ("vid_2", 0, "group_train"), ("vid_3", 1, "group_train"),
        ("vid_4", 0, "group_val"), ("vid_5", 1, "group_val"),
        ("vid_6", 0, "group_test"), ("vid_7", 1, "group_test"),
    ]
    split_for_group = {"group_train": "train", "group_val": "val", "group_test": "test"}

    rows = []
    img_idx = 0
    for video_id, label, group_id in video_specs:
        for _ in range(2):
            img_path = data_dir / f"img_{img_idx}.jpg"
            Image.new("RGB", (32, 32), color="red").save(img_path)
            rows.append({
                "image_path": str(img_path),
                "video_id": video_id,
                "label": label,
                "group_id": group_id,
                "split": split_for_group[group_id],
            })
            img_idx += 1

    manifest_path = tmp_path / "smoke_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    output_root = tmp_path / "outputs"
    env = os.environ.copy()
    env["OUTPUT_ROOT"] = str(output_root)
    env["PYTHONPATH"] = str(REPO_ROOT)

    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "train.py"),
            "--manifest", str(manifest_path),
            "--limit_batches", "1",
            "--epochs", "1",
            "--batch_size", "2",
            "--num_workers", "0",
            "--no-pretrained",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=300,
    )

    assert result.returncode == 0, f"Smoke test failed. Output:\n{result.stdout}\nError:\n{result.stderr}"
    assert "Training completed" in result.stdout
