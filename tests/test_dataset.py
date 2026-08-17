"""
Unit tests for Dataset splitting logic to guarantee zero video_id / group_id leakage.
"""

import pytest
import pandas as pd
import numpy as np
from dataset import assign_group_splits

def test_zero_group_leakage_across_splits():
    """Verify assign_group_splits guarantees zero group_id leakage across train, val, and test splits."""
    # Generate synthetic video manifest with 50 source video groups
    records = []
    for g in range(50):
        group_id = f"video_{g:03d}"
        for f in range(20):  # 20 frames per video
            records.append({
                "image_path": f"/tmp/frames/{group_id}_frame_{f:02d}.jpg",
                "video_id": group_id,
                "label": g % 2
            })
            
    df = pd.DataFrame(records)
    
    # Run assign_group_splits across multiple random seeds
    for seed in [42, 100, 2026, 999]:
        split_df = assign_group_splits(df, train_ratio=0.70, validation_ratio=0.15, seed=seed)
        
        train_groups = set(split_df[split_df["split"] == "train"]["group_id"])
        val_groups = set(split_df[split_df["split"] == "val"]["group_id"])
        test_groups = set(split_df[split_df["split"] == "test"]["group_id"])
        
        # Assert mutually disjoint sets (Zero leakage)
        assert len(train_groups.intersection(val_groups)) == 0, f"Leakage between train and val on seed {seed}"
        assert len(train_groups.intersection(test_groups)) == 0, f"Leakage between train and test on seed {seed}"
        assert len(val_groups.intersection(test_groups)) == 0, f"Leakage between val and test on seed {seed}"

def test_split_ratios():
    """Verify assign_group_splits correctly approximates 70% / 15% / 15% group allocation."""
    records = [{"video_id": f"vid_{i}", "label": i % 2} for i in range(100)]
    df = pd.DataFrame(records)
    
    split_df = assign_group_splits(df, train_ratio=0.70, validation_ratio=0.15, seed=42)
    group_counts = split_df.groupby("split")["group_id"].nunique()
    
    assert group_counts["train"] == 70
    assert abs(group_counts["val"] - 15) <= 1
    assert abs(group_counts["test"] - 15) <= 1
