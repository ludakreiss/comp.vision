import io
import random
import math
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import config
from transforms import get_transforms

class DeepfakeImageDataset(Dataset):
    def __init__(self, dataframe, transform, image_modifier=None):
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.transform = transform
        self.image_modifier = image_modifier

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_path = row["image_path"]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as error:
            raise RuntimeError(f"Could not load image: {image_path}") from error

        if self.image_modifier is not None:
            image = self.image_modifier(image)

        image = self.transform(image)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)

        return {
            "image": image,
            "label": label,
            "path": image_path,
            "video_id": row["video_id"],
        }

def assign_group_splits(dataframe, train_ratio=None, validation_ratio=None, seed=42):
    """Assign train/val/test splits with stratification by group-level majority label.

    Groups are bucketed into 'fake-dominant' (>=50% fake frames) and 'real-dominant'
    (<50% fake frames). Each bucket is independently shuffled and split by ratio,
    preventing the class-imbalanced splits that arise from purely count-based shuffling.

    Raises ValueError instead of silently duplicating groups across splits on fallback.
    """
    if train_ratio is None:
        train_ratio = getattr(config, "TRAIN_RATIO", 0.70)
    if validation_ratio is None:
        validation_ratio = getattr(config, "VAL_RATIO", 0.15)
    dataframe = dataframe.copy()
    if "group_id" not in dataframe.columns and "video_id" in dataframe.columns:
        # Extract base source video ID prefix (e.g. '000' from '000_003.mp4' or '000') to prevent leakage
        def extract_source_id(vid):
            s = str(vid).split('.')[0]
            parts = s.split('_')
            if len(parts) > 1 and parts[0].isdigit():
                return parts[0]
            return s
        dataframe["group_id"] = dataframe["video_id"].map(extract_source_id)

    # Stratify groups by their majority label to prevent class-imbalanced splits.
    # Compute per-group fake ratio: groups with fake_ratio >= 0.5 are "fake-dominant".
    group_fake_ratio = (
        dataframe.groupby("group_id")["label"]
        .mean()
        .rename("fake_ratio")
        .reset_index()
    )
    fake_dominant = sorted(
        group_fake_ratio.loc[group_fake_ratio["fake_ratio"] >= 0.5, "group_id"].astype(str).tolist()
    )
    real_dominant = sorted(
        group_fake_ratio.loc[group_fake_ratio["fake_ratio"] < 0.5, "group_id"].astype(str).tolist()
    )

    rng = np.random.default_rng(seed)
    rng.shuffle(fake_dominant)
    rng.shuffle(real_dominant)

    def _split_bucket(bucket, train_r, val_r):
        """Split a list of group IDs into (train, val, test) sub-lists by ratio.
        Returns empty lists for splits that cannot be populated (small buckets)."""
        n = len(bucket)
        if n == 0:
            return [], [], []
        if n == 1:
            return list(bucket), [], []
        if n == 2:
            return [bucket[0]], [bucket[1]], []
        train_end = max(1, int(n * train_r))
        val_end = train_end + max(1, int(n * val_r))
        if val_end >= n:
            val_end = n - 1
            train_end = max(1, val_end - 1)
        return bucket[:train_end], bucket[train_end:val_end], bucket[val_end:]

    fake_train, fake_val, fake_test = _split_bucket(fake_dominant, train_ratio, validation_ratio)
    real_train, real_val, real_test = _split_bucket(real_dominant, train_ratio, validation_ratio)

    train_groups = set(fake_train + real_train)
    validation_groups = set(fake_val + real_val)
    test_groups = set(fake_test + real_test)

    # Safety check: no group_id should appear in more than one split
    overlap_tv = train_groups & validation_groups
    overlap_tt = train_groups & test_groups
    overlap_vt = validation_groups & test_groups
    if overlap_tv or overlap_tt or overlap_vt:
        raise ValueError(
            f"Split leakage detected — overlapping group_ids: "
            f"train∩val={overlap_tv}, train∩test={overlap_tt}, val∩test={overlap_vt}"
        )
    if not validation_groups:
        raise ValueError(
            "Validation split is empty after stratified group assignment. "
            "The dataset has too few unique source videos to produce three non-empty splits."
        )
    if not test_groups:
        raise ValueError(
            "Test split is empty after stratified group assignment. "
            "The dataset has too few unique source videos to produce three non-empty splits."
        )

    def map_split(group_id):
        group_id = str(group_id)
        if group_id in train_groups:
            return "train"
        if group_id in validation_groups:
            return "val"
        if group_id in test_groups:
            return "test"
        raise ValueError(f"Unknown group_id '{group_id}' not assigned to any split.")

    dataframe["split"] = dataframe["group_id"].map(map_split)
    return dataframe


def get_dataloaders(manifest_df):
    # Ensure splits exist in manifest
    if "split" not in manifest_df.columns:
        manifest_df = assign_group_splits(manifest_df, seed=config.SEED)

    train_df = manifest_df[manifest_df["split"] == "train"].copy()
    val_df = manifest_df[manifest_df["split"] == "val"].copy()
    test_df = manifest_df[manifest_df["split"] == "test"].copy()

    # Apply DataFrame-level oversampling or undersampling if configured
    sampler = None
    if len(train_df) > 0 and train_df["label"].nunique() > 1:
        if config.BALANCING_STRATEGY == "oversampling":
            class_counts = train_df["label"].value_counts()
            max_size = class_counts.max()
            lst = []
            for class_label, group in train_df.groupby("label"):
                lst.append(group.sample(max_size, replace=True, random_state=config.SEED))
            train_df = pd.concat(lst, ignore_index=True)
            train_df = train_df.sample(frac=1.0, random_state=config.SEED).reset_index(drop=True)
        elif config.BALANCING_STRATEGY == "undersampling":
            class_counts = train_df["label"].value_counts()
            min_size = class_counts.min()
            lst = []
            for class_label, group in train_df.groupby("label"):
                lst.append(group.sample(min_size, replace=False, random_state=config.SEED))
            train_df = pd.concat(lst, ignore_index=True)
            train_df = train_df.sample(frac=1.0, random_state=config.SEED).reset_index(drop=True)
        elif config.BALANCING_STRATEGY == "sampler":
            class_counts = train_df["label"].astype(int).value_counts().to_dict()
            sample_weights = train_df["label"].astype(int).map(lambda label: 1.0 / class_counts[label]).to_numpy()
            sampler = WeightedRandomSampler(
                weights=torch.tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True
            )

    train_transform, eval_transform = get_transforms()

    train_dataset = DeepfakeImageDataset(train_df, train_transform)
    val_dataset = DeepfakeImageDataset(val_df, eval_transform)
    test_dataset = DeepfakeImageDataset(test_df, eval_transform)

    def worker_init_fn(worker_id):
        """Ensure each DataLoader worker uses unique random seeds for augmentations."""
        seed = config.SEED + worker_id
        random.seed(seed)
        np.random.seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(config.NUM_WORKERS > 0),
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(config.NUM_WORKERS > 0),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(config.NUM_WORKERS > 0),
    )

    return train_loader, val_loader, test_loader
