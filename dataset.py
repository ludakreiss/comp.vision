import io
import random
import math
from pathlib import Path
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
            try:
                image = self.image_modifier(image, image_path=str(image_path))
            except TypeError:
                image = self.image_modifier(image)

        image = self.transform(image)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)

        return {
            "image": image,
            "label": label,
            "path": image_path,
            "video_id": row["video_id"],
        }


class UnionFind:
    """Disjoint-set data structure for grouping connected video components."""
    def __init__(self):
        self.parent = {}

    def find(self, item):
        if item not in self.parent:
            self.parent[item] = item
            return item
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, item1, item2):
        root1 = self.find(item1)
        root2 = self.find(item2)
        if root1 != root2:
            self.parent[root1] = root2


def build_connected_groups(video_ids):
    """Build connected components from video IDs (e.g., '000_003' connects '000' and '003', 'id0_id1_0000' connects 'id0' and 'id1')."""
    uf = UnionFind()
    vids = set()

    for vid in video_ids:
        s = str(vid).split('.')[0]
        parts = s.split('_')
        vids.add(s)

        if len(parts) >= 2:
            id1, id2 = parts[0], parts[1]
            is_ffpp = id1.isdigit() and id2.isdigit()
            is_celebdf_pair = (id1.startswith('id') and id1[2:].isdigit()) and (id2.startswith('id') and id2[2:].isdigit())
            if is_ffpp or is_celebdf_pair:
                uf.union(id1, id2)
                uf.union(s, id1)
                uf.union(s, id2)
            elif id1.startswith('id') and id1[2:].isdigit():
                uf.union(s, id1)
            else:
                uf.find(s)
        elif len(parts) == 1 and parts[0].startswith('id') and parts[0][2:].isdigit():
            uf.union(s, parts[0])
        else:
            uf.find(s)

    roots = {}
    group_map = {}
    group_counter = 0

    for vid in sorted(vids):
        parts = vid.split('_')
        is_ffpp = len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit()
        is_celebdf_pair = len(parts) >= 2 and (parts[0].startswith('id') and parts[0][2:].isdigit()) and (parts[1].startswith('id') and parts[1][2:].isdigit())
        is_celebdf_single = len(parts) >= 1 and parts[0].startswith('id') and parts[0][2:].isdigit()
        base = parts[0] if (is_ffpp or is_celebdf_pair or is_celebdf_single) else vid
        root = uf.find(base)
        if root not in roots:
            roots[root] = f"group_{group_counter:04d}"
            group_counter += 1
        group_map[vid] = roots[root]

    return group_map


def assign_group_splits(dataframe, train_ratio=None, validation_ratio=None, seed=42):
    """Assign train/val/test splits with stratification by group-level majority label.

    Uses pre-computed group_id if present; otherwise computes robust graph connected-components.
    """
    if train_ratio is None:
        train_ratio = getattr(config, "TRAIN_RATIO", 0.70)
    if validation_ratio is None:
        validation_ratio = getattr(config, "VAL_RATIO", 0.15)
    dataframe = dataframe.copy()

    if "group_id" not in dataframe.columns and "video_id" in dataframe.columns:
        group_map = build_connected_groups(dataframe["video_id"].unique())
        dataframe["group_id"] = dataframe["video_id"].map(
            lambda v: group_map.get(str(v).split('.')[0], str(v))
        )

    # Stratify groups by majority label
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
        n = len(bucket)
        if n == 0:
            return [], [], []
        if n < 3:
            raise ValueError(f"Cannot create disjoint train/val/test split from only {n} group(s). Dataset has too few unique source video groups.")
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
            "Dataset has too few unique source videos to produce three non-empty splits."
        )
    if not test_groups:
        raise ValueError(
            "Test split is empty after stratified group assignment. "
            "Dataset has too few unique source videos to produce three non-empty splits."
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


def validate_celebdf_manifest(df_or_path):
    """Validate that a DataFrame or CSV file path represents a genuine Celeb-DF manifest across all rows."""
    if isinstance(df_or_path, (str, Path)):
        path = Path(df_or_path)
        if not path.exists() or path.stat().st_size == 0:
            return False, f"Celeb-DF manifest file does not exist or is empty: {path}"
        try:
            df = pd.read_csv(path)
        except Exception as e:
            return False, f"Failed to read Celeb-DF manifest CSV ({e}): {path}"
    else:
        df = df_or_path

    if df is None or len(df) == 0:
        return False, "Celeb-DF manifest is empty."

    required_cols = {"image_path", "video_id", "label"}
    if not required_cols.issubset(df.columns):
        return False, f"Celeb-DF manifest missing required columns: {required_cols - set(df.columns)}"

    sample_paths = df["image_path"].astype(str).tolist()
    for p in sample_paths:
        if "ffpp_c23" in p or "processed_faces/ffpp" in p or "ffpp_c40" in p:
            return False, f"Manifest contains FaceForensics++ paths instead of Celeb-DF: {p}"

    if "category" in df.columns:
        cats = set(df["category"].dropna().unique())
        valid_celebdf_cats = {"Celeb-real", "Celeb-synthesis", "YouTube-real"}
        if cats and not cats.intersection(valid_celebdf_cats):
            return False, f"Manifest categories {cats} do not match Celeb-DF categories {valid_celebdf_cats}"

    return True, "Valid Celeb-DF manifest."


def validate_manifest(df_or_path):
    """Validate full training/evaluation dataset manifest for structural integrity, split isolation, and labels."""
    if isinstance(df_or_path, (str, Path)):
        path = Path(df_or_path)
        if not path.exists() or path.stat().st_size == 0:
            return False, f"Manifest file does not exist or is empty: {path}"
        try:
            df = pd.read_csv(path)
        except Exception as e:
            return False, f"Failed to read manifest CSV ({e}): {path}"
    else:
        df = df_or_path

    if df is None or len(df) == 0:
        return False, "Manifest is empty."

    required_cols = {"image_path", "video_id", "label", "group_id"}
    if not required_cols.issubset(df.columns):
        return False, f"Manifest missing required columns: {required_cols - set(df.columns)}"

    labels = set(df["label"].dropna().unique())
    if not labels.issubset({0, 1, 0.0, 1.0}):
        return False, f"Manifest labels contain invalid values: {labels - {0, 1, 0.0, 1.0}}"

    if "split" in df.columns:
        train_grps = set(df[df["split"] == "train"]["group_id"].astype(str))
        val_grps = set(df[df["split"] == "val"]["group_id"].astype(str))
        test_grps = set(df[df["split"] == "test"]["group_id"].astype(str))

        if train_grps.intersection(val_grps):
            return False, f"Group leakage detected between train and val splits."
        if train_grps.intersection(test_grps):
            return False, f"Group leakage detected between train and test splits."
        if val_grps.intersection(test_grps):
            return False, f"Group leakage detected between val and test splits."

    return True, "Valid manifest."


def generate_celebdf_manifest(celebdf_root=None, output_path=None):
    """Canonical Celeb-DF manifest generator with connected-component identity grouping."""
    if celebdf_root is None:
        celebdf_root = getattr(config, "CELEBDF_ROOT", Path("datasets/Celeb-DF-v2"))
    else:
        celebdf_root = Path(celebdf_root)

    if output_path is None:
        output_path = getattr(config, "CELEBDF_MANIFEST_PATH", Path("deepfake_robustness/celebdf_manifest.csv"))
    else:
        output_path = Path(output_path)

    test_list_path = celebdf_root / "List_of_testing_videos.txt"
    test_videos = set()
    if test_list_path.exists():
        with open(test_list_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    test_videos.add(parts[1].strip())

    records = []
    image_extensions = {".jpg", ".jpeg", ".png"}

    for category_dir in celebdf_root.iterdir():
        if not category_dir.is_dir() or category_dir.name.startswith("."):
            continue
        label = 0 if category_dir.name in ["Celeb-real", "YouTube-real"] else 1
        category_name = category_dir.name

        for item in category_dir.rglob("*"):
            if item.is_file() and item.suffix.lower() in image_extensions:
                rel_video_path = f"{category_name}/{item.parent.name}.mp4"
                split = "test" if rel_video_path in test_videos else "train"
                vid_name = item.parent.name
                records.append({
                    "image_path": str(item.resolve()),
                    "label": label,
                    "video_id": vid_name,
                    "split": split,
                    "dataset": "Celeb-DF-v2",
                })

    df = pd.DataFrame(records)
    if not df.empty:
        group_map = build_connected_groups(df["video_id"].unique())
        df["group_id"] = df["video_id"].map(lambda v: group_map.get(str(v), str(v)))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    return df


def get_dataloaders(manifest_df):
    if "split" not in manifest_df.columns:
        manifest_df = assign_group_splits(manifest_df, seed=config.SEED)

    train_df = manifest_df[manifest_df["split"] == "train"].copy()
    val_df = manifest_df[manifest_df["split"] == "val"].copy()
    test_df = manifest_df[manifest_df["split"] == "test"].copy()

    sampler = None
    if len(train_df) > 0 and train_df["label"].nunique() > 1:
        if config.BALANCING_STRATEGY == "oversampling":
            class_counts = train_df["label"].value_counts()
            max_size = class_counts.max()
            lst = [group.sample(max_size, replace=True, random_state=config.SEED) for _, group in train_df.groupby("label")]
            train_df = pd.concat(lst, ignore_index=True).sample(frac=1.0, random_state=config.SEED).reset_index(drop=True)
        elif config.BALANCING_STRATEGY == "undersampling":
            class_counts = train_df["label"].value_counts()
            min_size = class_counts.min()
            lst = [group.sample(min_size, replace=False, random_state=config.SEED) for _, group in train_df.groupby("label")]
            train_df = pd.concat(lst, ignore_index=True).sample(frac=1.0, random_state=config.SEED).reset_index(drop=True)
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
        """Ensure reproducible worker initialization seeded from base torch seed."""
        worker_seed = torch.initial_seed() % 2**32 + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)

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

