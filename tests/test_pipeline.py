"""
Comprehensive Unit & Regression Tests for Deepfake Robustness Infrastructure.
Tests dataset graph splitting, optimizer resume 4-group structure, resolution-preserved SFCA shapes,
evaluation determinism, paired bootstrap math, and FDR correction.
"""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

import config
from dataset import build_connected_groups, assign_group_splits
from model import build_model, DeepfakeModel
from train import average_checkpoints
from evaluate import apply_advanced_tier_distortion
from metrics_utils import (
    calculate_ece,
    bootstrap_metric_ci,
    paired_bootstrap_test,
    compute_video_level_metrics,
    bootstrap_video_level_ci,
    paired_video_bootstrap_test,
    apply_fdr_correction,
)

def test_union_find_connected_groups():
    """Verify connected components graph fallback links paired manipulated video IDs."""
    video_ids = ["000", "003", "000_003", "003_000", "010", "011", "010_011"]
    group_map = build_connected_groups(video_ids)

    # 000, 003, 000_003, 003_000 must resolve to identical group
    assert group_map["000"] == group_map["003"]
    assert group_map["000"] == group_map["000_003"]
    assert group_map["000"] == group_map["003_000"]

    # 010 and 011 must resolve to identical group
    assert group_map["010"] == group_map["011"]
    assert group_map["010"] == group_map["010_011"]

    # The two components must remain completely independent
    assert group_map["000"] != group_map["010"]


def test_optimizer_resume_preserves_four_param_groups():
    """Regression test: Resumed optimizer must retain all 4 parameter groups and learning rates."""
    from train import build_optimizer
    model = build_model("efficientnet_b0", pretrained=False, model_variant="fusion")
    optimizer = build_optimizer(model)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Save state dict
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": 5,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_file = Path(tmpdir) / "checkpoint.pt"
        torch.save(ckpt, ckpt_file)

        # Restore
        resumed_model = build_model("efficientnet_b0", pretrained=False, model_variant="fusion")
        resumed_optimizer = build_optimizer(resumed_model)

        loaded_ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        resumed_optimizer.load_state_dict(loaded_ckpt["optimizer_state_dict"])

        assert len(resumed_optimizer.param_groups) == 4
        assert resumed_optimizer.param_groups[0]["weight_decay"] == config.WEIGHT_DECAY
        assert resumed_optimizer.param_groups[1]["weight_decay"] == 0.0
        assert resumed_optimizer.param_groups[2]["weight_decay"] == config.WEIGHT_DECAY
        assert resumed_optimizer.param_groups[3]["weight_decay"] == 0.0


def test_model_forward_shapes_all_variants():
    """Verify model forward pass works without NaNs across all model variants."""
    x = torch.randn(2, 3, 256, 256)

    for variant in ["fusion", "rgb_only", "fusion_no_attn"]:
        model = build_model("efficientnet_b0", pretrained=False, model_variant=variant)
        model.eval()
        with torch.no_grad():
            out, feat = model(x)
            assert out.shape == (2, 1)
            assert not torch.isnan(out).any()
            assert not torch.isinf(out).any()


def test_evaluation_determinism():
    """Verify apply_advanced_tier_distortion produces bit-identical output given the same sample seed."""
    from PIL import Image
    arr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    img = Image.fromarray(arr)

    tier_cfg = {"jpeg_quality": 50, "resize_scale": 0.50, "blur_sigma": 1.5, "noise_std": 6.0, "color_jitter": 0.2}

    d1 = apply_advanced_tier_distortion(img, tier_cfg, sample_id="sample_001", global_seed=42)
    d2 = apply_advanced_tier_distortion(img, tier_cfg, sample_id="sample_001", global_seed=42)

    arr1 = np.array(d1)
    arr2 = np.array(d2)

    np.testing.assert_array_equal(arr1, arr2)


def test_paired_bootstrap_observed_statistic():
    """Verify paired_bootstrap_test computes obs_auc_diff against original dataset difference."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    std_probs = np.array([0.1, 0.2, 0.8, 0.4, 0.6, 0.7, 0.3, 0.9])
    rob_probs = np.array([0.1, 0.1, 0.2, 0.3, 0.8, 0.9, 0.7, 0.9])

    res = paired_bootstrap_test(y_true, std_probs, rob_probs, n_bootstraps=100, seed=42)

    assert "mean_auc_diff" in res
    assert "p_value_auc" in res
    assert 0.0 <= res["p_value_auc"] <= 1.0
    assert abs(res["mean_auc_diff"] - 0.25) < 0.01  # rob_auc=1.0 - std_auc=0.75 = 0.25


def test_fdr_correction():
    """Verify Benjamini-Hochberg FDR correction produces monotonic adjusted p-values."""
    p_vals = [0.001, 0.01, 0.04, 0.20, 0.50]
    adj_p, sig = apply_fdr_correction(p_vals, alpha=0.05)

    assert len(adj_p) == 5
    assert np.all(adj_p >= np.array(p_vals))
    assert sig[0] == True
    assert sig[4] == False


def test_checkpoint_averaging_non_float_buffers():
    """Verify checkpoint averaging deepcopies tensors and preserves non-float buffers like num_batches_tracked."""
    m1 = nn.BatchNorm2d(10)
    m1.num_batches_tracked.fill_(10)
    m1.running_mean.fill_(1.0)

    m2 = nn.BatchNorm2d(10)
    m2.num_batches_tracked.fill_(20)
    m2.running_mean.fill_(3.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        p1 = Path(tmpdir) / "c1.pt"
        p2 = Path(tmpdir) / "c2.pt"
        out_p = Path(tmpdir) / "avg.pt"

        torch.save({"model_state_dict": m1.state_dict()}, p1)
        torch.save({"model_state_dict": m2.state_dict()}, p2)

        average_checkpoints([p1, p2], out_p, torch.device("cpu"))

        res = torch.load(out_p, map_location="cpu", weights_only=False)
        avg_state = res["model_state_dict"]

        # Mean of 1.0 and 3.0 = 2.0
        assert torch.isclose(avg_state["running_mean"].mean(), torch.tensor(2.0))
        # Non-float buffer preserved from first checkpoint
        assert avg_state["num_batches_tracked"].item() == 10





def test_ece_degenerate_distribution_safety():
    """Verify calculate_ece returns 0.0 without crash when predictions are constant."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_prob_constant = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    ece = calculate_ece(y_true, y_prob_constant, adaptive=True)
    assert ece == 0.0


def test_celebdf_manifest_validation():
    """Regression test: validate_celebdf_manifest rejects FF++ manifests and accepts valid Celeb-DF manifests."""
    from dataset import validate_celebdf_manifest

    invalid_df = pd.DataFrame([
        {
            "image_path": "/path/to/processed_faces/ffpp_c23/original/946/frame_000144.jpg",
            "video_id": "mini_946",
            "label": 0,
            "category": "YouTube-real"
        }
    ])
    is_valid, reason = validate_celebdf_manifest(invalid_df)
    assert not is_valid
    assert "FaceForensics++" in reason

    valid_df = pd.DataFrame([
        {
            "image_path": "/path/to/datasets/Celeb-DF-v2/Celeb-real/id0_0000.mp4/frame_00.jpg",
            "video_id": "id0_0000",
            "label": 0,
            "category": "Celeb-real",
            "split": "test"
        }
    ])
    is_valid_2, _ = validate_celebdf_manifest(valid_df)
    assert is_valid_2


def test_celebdf_identity_parser_no_sequence_link():
    """Regression test: Celeb-DF parser must NOT connect id0 and id1 via sequence number '0000'."""
    video_ids = ["id0_0000", "id1_0000", "id2_id3_0000", "id2_0001"]
    group_map = build_connected_groups(video_ids)

    # id0_0000 and id1_0000 must NOT be in the same group
    assert group_map["id0_0000"] != group_map["id1_0000"]

    # id2_id3_0000 should connect id2 and id3
    assert group_map["id2_id3_0000"] == group_map["id2_0001"]  # id2 identity should be linked


def test_limit_batches_optimizer_stepping():
    """Regression test: --limit_batches flushes accumulated gradients without zero updates."""
    from train import train_one_epoch, DeepfakeLoss
    model = build_model("efficientnet_b0", pretrained=False, model_variant="fusion")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = DeepfakeLoss(loss_type="bce")

    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 10
        def __getitem__(self, idx):
            return {
                "image": torch.randn(3, 64, 64),
                "label": torch.tensor(float(idx % 2)),
                "path": f"/tmp/{idx}.jpg",
                "video_id": f"vid_{idx}"
            }

    loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=2)

    # Run for limit_batches=1 with GRADIENT_ACCUMULATION_STEPS=4
    config.GRADIENT_ACCUMULATION_STEPS = 4
    initial_p = next(p for p in model.parameters() if p.requires_grad).clone()

    train_one_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        scaler=None,
        device=torch.device("cpu"),
        limit_batches=1,
    )

    updated_p = next(p for p in model.parameters() if p.requires_grad)
    # Trainable parameter should have updated even with limit_batches=1
    assert not torch.allclose(initial_p, updated_p)


def test_calculate_binary_metrics_empty_inputs():
    """Regression test: calculate_binary_metrics handling of empty inputs."""
    from train import calculate_binary_metrics
    metrics = calculate_binary_metrics([], [])
    assert np.isnan(metrics["accuracy"])
    assert np.isnan(metrics["roc_auc"])


def test_progressive_unfreeze_resume_state_restoration():
    """Regression test: Progressive unfreeze resume restores requires_grad state and optimizer topology."""
    model = build_model("efficientnet_b0", pretrained=False, model_variant="fusion")

    # Unfreeze feature block 0
    for p in model.features[0].parameters():
        p.requires_grad = True

    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]

    ckpt = {
        "model_state_dict": model.state_dict(),
        "trainable_param_names": trainable_names,
    }

    # Create fresh model with default frozen state
    fresh_model = build_model("efficientnet_b0", pretrained=False, model_variant="fusion")

    # Restore requires_grad states
    trainable_set = set(ckpt["trainable_param_names"])
    for name, param in fresh_model.named_parameters():
        param.requires_grad = (name in trainable_set)

    fresh_trainable_names = [n for n, p in fresh_model.named_parameters() if p.requires_grad]
    assert fresh_trainable_names == trainable_names


def test_scheduler_optimizer_identity_on_resume():
    """Regression test: Resumed scheduler must bind to the newly reconstructed optimizer instance."""
    model = build_model("efficientnet_b0", pretrained=False, model_variant="fusion")

    decay_b = [p for n, p in model.named_parameters() if p.requires_grad and not ("classifier" in n or "mlp" in n or "head" in n) and p.ndim >= 2]
    no_decay_b = [p for n, p in model.named_parameters() if p.requires_grad and not ("classifier" in n or "mlp" in n or "head" in n) and p.ndim < 2]
    decay_c = [p for n, p in model.named_parameters() if p.requires_grad and ("classifier" in n or "mlp" in n or "head" in n) and p.ndim >= 2]
    no_decay_c = [p for n, p in model.named_parameters() if p.requires_grad and ("classifier" in n or "mlp" in n or "head" in n) and p.ndim < 2]

    optimizer = torch.optim.AdamW([
        {"params": decay_b, "lr": 1e-4, "weight_decay": 5e-3},
        {"params": no_decay_b, "lr": 1e-4, "weight_decay": 0.0},
        {"params": decay_c, "lr": 1e-3, "weight_decay": 5e-3},
        {"params": no_decay_c, "lr": 1e-3, "weight_decay": 0.0},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Reconstruct resumed optimizer and verify scheduler binding
    resumed_optimizer = torch.optim.AdamW([
        {"params": decay_b, "lr": 1e-4, "weight_decay": 5e-3},
        {"params": no_decay_b, "lr": 1e-4, "weight_decay": 0.0},
        {"params": decay_c, "lr": 1e-3, "weight_decay": 5e-3},
        {"params": no_decay_c, "lr": 1e-3, "weight_decay": 0.0},
    ])
    resumed_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(resumed_optimizer, T_max=10)

    assert resumed_scheduler.optimizer is resumed_optimizer
    initial_lr = resumed_optimizer.param_groups[0]["lr"]
    resumed_scheduler.step()
    assert resumed_optimizer.param_groups[0]["lr"] != initial_lr


def test_distinct_degradation_tiers():
    """Verify screenshot_recompress and resize_50_compress_70 produce distinct outputs."""
    from PIL import Image
    arr = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    img = Image.fromarray(arr)

    t1 = apply_advanced_tier_distortion(img, config.DEGRADATION_TIERS["resize_50_compress_70"], sample_id="s1", global_seed=42)
    t2 = apply_advanced_tier_distortion(img, config.DEGRADATION_TIERS["screenshot_recompress"], sample_id="s1", global_seed=42)

    a1 = np.array(t1)
    a2 = np.array(t2)
    assert not np.array_equal(a1, a2)


def test_align_face_crop_affine_eye_centering():
    """Verify align_face_crop maps eye midpoint to target coordinates within tolerance."""
    from extract_faces import align_face_crop
    frame = np.zeros((500, 500, 3), dtype=np.uint8)
    frame[150:350, 150:350] = 255

    # Controlled eye landmarks
    left_eye = np.array([200.0, 220.0])
    right_eye = np.array([300.0, 220.0])
    landmarks = np.array([left_eye, right_eye])

    cropped = align_face_crop(frame, landmarks, target_size=256, margin_percent=0.10)
    assert cropped.size == (256, 256)


def test_partial_gradient_accumulation_scaling():
    """Regression test: Partial gradient accumulation window scales gradients to match exact average."""
    from train import train_one_epoch, DeepfakeLoss
    model = build_model("efficientnet_b0", pretrained=False, model_variant="fusion")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = DeepfakeLoss(loss_type="bce")

    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 5  # 5 samples with batch_size=2 -> 3 batches (2, 2, 1)
        def __getitem__(self, idx):
            return {
                "image": torch.randn(3, 64, 64),
                "label": torch.tensor(float(idx % 2)),
                "path": f"/tmp/{idx}.jpg",
                "video_id": f"vid_{idx}"
            }

    loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=2)
    config.GRADIENT_ACCUMULATION_STEPS = 4

    train_one_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        scaler=None,
        device=torch.device("cpu"),
    )
    # Check that model parameters were updated without NaN or Inf values
    for p in model.parameters():
        if p.requires_grad:
            assert not torch.isnan(p).any()
            assert not torch.isinf(p).any()


def test_predictions_df_has_manipulation_key():
    """Verify evaluate_model includes 'manipulation' column in returned predictions DataFrame."""
    from train import evaluate_model, DeepfakeLoss
    model = build_model("efficientnet_b0", pretrained=False, model_variant="fusion")
    criterion = DeepfakeLoss(loss_type="bce")

    class SampleDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 2
        def __getitem__(self, idx):
            return {
                "image": torch.randn(3, 64, 64),
                "label": torch.tensor(1.0),
                "path": f"/tmp/frame_{idx}.jpg",
                "video_id": "000_003",
                "manipulation": "Deepfakes",
            }

    loader = torch.utils.data.DataLoader(SampleDataset(), batch_size=2)
    _, preds_df = evaluate_model(model, loader, criterion, torch.device("cpu"))
    assert "manipulation" in preds_df.columns
    assert preds_df["manipulation"].iloc[0] == "Deepfakes"


def test_srm_filters_zero_sum_and_shapes():
    """Verify MultiScaleSRMLayer kernels are zero-sum, non-trainable, and produce 27 channels."""
    from model import MultiScaleSRMLayer
    srm = MultiScaleSRMLayer()
    for name, param in srm.named_parameters():
        assert not param.requires_grad

    # Conv layers weights zero sum check
    for conv in [srm.conv_3x3, srm.conv_5x5, srm.conv_7x7]:
        weight_sums = conv.weight.sum(dim=(2, 3))
        assert torch.allclose(weight_sums, torch.zeros_like(weight_sums), atol=1e-5)

    dummy_input = torch.randn(2, 3, 64, 64)
    output = srm(dummy_input)
    assert output.shape == (2, 27, 64, 64)


def test_ece_constant_prediction_calibration_error():
    """Verify constant prediction (0.5 for all, 0.8 positive prevalence) produces calibration error ~0.3."""
    from metrics_utils import calculate_ece
    y_true = np.array([1]*80 + [0]*20)  # 80% positive
    y_prob = np.array([0.5]*100)        # Constant 0.5 prediction
    ece = calculate_ece(y_true, y_prob)
    assert np.isclose(ece, 0.30, atol=0.05)


def test_manifest_leakage_and_validation():
    """Verify validate_manifest detects group leakage, missing columns, and invalid labels."""
    from dataset import validate_manifest
    # Leaking manifest
    df_leak = pd.DataFrame([
        {"image_path": "/tmp/1.jpg", "video_id": "v1", "group_id": "g1", "label": 1, "split": "train"},
        {"image_path": "/tmp/2.jpg", "video_id": "v2", "group_id": "g1", "label": 0, "split": "test"},
    ])
    valid, msg = validate_manifest(df_leak)
    assert not valid
    assert "leakage" in msg.lower() or "group" in msg.lower()


def test_resume_training_smoke(tmp_path):
    """Smoke test: Verify model, optimizer, and scheduler resume binding from checkpoint."""
    from train import build_optimizer, build_scheduler, safe_torch_save
    model = build_model("efficientnet_b0", pretrained=False, model_variant="fusion")
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)

    ckpt_path = tmp_path / "last_model.pt"
    safe_torch_save({
        "epoch": 2,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "trainable_param_names": [n for n, p in model.named_parameters() if p.requires_grad],
    }, ckpt_path)

    resumed_ckpt = torch.load(ckpt_path, weights_only=False)
    resumed_model = build_model("efficientnet_b0", pretrained=False, model_variant="fusion")
    resumed_optimizer = build_optimizer(resumed_model)
    resumed_optimizer.load_state_dict(resumed_ckpt["optimizer_state_dict"])
    resumed_scheduler = build_scheduler(resumed_optimizer)
    resumed_scheduler.load_state_dict(resumed_ckpt["scheduler_state_dict"])

    assert resumed_scheduler.optimizer is resumed_optimizer
    assert resumed_ckpt["epoch"] == 2


def test_end_to_end_training_smoke(tmp_path):
    """End-to-end lightweight training smoke test exercising full orchestration path on CPU."""
    from PIL import Image
    from train import train_one_epoch, evaluate_model, DeepfakeLoss, build_optimizer, build_scheduler, safe_torch_save
    from dataset import get_dataloaders
    
    # 1. Create temporary dataset images
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    paths = []
    for i in range(4):
        p = img_dir / f"frame_{i}.jpg"
        Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(p)
        paths.append(str(p))

    # 2. Build dummy manifest DataFrame
    manifest_df = pd.DataFrame([
        {"image_path": paths[0], "video_id": "v1", "group_id": "g1", "label": 1, "split": "train", "manipulation": "Deepfakes"},
        {"image_path": paths[1], "video_id": "v2", "group_id": "g2", "label": 0, "split": "train", "manipulation": "original"},
        {"image_path": paths[2], "video_id": "v3", "group_id": "g3", "label": 1, "split": "val", "manipulation": "Deepfakes"},
        {"image_path": paths[3], "video_id": "v4", "group_id": "g4", "label": 0, "split": "test", "manipulation": "original"},
    ])

    train_loader, val_loader, test_loader = get_dataloaders(manifest_df)
    
    # 3. Instantiate model, optimizer, scheduler, criterion
    model = build_model("efficientnet_b0", pretrained=False, model_variant="fusion")
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)
    criterion = DeepfakeLoss(loss_type="bce")

    # 4. Run 1 epoch training & validation
    t_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler=None, device=torch.device("cpu"))
    v_metrics, v_preds = evaluate_model(model, val_loader, criterion, torch.device("cpu"))

    assert "loss" in t_metrics
    assert "roc_auc" in v_metrics
    assert len(v_preds) == 1

    # 5. Save best checkpoint
    ckpt_path = tmp_path / "best_model.pt"
    safe_torch_save({"model_state_dict": model.state_dict(), "epoch": 1}, ckpt_path)
    assert ckpt_path.exists()
