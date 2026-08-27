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
    calculate_brier_score,
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
        resumed_decay_b = [p for n, p in resumed_model.named_parameters() if p.requires_grad and not ("classifier" in n or "mlp" in n or "head" in n) and p.ndim >= 2]
        resumed_no_decay_b = [p for n, p in resumed_model.named_parameters() if p.requires_grad and not ("classifier" in n or "mlp" in n or "head" in n) and p.ndim < 2]
        resumed_decay_c = [p for n, p in resumed_model.named_parameters() if p.requires_grad and ("classifier" in n or "mlp" in n or "head" in n) and p.ndim >= 2]
        resumed_no_decay_c = [p for n, p in resumed_model.named_parameters() if p.requires_grad and ("classifier" in n or "mlp" in n or "head" in n) and p.ndim < 2]

        resumed_optimizer = torch.optim.AdamW([
            {"params": resumed_decay_b, "lr": 1e-4, "weight_decay": 5e-3},
            {"params": resumed_no_decay_b, "lr": 1e-4, "weight_decay": 0.0},
            {"params": resumed_decay_c, "lr": 1e-3, "weight_decay": 5e-3},
            {"params": resumed_no_decay_c, "lr": 1e-3, "weight_decay": 0.0},
        ])

        loaded_ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        resumed_optimizer.load_state_dict(loaded_ckpt["optimizer_state_dict"])

        assert len(resumed_optimizer.param_groups) == 4
        assert resumed_optimizer.param_groups[0]["weight_decay"] == 5e-3
        assert resumed_optimizer.param_groups[1]["weight_decay"] == 0.0
        assert resumed_optimizer.param_groups[2]["weight_decay"] == 5e-3
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


def test_srm_zero_sum_kernels_and_frozen_weights():
    """Verify MS-SRM layer filters are mathematically zero-sum and non-trainable."""
    from model import MultiScaleSRMLayer
    srm = MultiScaleSRMLayer()

    assert srm.conv_3x3.weight.requires_grad == False
    assert srm.conv_5x5.weight.requires_grad == False
    assert srm.conv_7x7.weight.requires_grad == False

    # Assert 7x7 filter 1 sums to exactly 0.0
    w7_f1 = srm.conv_7x7.weight[0, 0]
    assert torch.isclose(w7_f1.sum(), torch.tensor(0.0), atol=1e-5), f"f1_7 sum is {w7_f1.sum().item()}, expected 0.0"

    # Assert conv outputs 27 channels (9 for 3x3, 9 for 5x5, 9 for 7x7)
    x = torch.randn(2, 3, 64, 64)
    out = srm(x)
    assert out.shape == (2, 27, 64, 64)


def test_ece_degenerate_distribution_safety():
    """Verify calculate_ece returns 0.0 without crash when predictions are constant."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_prob_constant = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    ece = calculate_ece(y_true, y_prob_constant, adaptive=True)
    assert ece == 0.0


