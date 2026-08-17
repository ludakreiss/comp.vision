"""
Metrics Utilities for Statistical Rigor, Calibration, and Hypothesis Testing.
Includes Non-parametric Bootstrap 95% Confidence Intervals, Paired Significance Testing,
and Expected Calibration Error (ECE) calculations.
"""

from typing import Dict, Tuple, Any
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error (ECE) using equal-width probability bins.
    
    Args:
        y_true: Binary ground truth targets (0 or 1).
        y_prob: Predicted confidence probabilities for positive class (0.0 to 1.0).
        n_bins: Number of probability calibration bins.
        
    Returns:
        Scalar ECE score (lower is better calibrated).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)
    
    if total_samples == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
            
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_size / total_samples) * np.abs(bin_acc - bin_conf)
            
    return float(ece)

def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 1000,
    threshold: float = 0.50,
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Compute 95% Non-parametric Percentile Confidence Intervals via Bootstrap Resampling.
    
    Args:
        y_true: Binary ground truth labels.
        y_prob: Predicted probabilities.
        n_bootstraps: Number of bootstrap resamples.
        threshold: Classification decision threshold.
        seed: Random seed for reproducibility.
        
    Returns:
        Dictionary mapping metric names to {'mean': val, 'ci_lower': val, 'ci_upper': val}.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n_samples = len(y_true)
    
    rng = np.random.default_rng(seed)
    
    boot_acc = []
    boot_f1 = []
    boot_auc = []
    
    for _ in range(n_bootstraps):
        indices = rng.integers(0, n_samples, n_samples)
        sub_y_true = y_true[indices]
        sub_y_prob = y_prob[indices]
        sub_y_pred = (sub_y_prob >= threshold).astype(int)
        
        # Guard against single-class bootstrap resamples
        if len(np.unique(sub_y_true)) < 2:
            continue
            
        boot_acc.append(accuracy_score(sub_y_true, sub_y_pred))
        boot_f1.append(f1_score(sub_y_true, sub_y_pred, zero_division=0))
        boot_auc.append(roc_auc_score(sub_y_true, sub_y_prob))

    def get_stats(arr: list) -> Dict[str, float]:
        if not arr:
            return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
        arr_np = np.array(arr)
        return {
            "mean": float(np.mean(arr_np)),
            "ci_lower": float(np.percentile(arr_np, 2.5)),
            "ci_upper": float(np.percentile(arr_np, 97.5)),
        }

    return {
        "accuracy": get_stats(boot_acc),
        "f1": get_stats(boot_f1),
        "roc_auc": get_stats(boot_auc),
    }

def paired_bootstrap_test(
    y_true: np.ndarray,
    y_prob_std: np.ndarray,
    y_prob_rob: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42
) -> Dict[str, float]:
    """
    Perform Paired Bootstrap Difference Test between Standard and Robustness-Aware Models.
    Computes two-sided p-values for difference in ROC-AUC and F1-Score.
    
    Args:
        y_true: Binary ground truth targets.
        y_prob_std: Predictions from Standard Model.
        y_prob_rob: Predictions from Robustness-Aware Model.
        n_bootstraps: Number of bootstrap iterations.
        seed: Random seed.
        
    Returns:
        Dictionary with AUC difference, F1 difference, and p-values.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob_std = np.asarray(y_prob_std).astype(float)
    y_prob_rob = np.asarray(y_prob_rob).astype(float)
    n_samples = len(y_true)
    
    rng = np.random.default_rng(seed)
    
    auc_diffs = []
    f1_diffs = []
    
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n_samples, n_samples)
        sub_true = y_true[idx]
        sub_std = y_prob_std[idx]
        sub_rob = y_prob_rob[idx]
        
        if len(np.unique(sub_true)) < 2:
            continue
            
        std_auc = roc_auc_score(sub_true, sub_std)
        rob_auc = roc_auc_score(sub_true, sub_rob)
        auc_diffs.append(rob_auc - std_auc)
        
        std_f1 = f1_score(sub_true, (sub_std >= 0.5).astype(int), zero_division=0)
        rob_f1 = f1_score(sub_true, (sub_rob >= 0.5).astype(int), zero_division=0)
        f1_diffs.append(rob_f1 - std_f1)

    auc_diffs = np.array(auc_diffs)
    f1_diffs = np.array(f1_diffs)
    
    # Correct two-sided bootstrap p-value (permutation principle):
    # Shift bootstrap distribution to be centered at 0 under H₀, then count
    # how often the null distribution's absolute deviation exceeds the observed one.
    # Reference: Efron & Tibshirani (1993), §16.4
    obs_auc_diff = float(np.mean(auc_diffs))
    obs_f1_diff = float(np.mean(f1_diffs))

    shifted_auc = auc_diffs - obs_auc_diff  # Center at 0
    shifted_f1 = f1_diffs - obs_f1_diff

    p_val_auc = float(np.mean(np.abs(shifted_auc) >= np.abs(obs_auc_diff)))
    p_val_f1 = float(np.mean(np.abs(shifted_f1) >= np.abs(obs_f1_diff)))
    
    return {
        "mean_auc_diff": obs_auc_diff,
        "p_value_auc": min(1.0, p_val_auc),
        "mean_f1_diff": obs_f1_diff,
        "p_value_f1": min(1.0, p_val_f1),
    }
