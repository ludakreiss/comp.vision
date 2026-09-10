import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from evaluate import (
    apply_advanced_tier_distortion,
    load_model_from_checkpoint,
)
from transforms import get_transforms


from model import build_model
import warnings

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _find_checkpoint(model_type):
    candidates = [
        config.OUTPUT_ROOT / f"efficientnet_b0_{model_type}" / "best_model.pt",
        PROJECT_ROOT / "deepfake_robustness" / "outputs" / f"efficientnet_b0_{model_type}" / "best_model.pt",
        config.OUTPUT_ROOT / f"efficientnet_b0_{model_type}" / "best_single_model.pt",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None

def _load_demo_model(model_type, default_variant="fusion"):
    ckpt_path = _find_checkpoint(model_type)
    if ckpt_path is not None:
        model, threshold, ckpt = load_model_from_checkpoint(ckpt_path, DEVICE)
    else:
        warnings.warn(f"Checkpoint for '{model_type}' not found. Initializing untrained fallback model for demo startup.")
        variant = "modular_order" if model_type == "degradation" else default_variant
        model = build_model("efficientnet_b0", pretrained=False, model_variant=variant).to(DEVICE)
        threshold = 0.50
        ckpt = {}
    return model, threshold, ckpt

clean_model, clean_threshold, clean_ckpt = _load_demo_model("clean", default_variant="fusion")
robust_model, robust_threshold, robust_ckpt = _load_demo_model("degradation", default_variant="modular_order")

clean_model.eval()
robust_model.eval()

_, EVAL_TRANSFORM = get_transforms()


def apply_demo_degradation(image, degradation_name):
    if image is None:
        return None

    tier_cfg = config.DEGRADATION_TIERS[degradation_name]

    degraded = apply_advanced_tier_distortion(
        image,
        tier_cfg,
        sample_id="demo_input",
        global_seed=config.SEED,
    )

    return degraded


def predict_model(model, image, threshold):
    tensor = EVAL_TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        out = model(tensor)
        if isinstance(out, dict):
            logits = out["deepfake_logit"]
        else:
            logits, _ = out

        fake_probability = torch.sigmoid(
            logits.squeeze()
        ).item()

    prediction = (
        "FAKE"
        if fake_probability >= threshold
        else "REAL"
    )

    return {
        "prediction": prediction,
        "fake_probability": fake_probability,
        "threshold": threshold,
    }


def analyze_image(image, degradation_name):
    if image is None:
        return None, None, None

    degraded_image = apply_demo_degradation(
        image,
        degradation_name,
    )

    standard_result = predict_model(
        clean_model,
        degraded_image,
        clean_threshold,
    )

    robust_result = predict_model(
        robust_model,
        degraded_image,
        robust_threshold,
    )

    return (
        degraded_image,
        standard_result,
        robust_result,
    )