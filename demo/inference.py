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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLEAN_CKPT = (
    PROJECT_ROOT
    / "deepfake_robustness"
    / "outputs"
    / "efficientnet_b0_clean"
    / "best_model.pt"
)

ROBUST_CKPT = (
    PROJECT_ROOT
    / "deepfake_robustness"
    / "outputs"
    / "efficientnet_b0_degradation"
    / "best_model.pt"
)


# Load once when the app starts
clean_model, clean_threshold, clean_ckpt = load_model_from_checkpoint(
    CLEAN_CKPT,
    DEVICE,
)

robust_model, robust_threshold, robust_ckpt = load_model_from_checkpoint(
    ROBUST_CKPT,
    DEVICE,
)

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
        logits, _ = model(tensor)

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