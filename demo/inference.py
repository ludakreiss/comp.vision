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


def _find_checkpoint(model_type):
    candidates = [
        config.OUTPUT_ROOT
        / f"efficientnet_b0_{model_type}"
        / "best_model.pt",

        PROJECT_ROOT
        / "deepfake_robustness"
        / "outputs"
        / f"efficientnet_b0_{model_type}"
        / "best_model.pt",

        config.OUTPUT_ROOT
        / f"efficientnet_b0_{model_type}"
        / "best_single_model.pt",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def _load_demo_model(model_type):
    ckpt_path = _find_checkpoint(model_type)

    if ckpt_path is None:
        raise FileNotFoundError(
            f"Required checkpoint for '{model_type}' was not found."
        )

    print(f"Loading {model_type} model from:")
    print(ckpt_path)

    model, threshold, ckpt = load_model_from_checkpoint(
        ckpt_path,
        DEVICE,
    )

    model.eval()

    print(
        f"{model_type} model loaded "
        f"(threshold={threshold:.3f})"
    )

    return model, threshold, ckpt


# ---------------------------------------------------------
# Load both checkpoints once when the demo starts
# ---------------------------------------------------------

clean_model, clean_threshold, clean_ckpt = _load_demo_model(
    "clean"
)

robust_model, robust_threshold, robust_ckpt = _load_demo_model(
    "degradation"
)


# ---------------------------------------------------------
# Evaluation preprocessing
# ---------------------------------------------------------

_, EVAL_TRANSFORM = get_transforms()


# ---------------------------------------------------------
# Degradation
# ---------------------------------------------------------

def apply_demo_degradation(image, degradation_name):
    if image is None:
        return None

    if degradation_name not in config.DEGRADATION_TIERS:
        raise ValueError(
            f"Unknown degradation: {degradation_name}"
        )

    tier_cfg = config.DEGRADATION_TIERS[
        degradation_name
    ]

    degraded_image = apply_advanced_tier_distortion(
        image,
        tier_cfg,
        sample_id="demo_input",
        global_seed=config.SEED,
    )

    return degraded_image


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

def predict_model(model, image, threshold):
    tensor = (
        EVAL_TRANSFORM(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.inference_mode():
        output = model(tensor)

        # Newer model variants may return a dictionary.
        if isinstance(output, dict):
            logits = output["deepfake_logit"]

        # Existing fusion models return:
        # (logits, features)
        else:
            logits, _ = output

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


# ---------------------------------------------------------
# Main demo inference
# ---------------------------------------------------------

def analyze_image(image, degradation_name):
    if image is None:
        return None, None, None

    print("\n--- DEMO INFERENCE ---")
    print("Selected degradation:", degradation_name)
    print("Original image size:", image.size)

    # Create degraded copy
    degraded_image = apply_demo_degradation(
        image,
        degradation_name,
    )

    print("Degraded image size:", degraded_image.size)

    # -----------------------------------------------------
    # Standard model
    # -----------------------------------------------------

    standard_clean = predict_model(
        clean_model,
        image,
        clean_threshold,
    )

    standard_degraded = predict_model(
        clean_model,
        degraded_image,
        clean_threshold,
    )

    # -----------------------------------------------------
    # Robust model
    # -----------------------------------------------------

    robust_clean = predict_model(
        robust_model,
        image,
        robust_threshold,
    )

    robust_degraded = predict_model(
        robust_model,
        degraded_image,
        robust_threshold,
    )

    # -----------------------------------------------------
    # Debug output
    # -----------------------------------------------------

    print(
        "Standard clean:",
        standard_clean["fake_probability"],
    )

    print(
        "Standard degraded:",
        standard_degraded["fake_probability"],
    )

    print(
        "Robust clean:",
        robust_clean["fake_probability"],
    )

    print(
        "Robust degraded:",
        robust_degraded["fake_probability"],
    )

    # -----------------------------------------------------
    # Calculate probability change
    # -----------------------------------------------------

    standard_change = (
        standard_degraded["fake_probability"]
        - standard_clean["fake_probability"]
    )

    robust_change = (
        robust_degraded["fake_probability"]
        - robust_clean["fake_probability"]
    )

    standard_result = {
        "clean": standard_clean,
        "degraded": standard_degraded,
        "change": standard_change,
    }

    robust_result = {
        "clean": robust_clean,
        "degraded": robust_degraded,
        "change": robust_change,
    }

    print(
        "Standard change:",
        standard_change,
    )

    print(
        "Robust change:",
        robust_change,
    )

    return (
        degraded_image,
        standard_result,
        robust_result,
    )