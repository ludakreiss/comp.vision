import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from evaluate import (
    apply_advanced_tier_distortion,
    load_model_from_checkpoint,
)
from transforms import get_transforms
from gradcam import compute_gradcam_overlay


try:
    import spaces
    GPU = spaces.GPU
except (ImportError, AttributeError):
    def GPU(fn=None, **kwargs):
        if fn is not None:
            return fn
        def decorator(f):
            return f
        return decorator


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _find_checkpoint(model_type):
    # 1. Environment variable override
    env_dir = os.getenv("CHECKPOINT_DIR") or os.getenv("MODEL_DIR")
    if env_dir:
        env_candidates = [
            Path(env_dir) / f"efficientnet_b0_{model_type}" / "best_model.pt",
            Path(env_dir) / f"efficientnet_b0_{model_type}" / "best_single_model.pt",
            Path(env_dir) / model_type / "best_model.pt",
            Path(env_dir) / f"{model_type}.pt",
        ]
        for candidate in env_candidates:
            if candidate.exists():
                return candidate

    # 2. Local dedicated checkpoints folder (Space deployment standard)
    repo_candidates = [
        PROJECT_ROOT / "checkpoints" / f"efficientnet_b0_{model_type}" / "best_model.pt",
        PROJECT_ROOT / "checkpoints" / f"efficientnet_b0_{model_type}" / "best_single_model.pt",
        PROJECT_ROOT / "checkpoints" / model_type / "best_model.pt",
        PROJECT_ROOT / "checkpoints" / f"{model_type}.pt",
        Path("checkpoints") / f"efficientnet_b0_{model_type}" / "best_model.pt",
        Path("checkpoints") / model_type / "best_model.pt",
    ]
    for candidate in repo_candidates:
        if candidate.exists():
            return candidate

    # 3. Training outputs directory (local development backward compatibility)
    output_candidates = [
        config.OUTPUT_ROOT
        / f"efficientnet_b0_{model_type}"
        / "best_model.pt",
        config.OUTPUT_ROOT
        / f"efficientnet_b0_{model_type}"
        / "best_single_model.pt",
    ]
    for candidate in output_candidates:
        if candidate.exists():
            return candidate

    # 4. Hugging Face Hub download fallback (if HF_MODEL_REPO_ID is set)
    hf_repo_id = os.getenv("HF_MODEL_REPO_ID")
    if hf_repo_id:
        try:
            from huggingface_hub import hf_hub_download

            target_filename = f"checkpoints/efficientnet_b0_{model_type}/best_model.pt"
            print(f"Downloading checkpoint for '{model_type}' from Hugging Face Hub: {hf_repo_id}...")
            downloaded = hf_hub_download(
                repo_id=hf_repo_id,
                filename=target_filename,
            )
            return Path(downloaded)
        except Exception as err:
            print(f"Failed downloading checkpoint from HF Hub ({hf_repo_id}): {err}")

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    first_param = next(model.parameters(), None)
    if first_param is not None and first_param.device != device:
        model.to(device)

    tensor = (
        EVAL_TRANSFORM(image)
        .unsqueeze(0)
        .to(device)
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
# Degradation level mapping
# ---------------------------------------------------------

DEGRADATION_LEVEL_NAMES = {
    0: "clean",
    1: "weak_compression",
    2: "medium_compression",
    3: "strong_compression",
    4: "extreme_compression",
}


# ---------------------------------------------------------
# Main demo inference with severity curve
# ---------------------------------------------------------

@GPU
def analyze_image_with_curve(image, degradation_name):
    if image is None:
        return None, None, None, None

    print("\n--- DEMO INFERENCE WITH SEVERITY CURVE ---")
    print("Selected degradation:", degradation_name)
    print("Original image size:", image.size)

    # 1. Clean predictions (Severity 0)
    standard_clean = predict_model(
        clean_model,
        image,
        clean_threshold,
    )
    robust_clean = predict_model(
        robust_model,
        image,
        robust_threshold,
    )

    # 2. Evaluate across all 5 severity levels (0 to 4)
    curve_levels = [0, 1, 2, 3, 4]
    curve_labels = ["Clean", "Weak", "Medium", "Strong", "Extreme"]
    curve_std_probs = []
    curve_rob_probs = []
    level_images = {}

    for lvl in curve_levels:
        lvl_name = DEGRADATION_LEVEL_NAMES[lvl]
        if lvl == 0:
            deg_img = image
            std_res = standard_clean
            rob_res = robust_clean
        else:
            deg_img = apply_demo_degradation(image, lvl_name)
            std_res = predict_model(clean_model, deg_img, clean_threshold)
            rob_res = predict_model(robust_model, deg_img, robust_threshold)

        level_images[lvl_name] = deg_img
        curve_std_probs.append(std_res["fake_probability"])
        curve_rob_probs.append(rob_res["fake_probability"])

    # 3. Target the user's selected degradation level
    selected_image = level_images.get(degradation_name, image)
    if degradation_name == "clean":
        standard_degraded = standard_clean
        robust_degraded = robust_clean
    else:
        name_list = [DEGRADATION_LEVEL_NAMES[l] for l in curve_levels]
        idx = name_list.index(degradation_name) if degradation_name in name_list else 0
        standard_degraded = {
            "prediction": "FAKE" if curve_std_probs[idx] >= clean_threshold else "REAL",
            "fake_probability": curve_std_probs[idx],
            "threshold": clean_threshold,
        }
        robust_degraded = {
            "prediction": "FAKE" if curve_rob_probs[idx] >= robust_threshold else "REAL",
            "fake_probability": curve_rob_probs[idx],
            "threshold": robust_threshold,
        }

    # 4. Compute probability changes
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

    curve_data = {
        "levels": curve_levels,
        "labels": curve_labels,
        "standard_probs": curve_std_probs,
        "robust_probs": curve_rob_probs,
    }

    print("Standard clean:", standard_clean["fake_probability"])
    print("Standard degraded:", standard_degraded["fake_probability"])
    print("Standard change:", standard_change)
    print("Robust clean:", robust_clean["fake_probability"])
    print("Robust degraded:", robust_degraded["fake_probability"])
    print("Robust change:", robust_change)

    return selected_image, standard_result, robust_result, curve_data


@GPU
def analyze_image(image, degradation_name):
    selected_image, standard_result, robust_result, _ = analyze_image_with_curve(
        image, degradation_name
    )
    return selected_image, standard_result, robust_result


# ---------------------------------------------------------
# Grad-CAM (on-demand, explicit button trigger only)
# ---------------------------------------------------------

@GPU
def generate_gradcam_grid(image, degradation_name):
    """
    Runs Grad-CAM for both models against both the clean and distorted
    versions of `image`. Only invoked when the user explicitly clicks
    "Generate Grad-CAM" in the UI -- never on slider movement.

    Returns 4 PIL.Image overlays:
        standard_clean_cam, standard_distorted_cam,
        robust_clean_cam, robust_distorted_cam
    """
    if image is None:
        return None, None, None, None

    distorted_image = (
        image
        if degradation_name == "clean"
        else apply_demo_degradation(image, degradation_name)
    )

    standard_clean_cam, _ = compute_gradcam_overlay(clean_model, EVAL_TRANSFORM, image)
    standard_distorted_cam, _ = compute_gradcam_overlay(clean_model, EVAL_TRANSFORM, distorted_image)
    robust_clean_cam, _ = compute_gradcam_overlay(robust_model, EVAL_TRANSFORM, image)
    robust_distorted_cam, _ = compute_gradcam_overlay(robust_model, EVAL_TRANSFORM, distorted_image)

    return standard_clean_cam, standard_distorted_cam, robust_clean_cam, robust_distorted_cam