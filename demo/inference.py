import sys
from pathlib import Path

# Add project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import apply_advanced_tier_distortion
import config


def apply_demo_degradation(image, degradation_name):
    if image is None:
        return None

    print("\n--- DEMO DEBUG ---")
    print("Selected degradation:", degradation_name)
    print("Original size:", image.size)

    tier_cfg = config.DEGRADATION_TIERS[degradation_name]

    print("Tier config:", tier_cfg)

    degraded = apply_advanced_tier_distortion(
        image,
        tier_cfg,
        sample_id="demo_input",
        global_seed=config.SEED
    )

    print("Output size:", degraded.size)
    print("------------------\n")

    return degraded