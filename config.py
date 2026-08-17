import torch
from pathlib import Path

# Environment Detection (Google Colab vs Local)
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Base Paths (Relative to the script location)
BASE_DIR = Path(__file__).resolve().parent

if IN_COLAB:
    PROJECT_ROOT = Path("/content/drive/MyDrive/deepfake_robustness")
    DATASET_ROOT = Path("/content/drive/MyDrive/datasets/FaceForensics")
else:
    PROJECT_ROOT = BASE_DIR / "deepfake_robustness"
    DATASET_ROOT = BASE_DIR / "datasets" / "FaceForensics"

DOWNLOAD_SCRIPT = BASE_DIR / "download-FaceForensics.py"
PROCESSED_ROOT = PROJECT_ROOT / "processed_faces"
FFPP_FACE_ROOT = PROCESSED_ROOT / "ffpp_c23"          # Primary training dataset (C23 compression)
FFPP_C40_FACE_ROOT = PROCESSED_ROOT / "ffpp_c40"      # C40 stress-test eval tier (eval-only, not used in training)
FFPP_C40_MANIFEST_PATH = PROJECT_ROOT / "ffpp_c40_manifest.csv"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
MANIFEST_PATH = PROJECT_ROOT / "ffpp_manifest.csv"

# Global Random Seed
SEED = 42

# Training & Split Hyperparameters
IMAGE_SIZE = 256  # 256x256 high-resolution crop for sharp SRM frequency traces
BATCH_SIZE = 32  # Per-GPU micro batch size
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 32 * 4 = 128 for stable BN & optimization
NUM_EPOCHS = 30  # 30 epochs for full convergence on expanded dataset
BACKBONE_LR = 4e-5  # Reduced learning rate for pretrained backbone to prevent overfitting
CLASSIFIER_LR = 1e-3  # Scaled learning rate for linear classifier head
WEIGHT_DECAY = 5e-3  # Increased weight decay for stronger regularization
PATIENCE = 7
NUM_WORKERS = 4  # Dataloader workers

# Data Composition & Split Configuration (University of Technology Nuremberg Proposal Specification)
TRAIN_RATIO = 0.70  # 70% Training
VAL_RATIO = 0.15    # 15% Validation
TEST_RATIO = 0.15   # 15% Testing

# Face Extraction Configuration
FRAME_INTERVAL = 5  # Extract every 5th frame (2x denser sampling)
MAX_FRAMES_PER_VIDEO = 60  # Extract 60 frames per video (2x dataset density)
MIN_FACE_PROBABILITY = 0.90
FACE_MARGIN = 20
FACE_MARGIN_PERCENT = 0.10
FORCE_REEXTRACT = False  # Enabled incremental resume: skips videos already extracted
ALIGN_FACES = True
MAX_VIDEOS_PER_CATEGORY = None
MAX_EXTRACTION_VIDEOS_PER_CATEGORY = None

# Data Paths (FaceForensics++ Main & Celeb-DF External Dataset)
CELEBDF_ROOT = BASE_DIR / "datasets" / "Celeb-DF-v2"
CELEBDF_MANIFEST_PATH = PROJECT_ROOT / "celebdf_manifest.csv"

# Official Celeb-DF Download Links
CELEBDF_V2_GDRIVE_URL = "https://drive.google.com/open?id=1iLx76wsbi9itnkxSqz9BVBl4ZvnbIazj"
CELEBDF_V1_GDRIVE_URL = "https://drive.google.com/open?id=10NGF38RgF8FZneKOuCOdRIsPzpC7_WDd"
CELEBDF_V2_BAIDU_URL = "https://pan.baidu.com/s/1EcYX0s4U3kbI1V2vdrP46A"  # Passcode: yxa1
CELEBDF_V1_BAIDU_URL = "https://pan.baidu.com/s/16QulfMFG4TQB9iMZIZnsjQ"  # Passcode: ku0s

# Model Configuration
MODEL_NAME = "efficientnet_b0"  # Lightweight CNN backbone
PRETRAINED = True  # ImageNet pretrained weights initialized for high performance (>85-92% AUC)
STOCHASTIC_DEPTH_PROB = 0.30  # DropPath probability increased for regularization
BRANCH_MODE = "fusion"  # Dual RGB + Frequency branch with Spatial-Frequency Cross-Attention

# Planned Method: Two Model Strategies
# 'clean': Standard model trained only on clean images (no degradation augmentation)
# 'degradation': Robustness-aware model trained with simulated degradations
TRAINING_STRATEGY = "clean"  # Options: 'clean', 'degradation'

# Degradation Evaluation Matrix Settings (Weak / Medium / Strong / Extreme + Resizing + Platform Chains)
DEGRADATION_TIERS = {
    "clean": {"jpeg_quality": None, "resize_scale": 1.0, "blur_sigma": None, "noise_std": None},
    "weak_compression": {"jpeg_quality": 90, "resize_scale": 1.0},
    "medium_compression": {"jpeg_quality": 70, "resize_scale": 1.0},
    "strong_compression": {"jpeg_quality": 45, "resize_scale": 1.0},
    "extreme_compression": {"jpeg_quality": 30, "resize_scale": 1.0},
    "resize_75": {"jpeg_quality": None, "resize_scale": 0.75},
    "resize_50": {"jpeg_quality": None, "resize_scale": 0.50},
    "resize_25": {"jpeg_quality": None, "resize_scale": 0.25},
    "gaussian_blur": {"blur_sigma": 1.5},
    "motion_blur": {"motion_blur_size": 5},
    "gaussian_noise": {"noise_std": 6.0},
    "color_jitter": {"color_jitter": 0.2},
    "resize_50_compress_70": {"jpeg_quality": 70, "resize_scale": 0.50},
    "screenshot_recompress": {"resize_scale": 0.50, "jpeg_quality": 70, "chain": "screenshot"},
    "social_media_pipeline": {"resize_scale": 0.50, "jpeg_quality": 50, "motion_blur_size": 3, "chain": "social"},
}

# Downloader & Face Preprocessing Configuration
DOWNLOAD_NUM_VIDEOS = None
DOWNLOAD_SERVER = "EU2"
SAMPLING_STRATEGY = "spaced"
BLUR_THRESHOLD = 25.0
MIN_FACE_SIZE = 60

# Loss & Balancing Configuration
BALANCING_STRATEGY = "class_balanced_loss"  # Class-Balanced Loss weighted by inverse effective number of samples
CB_BETA = 0.999
FOCAL_ALPHA = 0.50
FOCAL_GAMMA = 1.5

# Forensic-Preserving Augmentations (For Robustness-Aware Model)
AUG_JPEG_PROB = 0.70
AUG_JPEG_QUALITY_MIN = 45
AUG_JPEG_QUALITY_MAX = 95
AUG_DOWNSCALE_PROB = 0.50
AUG_DOWNSCALE_SCALES = [0.25, 0.50, 0.75, 1.00]
AUG_BLUR_PROB = 0.50
AUG_BLUR_SIGMA_MIN = 0.1
AUG_BLUR_SIGMA_MAX = 1.2
AUG_MOTION_BLUR_PROB = 0.30
AUG_MOTION_BLUR_SIZES = [3, 5]
AUG_SHARPEN_PROB = 0.20
AUG_SHARPEN_FACTOR_MIN = 1.1
AUG_SHARPEN_FACTOR_MAX = 1.4
AUG_COLOR_JITTER_PROB = 0.4
AUG_COLOR_JITTER_BRIGHTNESS = 0.1
AUG_COLOR_JITTER_CONTRAST = 0.1
AUG_COLOR_JITTER_SATURATION = 0.1
AUG_NOISE_PROB = 0.40
AUG_NOISE_STD_MIN = 1.0
AUG_NOISE_STD_MAX = 8.0

# Model Improvements Configuration
FREEZE_PERCENT = 0.50  # Freeze first 50% of backbone blocks; progressively unfrozen during training
PROGRESSIVE_UNFREEZE = True   # Gradually unfreeze backbone blocks over 80% of training
DROPOUT = 0.50  # Classifier dropout increased for stronger regularization
LABEL_SMOOTHING = 0.05  # Light label smoothing to prevent overconfident predictions
USE_MIXED_PRECISION = True

# Degradation Curriculum Configuration (degradation strategy only)
# Severity scale ramps linearly from CURRICULUM_SEVERITY_START to 1.0
# over the first CURRICULUM_RAMP_EPOCHS epochs, then holds at 1.0.
CURRICULUM_ENABLED = True
CURRICULUM_SEVERITY_START = 0.3   # Initial severity (30% of max aug strength)
CURRICULUM_RAMP_EPOCHS = 10       # Ramp over first 10 epochs (50% of 20-epoch run)
GRADIENT_CLIPPING = 1.0
SCHEDULER = "cosine"
WARMUP_EPOCHS = 2  # 2 warmup epochs for fine-tuning
EARLY_STOPPING_PATIENCE = 7
EMA_DECAY = 0.9999

# Mixup Configuration (only active for 'degradation' and 'combined' strategies — never for 'clean')
USE_MIXUP = True
MIXUP_PROB = 0.70
MIXUP_ALPHA = 0.8
# CUTMIX_ALPHA removed: CutMix was never implemented; dead config key deleted to prevent confusion

# Part 5: Training & Post-Training Checkpoint Averaging Configuration
NUM_CHECKPOINTS_TO_AVERAGE = 3  # Average top K validation checkpoints
TENSORBOARD_DIR = OUTPUT_ROOT / "runs"

def ensure_directories():
    """Ensure output and data directories exist."""
    for folder in [PROJECT_ROOT, DATASET_ROOT, PROCESSED_ROOT, FFPP_FACE_ROOT, OUTPUT_ROOT, TENSORBOARD_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

# Run directory creation by default when imported
ensure_directories()

