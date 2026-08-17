import io
import random
import numpy as np
import cv2
from PIL import Image, ImageEnhance

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import config

class RandomJPEGCompression:
    def __init__(self, quality_range=(20, 95), probability=0.75):
        self.quality_range = quality_range
        self.probability = probability

    def __call__(self, image):
        if random.random() >= self.probability:
            return image

        quality = random.randint(self.quality_range[0], self.quality_range[1])
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed_image = Image.open(buffer).convert("RGB").copy()
        return compressed_image


class RandomDownscaleRestore:
    def __init__(self, scales=(0.25, 0.50, 0.75, 1.00), probability=0.75):
        self.scales = scales
        self.probability = probability

    def __call__(self, image):
        if random.random() >= self.probability:
            return image

        width, height = image.size
        scale = random.choice(self.scales)
        if scale >= 1.0:
            return image

        small_width = max(16, int(width * scale))
        small_height = max(16, int(height * scale))

        # Use BILINEAR for both down and upscale to match the deterministic
        # evaluation pipeline (apply_advanced_tier_distortion, apply_resize).
        # Mixing BICUBIC/LANCZOS during training but BILINEAR at eval creates
        # mismatched ringing artifacts that the SRM branch can over-fit to.
        image = image.resize((small_width, small_height), Image.Resampling.BILINEAR)
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        return image


class RandomSharpen:
    def __init__(self, factor_range=(1.1, 1.5), probability=0.2):
        self.factor_range = factor_range
        self.probability = probability

    def __call__(self, image):
        if random.random() >= self.probability:
            return image
        factor = random.uniform(self.factor_range[0], self.factor_range[1])
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(factor)


class RandomGaussianNoise:
    def __init__(self, std_range=(2.0, 10.0), probability=0.1):
        self.std_range = std_range
        self.probability = probability

    def __call__(self, image):
        if random.random() >= self.probability:
            return image
        std = random.uniform(self.std_range[0], self.std_range[1])
        img_np = np.array(image).astype(float)
        noise = np.random.normal(0, std, img_np.shape)
        noisy = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)


class RandomMotionBlur:
    def __init__(self, sizes=(3, 5), probability=0.15):
        self.sizes = sizes
        self.probability = probability

    def __call__(self, image):
        if random.random() >= self.probability:
            return image
        size = random.choice(self.sizes)
        img_np = np.array(image)
        kernel = np.zeros((size, size))
        if random.random() > 0.5:
            kernel[int((size - 1) / 2), :] = np.ones(size)
        else:
            kernel[:, int((size - 1) / 2)] = np.ones(size)
        kernel /= size
        blurred = cv2.filter2D(img_np, -1, kernel)
        return Image.fromarray(blurred)


def get_transforms():
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    clean_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=3, translate=(0.02, 0.02), scale=(0.98, 1.02)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    standard_transform = transforms.Compose([
        transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.85, 1.0), ratio=(0.95, 1.05), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
        ], p=0.20),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # Forensic-preserving degradation pipeline (full severity; used at eval time and as the epoch-0 template)
    degradation_transform = _build_degradation_transform(severity_scale=1.0)

    evaluation_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    training_transforms = {
        "clean": clean_transform,
        "standard": standard_transform,
        "degradation": degradation_transform,
    }

    if config.TRAINING_STRATEGY not in training_transforms:
        raise ValueError(f"Invalid TRAINING_STRATEGY: {config.TRAINING_STRATEGY}")

    return training_transforms[config.TRAINING_STRATEGY], evaluation_transform


def _build_degradation_transform(severity_scale: float = 1.0):
    """Build the forensic-preserving degradation transform with a given severity_scale in [0, 1].

    All probabilities and intensity ranges are linearly interpolated between a minimum
    baseline and their full config values by severity_scale. This is used by the
    curriculum scheduler to produce progressively harder augmentations over training.

    Args:
        severity_scale: Float in [0, 1]. 1.0 = full config strength. 0.3 = gentle onset.
    """
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    s = float(severity_scale)

    def scale_prob(p_full: float, p_min: float = 0.05) -> float:
        """Scale a probability from p_min (at s=0) to p_full (at s=1)."""
        return p_min + s * (p_full - p_min)

    # JPEG quality: at low severity keep quality high (little compression); ramp down to full range
    jpeg_quality_min = int(config.AUG_JPEG_QUALITY_MIN + (1.0 - s) * (95 - config.AUG_JPEG_QUALITY_MIN))
    jpeg_quality_max = config.AUG_JPEG_QUALITY_MAX  # Max quality is always accessible

    # Downscale scales: at low severity only allow mild downscaling (0.75+)
    # We filter out very aggressive scales when severity is low.
    all_scales = list(config.AUG_DOWNSCALE_SCALES)
    if s < 0.5:
        curriculum_scales = [sc for sc in all_scales if sc >= 0.5] or [0.75, 1.0]
    elif s < 0.8:
        curriculum_scales = [sc for sc in all_scales if sc >= 0.25] or all_scales
    else:
        curriculum_scales = all_scales

    # Noise: ramp std range from near-zero to full
    noise_std_min = config.AUG_NOISE_STD_MIN * s
    noise_std_max = config.AUG_NOISE_STD_MAX * s
    if noise_std_min < 0.1:
        noise_std_min = 0.1
    if noise_std_max < noise_std_min:
        noise_std_max = noise_std_min

    return transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=config.AUG_COLOR_JITTER_BRIGHTNESS * s,
                contrast=config.AUG_COLOR_JITTER_CONTRAST * s,
                saturation=config.AUG_COLOR_JITTER_SATURATION * s,
            )
        ], p=scale_prob(config.AUG_COLOR_JITTER_PROB)),
        RandomDownscaleRestore(scales=curriculum_scales, probability=scale_prob(config.AUG_DOWNSCALE_PROB)),
        RandomJPEGCompression(quality_range=(jpeg_quality_min, jpeg_quality_max), probability=scale_prob(config.AUG_JPEG_PROB)),
        RandomMotionBlur(sizes=config.AUG_MOTION_BLUR_SIZES, probability=scale_prob(config.AUG_MOTION_BLUR_PROB)),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(config.AUG_BLUR_SIGMA_MIN, config.AUG_BLUR_SIGMA_MAX * s + 0.01))
        ], p=scale_prob(config.AUG_BLUR_PROB)),
        RandomSharpen(factor_range=(config.AUG_SHARPEN_FACTOR_MIN, config.AUG_SHARPEN_FACTOR_MAX), probability=scale_prob(config.AUG_SHARPEN_PROB)),
        RandomGaussianNoise(std_range=(noise_std_min, noise_std_max), probability=scale_prob(config.AUG_NOISE_PROB)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_degradation_transform_for_epoch(epoch: int, num_epochs: int) -> "transforms.Compose":
    """Return a degradation transform with curriculum-scaled severity for the given epoch.

    Severity ramps linearly from CURRICULUM_SEVERITY_START to 1.0 over the first
    CURRICULUM_RAMP_EPOCHS epochs, then holds at 1.0 for the remainder of training.
    Only active when config.CURRICULUM_ENABLED is True and TRAINING_STRATEGY == 'degradation'.
    Returns full-strength transform for all other strategies.

    Args:
        epoch: Current epoch (1-indexed).
        num_epochs: Total number of training epochs.

    Returns:
        A Compose transform appropriate for this epoch's curriculum stage.
    """
    if (
        not getattr(config, "CURRICULUM_ENABLED", False)
        or config.TRAINING_STRATEGY != "degradation"
    ):
        return _build_degradation_transform(severity_scale=1.0)

    ramp_epochs = getattr(config, "CURRICULUM_RAMP_EPOCHS", max(1, num_epochs // 2))
    severity_start = getattr(config, "CURRICULUM_SEVERITY_START", 0.3)

    # Linear ramp: 0-indexed epoch fraction within the ramp window
    if epoch <= ramp_epochs:
        frac = (epoch - 1) / max(1, ramp_epochs - 1)  # 0.0 at epoch 1 → 1.0 at epoch ramp_epochs
        severity = severity_start + frac * (1.0 - severity_start)
    else:
        severity = 1.0

    return _build_degradation_transform(severity_scale=severity)
