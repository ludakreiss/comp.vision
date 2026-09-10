
# Deepfake Detection Robustness Benchmarking for Social-Media Content Moderation

[![University](https://img.shields.io/badge/Institution-University_of_Technology_Nuremberg_(UTN)-blue.svg)](https://www.utn.de/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Architecture](https://img.shields.io/badge/Model-EfficientNet--B0_Dual--Branch_Fusion-purple.svg)](model.py)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Robust Deepfake Detection Under Compression and Resizing Using Lightweight CNNs**
>
> Developed at the University of Technology Nuremberg (UTN)

---

## 📌 Project Overview

This project studies how common image degradations affect lightweight deepfake detectors and whether robustness-aware training can reduce the resulting performance loss.

The main experiment uses FaceForensics++ as the training and primary evaluation dataset. Two versions of the same detector are trained:

1. **Clean / Standard Model** — trained using only clean aligned face crops.
2. **Robustness-Aware Model** — trained using synthetic degradations such as JPEG compression, resizing, blur, noise, and color changes.

Both models are evaluated on the same clean and degraded test samples.

Celeb-DF v2 is used only as an **external generalization dataset**. It is not used for training or validation.

---

## 🧠 Main Architecture

The primary model is implemented in [`model.py`](model.py) and uses a dual-branch spatial-frequency architecture.

### RGB Branch

The RGB branch uses an ImageNet-pretrained **EfficientNet-B0** backbone.

Input:

```text
256 × 256 RGB face crop
```

The backbone produces spatial features describing facial appearance and semantic structure.

### Frequency Branch

A Multi-Scale SRM frequency branch applies fixed high-pass filters to expose high-frequency image artifacts that may be introduced by face manipulation.

The branch uses:

- 3×3 SRM filters
- 5×5 SRM filters
- 7×7 SRM filters
- lightweight convolutional feature extraction

The SRM filters are fixed and are not learned during training.

### Spatial-Frequency Cross-Attention

The main fusion model uses Spatial-Frequency Cross-Attention (SFCA):

```
RGB features        → Query
Frequency features  → Key / Value
```

This allows spatial RGB features to selectively use relevant frequency-domain evidence before classification.

### Classification

The fused representation is passed through a lightweight classifier that outputs one raw binary logit:

- `real = 0`
- `fake = 1`

Sigmoid is applied outside the model when converting logits into probabilities.

---

## 🧩 Available Model Variants

The repository currently supports four model variants:

### `fusion`

Primary dual-branch architecture:

```
EfficientNet-B0 RGB branch
        +
Multi-Scale SRM frequency branch
        ↓
Spatial-Frequency Cross-Attention
        ↓
Binary classifier
```

### `rgb_only`

Uses only the EfficientNet-B0 RGB branch.

This variant is useful as an ablation to measure how much the frequency branch contributes.

### `fusion_no_attn`

Uses both RGB and frequency features but removes cross-attention.

This allows evaluation of whether SFCA itself improves performance.

### `modular_order`

An extended CoRe-DF-style architecture that adds explicit degradation-order reasoning.

The implementation uses components from `core_df.py`:

- `ModularExpertHead`
- `DynamicRouter`
- `OrderInferenceModule`
- `OrderConditionedFusion`

The model can reason about combinations and ordering of degradations and dynamically route information through specialized expert modules.

The `modular_order` variant is currently treated as an extension of the primary fusion benchmark architecture.

---

## 🔄 Compositional Degradation Support

The repository also supports ordered combinations of degradations.

`pipeline_registry.py` constructs degradation pipelines such as:

```
resize_50 -> gaussian_blur -> jpeg
```

Order matters because:

```
resize -> blur -> jpeg
```

is not necessarily equivalent to:

```
jpeg -> blur -> resize
```

`transforms.py` includes `OrderPreservingDegradationTransform`, while `dataset.py` includes `CompositionalDegradationDataset`.

These components support experiments involving combinations and ordering of realistic social-media transformations.

---

## 📂 Dataset Structure

The repository currently uses the following layout:

```
comp.vision/
│
├── datasets/
│   │
│   ├── FaceForensics/
│   │   ├── original_sequences/
│   │   └── manipulated_sequences/
│   │
│   ├── processed_faces/
│   │   └── ffpp_c23/
│   │       ├── original/
│   │       ├── Deepfakes/
│   │       ├── Face2Face/
│   │       ├── FaceSwap/
│   │       └── NeuralTextures/
│   │
│   └── Celeb-DF-v2/
│       ├── Celeb-real/
│       ├── Celeb-synthesis/
│       ├── YouTube-real/
│       ├── List_of_testing_videos.txt
│       ├── processed_faces/
│       └── processed_faces_aligned/
│
├── manifests/
│   ├── ffpp_manifest.csv
│   └── celebdf_manifest.csv
│
└── outputs/
```

### FaceForensics++

The primary experiment uses FaceForensics++ with the c23 compression tier.

Manipulation categories:

- Deepfakes
- Face2Face
- FaceSwap
- NeuralTextures

Labels:

- `Original = 0`
- `Manipulated = 1`

Face crops are stored under:

```
datasets/processed_faces/ffpp_c23/
```

### Celeb-DF v2

Celeb-DF is used only for external evaluation.

Raw videos are stored under:

```
datasets/Celeb-DF-v2/
```

Only videos from:

```
datasets/Celeb-DF-v2/List_of_testing_videos.txt
```

are used for external testing.

Aligned Celeb-DF faces are stored under:

```
datasets/Celeb-DF-v2/processed_faces_aligned/
```

The older:

```
datasets/Celeb-DF-v2/processed_faces/
```

directory contains raw sampled frames from an earlier preprocessing implementation and is not used for the final aligned-face evaluation pipeline.

---

## 🖼️ Face Extraction and Alignment

Face extraction is implemented in `extract_faces.py`.

The pipeline uses:

- MTCNN face detection
- facial landmark detection
- eye-based affine alignment
- 256×256 output resolution
- minimum face-size filtering
- detection-confidence filtering
- blur filtering using Laplacian variance
- spaced frame sampling

The goal is to make the FF++ and Celeb-DF model inputs consistent.

### FaceForensics++ Extraction

To process FaceForensics++:

```bash
python extract_faces.py
```

For a small debugging run:

```bash
python extract_faces.py --sanity
```

The extraction pipeline generates aligned faces and creates:

```
manifests/ffpp_manifest.csv
```

The FF++ manifest contains information including:

- image path
- label
- video ID
- group ID
- train / validation / test split

Splitting is performed at the video-group level to prevent leakage between related original and manipulated videos.

### Celeb-DF Extraction

Celeb-DF uses the same MTCNN and alignment pipeline as FF++.

Run:

```bash
python extract_faces.py --celebdf
```

This:

- reads the official Celeb-DF test list
- processes only those videos
- extracts aligned 256×256 face crops
- stores them under `datasets/Celeb-DF-v2/processed_faces_aligned/`
- generates `manifests/celebdf_manifest.csv`

Celeb-DF labels are:

- `Celeb-real = 0`
- `YouTube-real = 0`
- `Celeb-synthesis = 1`

Celeb-DF remains an external dataset and is never used for FF++ model training or validation.

---

## 🔒 Leakage Prevention

`dataset.py` performs group-aware splitting using connected components / Union-Find.

For example:

- Original video `000`
- Original video `003`
- Manipulated video `000_003`

are assigned to the same group.

This prevents:

- frames from one video appearing across multiple splits
- manipulated videos being separated from their related source videos
- train/test contamination

The target split is:

| Split | Proportion |
|---|---|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

The exact number of frames in each split can differ slightly because assignment is performed at group level.

If the data cannot form valid disjoint groups, the code raises an error rather than intentionally allowing leakage.

---

## 🧪 Training

Training is implemented in `train.py`.

Main settings include:

- EfficientNet-B0
- ImageNet initialization
- AdamW optimizer
- differential backbone/classifier learning rates
- learning-rate warmup
- cosine annealing
- binary focal loss
- Mixup
- EMA model weights
- progressive backbone unfreezing
- early stopping
- validation-based checkpoint selection
- F1-optimal threshold calibration
- top-checkpoint averaging

ImageNet pretrained weights are initialization only. They do not resume a previous deepfake checkpoint.

### Fresh Clean Model Training

To train the primary fusion architecture from a fresh run:

```bash
OUTPUT_ROOT=outputs/fresh_clean \
python train.py \
  --model efficientnet_b0 \
  --variant fusion \
  --strategy clean \
  --epochs 30 \
  --batch_size 32 \
  --fresh
```

The clean model receives no synthetic degradation augmentation.

### Fresh Robustness-Aware Training

```bash
OUTPUT_ROOT=outputs/fresh_degradation \
python train.py \
  --model efficientnet_b0 \
  --variant fusion \
  --strategy degradation \
  --epochs 30 \
  --batch_size 32 \
  --fresh
```

The robustness-aware model uses synthetic degradations during training.

The architecture and other major training settings are kept the same so that the primary experimental variable is the training transformation strategy.

### Modular-Order Training

The extended architecture can also be trained separately.

Clean:

```bash
python train.py \
  --model efficientnet_b0 \
  --variant modular_order \
  --strategy clean \
  --epochs 30 \
  --batch_size 32 \
  --fresh
```

Robust:

```bash
python train.py \
  --model efficientnet_b0 \
  --variant modular_order \
  --strategy degradation \
  --epochs 30 \
  --batch_size 32 \
  --fresh
```

---

## 📉 Robustness-Aware Augmentation

The degradation strategy can include:

- JPEG compression
- spatial resizing
- Gaussian blur
- motion blur
- Gaussian noise
- color jitter
- sharpening
- combined degradation pipelines

A curriculum schedule gradually increases degradation severity during the early training epochs. This prevents the model from being exposed only to strong corruptions at the beginning of training.

---

## 🎯 Threshold Calibration

After training, the binary decision threshold is calibrated using the validation split.

The selected threshold is:

```
t* = argmax F1(t)
```

In other words, the threshold that maximizes validation F1 is selected. 
The threshold is stored with the model checkpoint and later reused during testing.

Celeb-DF is not used for threshold calibration.

---

## 📊 Evaluation

Evaluation is implemented in `evaluate.py`.

The main comparative benchmark evaluates the clean and robustness-aware models on the same test images.

Run:

```bash
python evaluate.py --mode comparative --bootstraps 1000
```

The benchmark contains 15 degradation tiers:

1. clean
2. weak JPEG compression
3. medium JPEG compression
4. strong JPEG compression
5. extreme JPEG compression
6. resize ×0.75
7. resize ×0.50
8. resize ×0.25
9. Gaussian blur
10. motion blur
11. Gaussian noise
12. color jitter
13. resize ×0.50 + JPEG Q70
14. screenshot/recompression simulation
15. social-media transformation pipeline

### Evaluation Metrics

Metrics are calculated at both frame and video level where applicable.

Reported metrics include:

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- Expected Calibration Error (ECE)

Video-level predictions are calculated by averaging frame-level fake probabilities for each video.

### Statistical Analysis

The evaluation framework supports:

- 95% non-parametric percentile bootstrap confidence intervals
- video-level bootstrap resampling
- paired bootstrap significance testing
- Benjamini-Hochberg False Discovery Rate correction

These tools allow performance changes between the clean and robustness-aware models to be analyzed statistically rather than relying only on single metric values.

---

## 🌍 Celeb-DF External Generalization

Celeb-DF is used only after the FF++ models have been trained and selected.

The workflow is:

```
FaceForensics++
      ↓
Train / Validation / Test
      ↓
Train Clean + Robust Models
      ↓
Primary FF++ Robustness Evaluation
      ↓
Freeze Models
      ↓
Celeb-DF External Evaluation
```

The models are not retrained or fine-tuned on Celeb-DF.

Before evaluation, verify the current Celeb-DF CLI using:

```bash
python evaluate.py --help
```

The current repository supports a Celeb-DF evaluation mode through `evaluate.py`.

---

## 🔥 Grad-CAM Explainability

Grad-CAM support is included for qualitative analysis. It can be used to visualize which image regions influence the model's prediction.

Grad-CAM supports:

- clean vs degraded image comparisons
- clean-model vs robust-model comparisons
- RGB branch visualization
- frequency-related visualization where supported

Grad-CAM is used only as a post-hoc interpretability tool. It does not identify the true manipulation boundary and should not be interpreted as a segmentation map.

---

## 🖥️ Gradio Demo

The repository contains an interactive Gradio application under:

```
demo/
├── app.py
└── inference.py
```

Install the required dependencies:

```bash
pip install -r requirements.txt
pip install gradio
```

Then launch the demo from the repository root:

```bash
python demo/app.py
```

By default, Gradio is available at:

```
http://127.0.0.1:7860
```

The inference wrapper supports the standard model output format and the dictionary output used by the `modular_order` architecture.

---

## 📈 TensorBoard

Training logs are written under `outputs/`.

To monitor training:

```bash
tensorboard --logdir=outputs
```

If using a separate fresh output root, point TensorBoard to that directory instead. For example:

```bash
tensorboard --logdir=outputs/fresh_clean
```

---

## ✅ Testing

The project includes unit, integration, leakage, model, and smoke tests under:

```
tests/
├── test_dataset.py
├── test_pipeline.py
└── test_smoke.py
```

Run:

```bash
PYTHONPATH=. ./venv/bin/pytest -q
```

You can also verify that the Python source compiles with:

```bash
python -m compileall .
```

---

## 📂 Repository Sitemap

```
comp.vision/
├── config.py
├── core_df.py
├── dataset.py
├── download_celebdf.py
├── download-FaceForensics.py
├── extract_faces.py
├── train.py
├── evaluate.py
├── model.py
├── transforms.py
├── metrics_utils.py
├── pipeline_registry.py
├── requirements.txt
├── README.md
├── Phase_5_Final_Report.md
├── LICENSE
│
├── demo/
│   ├── app.py
│   └── inference.py
│
├── datasets/
│   ├── FaceForensics/
│   │   ├── original_sequences/
│   │   └── manipulated_sequences/
│   │
│   ├── processed_faces/
│   │   └── ffpp_c23/
│   │
│   └── Celeb-DF-v2/
│       ├── Celeb-real/
│       ├── Celeb-synthesis/
│       ├── YouTube-real/
│       ├── List_of_testing_videos.txt
│       ├── processed_faces/
│       └── processed_faces_aligned/
│
├── manifests/
│   ├── ffpp_manifest.csv
│   └── celebdf_manifest.csv
│
├── outputs/
│   ├── efficientnet_b0_clean/
│   ├── efficientnet_b0_degradation/
│   ├── predictions/
│   ├── runs/
│   └── Grad-CAM / evaluation outputs
│
└── tests/
    ├── test_dataset.py
    ├── test_pipeline.py
    └── test_smoke.py
```

---

## 🧭 Experimental Workflow

The intended experimental sequence is:

1. Download FaceForensics++ c23 videos
2. Extract aligned 256×256 faces
3. Generate zero-leakage FF++ manifest
4. Train clean model
5. Train robustness-aware model
6. Evaluate both on the same FF++ test samples
7. Evaluate across 15 degradation tiers
8. Perform statistical comparison
9. Evaluate frozen models on Celeb-DF
10. Use Grad-CAM for qualitative analysis

The optional `modular_order` model extends this experiment by explicitly modeling compositional degradation order and expert routing.

---

## 📝 License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.