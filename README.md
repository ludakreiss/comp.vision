# Deepfake Detection Robustness Benchmarking for Social-Media Content Moderation

[![University](https://img.shields.io/badge/Institution-University_of_Technology_Nuremberg_(UTN)-blue.svg)](https://www.utn.de/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Architecture](https://img.shields.io/badge/Model-EfficientNet--B0_Dual--Branch_Fusion-purple.svg)](model.py)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Empirical Benchmarking and Mitigation Framework for Lightweight Deepfake Detectors under Lossy Social-Media Transformations**
> *Developed at the University of Technology Nuremberg (UTN)*

---

## Executive Summary and Research Motivation

Lightweight Convolutional Neural Networks (CNNs) such as **EfficientNet-B0** offer high computational efficiency (**1.02 GFLOPs**, **~3.65 ms/frame** inference latency), making them strong candidates for real-time video content moderation in high-throughput social-media pipelines. However, standard deepfake detection architectures are predominantly trained and validated on uncompressed, high-resolution benchmark datasets.

When users upload media to messaging platforms or social networks (such as WhatsApp, Instagram, Telegram, or Facebook), automated ingestion pipelines execute aggressive lossy transformations — notably **lossy JPEG compression**, **spatial downscaling/sub-sampling**, **gaussian/motion blurs**, and **multi-stage re-encoding chains**. These lossy operations severely attenuate high-frequency spatial artifacts, resulting in significant accuracy degradation in standard deepfake detectors.

This project delivers a statistically rigorous empirical benchmarking framework and mitigation methodology designed to answer two central research questions:

1. **Performance Drop Quantification**: How severely do individual and chained lossy transformations degrade the classification accuracy, discrimination power (ROC-AUC), and probability calibration (ECE) of lightweight deepfake detectors?
2. **Robustness-Aware Mitigation**: Can robustness-aware training — coupling multi-scale high-pass spatial noise extractors, spatial-frequency cross-attention, forensic degradation augmentations, linear curriculum ramping, and Mixup regularization — recover performance across degraded media without sacrificing baseline speed or uncompressed accuracy?

---

## Architecture Overview: Dual-Branch Spatial-Frequency Fusion

The core model ([`model.py`](model.py)) couples spatial RGB representation learning with high-frequency residual noise analysis via a **Spatial-Frequency Cross-Attention (SFCA)** mechanism.

```mermaid
flowchart TD
    A["Input Face Crop: 3 x 256 x 256"] --> B["RGB Branch: EfficientNet-B0"]
    A --> C["Frequency Branch: Multi-Scale SRM Filters"]

    B -->|"Features: 1280 x 8 x 8"| D["Spatial-Frequency Cross-Attention (SFCA)"]

    C -->|"3x3, 5x5, 7x7 Kernels: 27 Channels"| E["Lightweight 2D Conv Extractor"]
    E -->|"Features: 128 x 16 x 16"| D

    D -->|"Queries: RGB, Keys/Values: Frequency"| F["Adaptive Global Average Pooling"]
    F -->|"Concatenated Representation: 1408-dim"| G["Enhanced Classifier Head"]
    G -->|"FC → BN → ReLU → Dropout 0.50 → FC"| H["Sigmoid Logit / Prediction Probability"]
```

### Key Architectural Components

1. **RGB Spatial Backbone**: ImageNet-pretrained **EfficientNet-B0** feature extractor returning a $1{,}280$-dimensional spatial map ($8 \times 8$). Progressive unfreezing (`FREEZE_PERCENT = 0.50`) optimizes transfer learning while preserving lower-level representations.
2. **Multi-Scale Spatial Rich Model (SRM) Layer (`MultiScaleSRMLayer`)**: Extracts high-frequency residual noise maps across three distinct spatial receptive fields ($3 \times 3$, $5 \times 5$, and $7 \times 7$), forming a 27-channel noise residual tensor targeting boundary artifacts and blending seams.
3. **Spatial-Frequency Cross-Attention (`SpatialFrequencyCrossAttention`)**: Cross-attends spatial RGB feature vectors ($Q$) with frequency residual vectors ($K, V$), dynamically steering network focus toward facial regions where high-frequency residual distributions exhibit manipulation artifacts.
4. **Enhanced Classifier Head (`EnhancedHead`)**: Dense projection ($1408 \to 512 \to 1$) incorporating Batch Normalization, ReLU activation, and Dropout ($0.50$) for strong regularization.

---

## Empirical Benchmarking Matrix and Key Results

Evaluated across **15 distinct degradation tiers** on the FaceForensics++ `c23` test split. Statistical validity is enforced using **95% Non-parametric Percentile Bootstrap Confidence Intervals** (1,000 resamples), **Expected Calibration Error (ECE)**, and **Paired Bootstrap Hypothesis Testing** ($p$-values) with Benjamini-Hochberg FDR correction.

### Degradation Benchmark Matrix (FaceForensics++ `c23` Test Split)

> **Note:** The benchmark matrix below is automatically generated by running `python evaluate.py --mode comparative --bootstraps 1000`. Full statistics and p-values are exported directly to `deepfake_robustness/outputs/comparative_robustness_report.md` and `deepfake_robustness/outputs/comparative_robustness_stat_results.csv`.

| Degradation Tier | Standard ROC-AUC (95% CI) | Robustness ROC-AUC (95% CI) | $\Delta$ ROC-AUC | Paired $p$-value |
| :--- | :---: | :---: | :---: | :---: |
| `clean` | *[TBD after final training]* | *[TBD]* | *[TBD]* | *[TBD]* |
| `weak_compression` ($Q=90$) | *[TBD]* | *[TBD]* | *[TBD]* | *[TBD]* |
| `medium_compression` ($Q=70$) | *[TBD]* | *[TBD]* | *[TBD]* | *[TBD]* |
| `strong_compression` ($Q=45$) | *[TBD]* | *[TBD]* | *[TBD]* | *[TBD]* |
| `extreme_compression` ($Q=30$) | *[TBD]* | *[TBD]* | *[TBD]* | *[TBD]* |
| `resize_75` ($0.75\times$) | *[TBD]* | *[TBD]* | *[TBD]* | *[TBD]* |
| `resize_50` ($0.50\times$) | *[TBD]* | *[TBD]* | *[TBD]* | *[TBD]* |
| `resize_25` ($0.25\times$) | *[TBD]* | *[TBD]* | *[TBD]* | *[TBD]* |
| `gaussian_blur` | *[TBD]* | *[TBD]* | *[TBD]* | *[TBD]* |
| `motion_blur` | *[TBD]* | *[TBD]* | *[TBD]* | *[TBD]* |
| `gaussian_noise` | *[TBD]* | *[TBD]* | *[TBD]* | *[TBD]* |
| `color_jitter` | *[TBD]* | *[TBD]* | *[TBD]* | *[TBD]* |
| `resize_50_compress_70` | *[TBD]* | *[TBD]* | *[TBD]* | *[TBD]* |
| `screenshot_recompress` | *[TBD]* | *[TBD]* | *[TBD]* | *[TBD]* |
| `social_media_pipeline` | *[TBD]* | *[TBD]* | *[TBD]* | *[TBD]* |

> [!IMPORTANT]
> Results will be populated after running the full training and evaluation pipeline. Run the commands in Steps 4 and 5 below.

---

## Complete Execution Guide and Command Reference

### Step 1: Environment and Dependency Setup

```bash
pip install -r requirements.txt
```

---

### Step 2: Dataset Acquisition and Setup

#### Option A: FaceForensics++ Dataset
Download raw FaceForensics++ video sequences (`c23` compression tier). Consult `python download-FaceForensics.py --help` for the full list of dataset choices.

```bash
python download-FaceForensics.py datasets/FaceForensics -t videos
```

#### Option B: Celeb-DF v2 Dataset

Download the full Celeb-DF v2 video dataset:
```bash
python download_celebdf.py
```

> **Official Celeb-DF Direct Links (Full Dataset)**:
> - **Google Drive**: [Celeb-DF (v2) Dataset](https://drive.google.com/open?id=1iLx76wsbi9itnkxSqz9BVBl4ZvnbIazj)
> - **Baidu Net Disk**: [Celeb-DF (v2) Dataset](https://pan.baidu.com/s/1EcYX0s4U3kbI1V2vdrP46A) *(Passcode: `yxa1`)*

---

### Step 3: MTCNN Face Extraction and Group Stratification

Extract aligned $256 \times 256$ face crops with 10% margin expansion, MTCNN landmark alignment, and zero-leakage group assignment:
```bash
python extract_faces.py
```

---

### Step 4: Model Training Execution

#### Train Standard Baseline Model (Clean Data Only)
```bash
python train.py --model efficientnet_b0 --strategy clean --epochs 30 --batch_size 32
```

#### Train Robustness-Aware Model (Simulated Degradations and Mixup)
```bash
python train.py --model efficientnet_b0 --strategy degradation --epochs 30 --batch_size 32
```

#### Advanced Training Options (Custom Hyperparameters)
```bash
python train.py \
    --model efficientnet_b0 \
    --strategy degradation \
    --epochs 40 \
    --batch_size 32 \
    --backbone_lr 4e-5 \
    --classifier_lr 1e-3
```

---

### Step 5: Comparative Benchmarking and Evaluation

#### Run Complete 15-Tier Degradation Benchmark Suite
```bash
python evaluate.py --mode comparative --bootstraps 1000
```

#### Run Quick Benchmarking Smoke-Test (Limited Batches)
```bash
python evaluate.py --mode comparative --limit_batches 10
```

#### Run Celeb-DF v2 Cross-Dataset Generalization Test
```bash
python evaluate.py --celebdf
```

---

### Step 6: Unit and Integration Testing

Execute the full test suite covering group-level dataset splitting, video stratification, zero frame-leakage between splits, SRM kernel verification, and end-to-end training smoke tests:
```bash
python -m pytest -q
```

---

### Step 7: Live Training and Metrics Monitoring (TensorBoard)

```bash
tensorboard --logdir=deepfake_robustness/outputs/runs
```

---

## Repository Sitemap and Structure

```
.
├── config.py                            # Central hyperparameters, split ratios (70/15/15), 15 degradation definitions
├── model.py                             # EfficientNet-B0 Dual-Branch model + Multi-Scale SRM Layer + Cross-Attention
├── dataset.py                           # PyTorch Dataset, group-level video leakage prevention, Celeb-DF manifest handling
├── transforms.py                        # Augmentation pipelines (Standard vs. Forensic Degradation transforms)
├── train.py                             # Training engine (AdamW, progressive unfreezing, Mixup, Cosine scheduler, EMA)
├── evaluate.py                          # Unified evaluation suite (15-tier benchmark matrix & Celeb-DF v2 generalization)
├── extract_faces.py                     # MTCNN face extraction and landmark alignment utility
├── metrics_utils.py                     # Evaluation metrics (AUC, ECE, Bootstrap 95% CIs, Youden's J calibration)
├── download-FaceForensics.py            # FaceForensics++ dataset downloader script
├── download_celebdf.py                  # Celeb-DF v2 dataset setup & manifest generator script
├── requirements.txt                     # Python package dependencies
├── LICENSE                              # MIT License
├── README.md                            # Primary documentation and execution guide
├── tests/
│   ├── test_pipeline.py                 # Core pipeline regression and smoke tests (27+ tests)
│   └── test_dataset.py                  # Dataset splitting and leakage prevention tests
├── .github/
│   └── workflows/ci.yml                # CI pipeline: lint, compile, test
└── deepfake_robustness/
    ├── ffpp_manifest.csv                # FaceForensics++ video metadata manifest (generated)
    └── outputs/                         # Model checkpoints, evaluation reports, and CSV statistics
```

---

## Methodological Protocols and Data Leakage Prevention

To ensure empirical validity and prevent evaluation bias:

1. **Group-Level Video Stratification**: Face crops inherit the unique source `video_id`. Strict group-based splitting ([`dataset.py`](dataset.py)) ensures frames from the same source video never span across training ($70\%$), validation ($15\%$), or test ($15\%$) sets.
2. **Decision Threshold Calibration**: Optimal binary classification thresholds are determined strictly on validation probabilities using **Youden's J statistic** ($\max(\text{TPR} - \text{FPR})$) and saved inside checkpoint dictionaries before testing.
3. **Linear Curriculum Degradation Ramping**: Augmentation severity ramps linearly from $30\%$ to $100\%$ over the initial epochs, stabilizing model convergence.
4. **Class-Balanced Loss Weighting**: Loss weights are adjusted by inverse effective sample frequency (`CB_BETA = 0.999`) to handle minor class imbalances.
5. **Paired Bootstrap Hypothesis Testing**: All comparative benchmark results use non-parametric paired bootstrap tests with Benjamini-Hochberg FDR multiple-testing correction across degradation tiers.

---

## License and Citation

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

If you find this repository, benchmark matrix, or architecture useful in your research, please cite:

```bibtex
@misc{utn_deepfake_robustness_2026,
  author = {University of Technology Nuremberg (UTN)},
  title = {Robustness Benchmarking for Lightweight Deepfake Detection in Social-Media Content Moderation},
  year = {2026},
  publisher = {GitHub},
  journal = {UTN Computer Vision \& Security Research Repository},
  url = {https://github.com/sandeep848/comp.vision}
}
```
