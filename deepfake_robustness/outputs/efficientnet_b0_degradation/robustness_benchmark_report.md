# Robustness Evaluation: efficientnet_b0

This report evaluates the performance degradation of the trained **efficientnet_b0** model under various image corruptions and perturbations. Evaluations are performed on the test split using validation-calibrated classification thresholds.

## Robustness Benchmark Table

| Corruption | Frame Acc | Frame BalAcc | Frame ROC-AUC | Video Acc | Video BalAcc | Video ROC-AUC |
|---|---|---|---|---|---|---|
| Clean | 0.6383 | 0.6303 | 0.6847 | 0.6481 | 0.6473 | 0.6886 |
| JPEG-100 | 0.6365 | 0.6288 | 0.6826 | 0.6438 | 0.6431 | 0.6853 |
| JPEG-90 | 0.6296 | 0.6223 | 0.6682 | 0.6266 | 0.6259 | 0.6701 |
| JPEG-70 | 0.6227 | 0.6137 | 0.6661 | 0.6223 | 0.6214 | 0.6767 |
| JPEG-50 | 0.5990 | 0.5882 | 0.6511 | 0.5966 | 0.5955 | 0.6719 |
| JPEG-30 | 0.5638 | 0.5512 | 0.6247 | 0.5494 | 0.5481 | 0.6437 |
| Resize-75% | 0.6220 | 0.6151 | 0.6524 | 0.6137 | 0.6130 | 0.6558 |
| Resize-50% | 0.5907 | 0.5841 | 0.6096 | 0.5837 | 0.5830 | 0.6116 |
| Resize-25% | 0.5576 | 0.5536 | 0.5645 | 0.5451 | 0.5446 | 0.5685 |
| Blur-Gaussian | 0.5647 | 0.5595 | 0.5679 | 0.5622 | 0.5617 | 0.5736 |
| Blur-Motion | 0.6144 | 0.6075 | 0.6479 | 0.6137 | 0.6130 | 0.6547 |
| Noise-Gaussian | 0.5720 | 0.5652 | 0.5879 | 0.5579 | 0.5572 | 0.6060 |
| Noise-SP | 0.6017 | 0.5953 | 0.6484 | 0.6009 | 0.6002 | 0.6639 |
| Brightness-Dark | 0.6107 | 0.6141 | 0.6732 | 0.6266 | 0.6267 | 0.6715 |
| Brightness-Bright | 0.5983 | 0.5868 | 0.6454 | 0.5923 | 0.5912 | 0.6634 |

## Strategic Insights & Observations

1. **Compression Degradation (JPEG):** Video compression is one of the most critical challenges in deployable deepfake systems. Lower quality levels (e.g. JPEG-30) throw away high-frequency DCT details, which typically causes a noticeable drop in frame-level AUC. Models trained with the *degradation* strategy should remain significantly more robust.
2. **Resolution Scaling:** Resizing images simulates different distances from the camera or post-process downscaling. A drop in input resolution (e.g. 25%) directly degrades classification performance, with larger models (like B4) suffering more if they rely heavily on fine resolution features.
3. **Noise and Blur:** Motion blur and Gaussian noise disrupt pixel-level distributions, validating the necessity of using video-level temporal aggregation to smooth out individual corrupted frames.
