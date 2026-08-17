# Robustness Evaluation: efficientnet_b0

This report evaluates the performance degradation of the trained **efficientnet_b0** model under various image corruptions and perturbations. Evaluations are performed on the test split using validation-calibrated classification thresholds.

## Robustness Benchmark Table

| Corruption | Frame Acc | Frame BalAcc | Frame ROC-AUC | Video Acc | Video BalAcc | Video ROC-AUC |
|---|---|---|---|---|---|---|
| Clean | 0.6566 | 0.6507 | 0.7258 | 0.6567 | 0.6561 | 0.7103 |
| JPEG-100 | 0.6526 | 0.6479 | 0.7190 | 0.6395 | 0.6390 | 0.6996 |
| JPEG-90 | 0.6528 | 0.6461 | 0.7170 | 0.6524 | 0.6516 | 0.7051 |
| JPEG-70 | 0.6068 | 0.5954 | 0.6818 | 0.5880 | 0.5868 | 0.6865 |
| JPEG-50 | 0.5702 | 0.5560 | 0.6442 | 0.5708 | 0.5694 | 0.6491 |
| JPEG-30 | 0.5408 | 0.5248 | 0.6004 | 0.5365 | 0.5349 | 0.5992 |
| Resize-75% | 0.5734 | 0.5763 | 0.6244 | 0.5708 | 0.5710 | 0.6008 |
| Resize-50% | 0.5168 | 0.5249 | 0.5558 | 0.5193 | 0.5199 | 0.5390 |
| Resize-25% | 0.5000 | 0.5078 | 0.5270 | 0.5193 | 0.5199 | 0.5233 |
| Blur-Gaussian | 0.4929 | 0.5049 | 0.5183 | 0.5107 | 0.5117 | 0.5063 |
| Blur-Motion | 0.5939 | 0.5966 | 0.6346 | 0.6052 | 0.6053 | 0.6239 |
| Noise-Gaussian | 0.5202 | 0.5044 | 0.5229 | 0.5107 | 0.5088 | 0.5340 |
| Noise-SP | 0.6441 | 0.6397 | 0.6985 | 0.6438 | 0.6433 | 0.6887 |
| Brightness-Dark | 0.6107 | 0.6175 | 0.6949 | 0.6395 | 0.6400 | 0.6791 |
| Brightness-Bright | 0.6169 | 0.6045 | 0.7109 | 0.6137 | 0.6125 | 0.7231 |

## Strategic Insights & Observations

1. **Compression Degradation (JPEG):** Video compression is one of the most critical challenges in deployable deepfake systems. Lower quality levels (e.g. JPEG-30) throw away high-frequency DCT details, which typically causes a noticeable drop in frame-level AUC. Models trained with the *degradation* strategy should remain significantly more robust.
2. **Resolution Scaling:** Resizing images simulates different distances from the camera or post-process downscaling. A drop in input resolution (e.g. 25%) directly degrades classification performance, with larger models (like B4) suffering more if they rely heavily on fine resolution features.
3. **Noise and Blur:** Motion blur and Gaussian noise disrupt pixel-level distributions, validating the necessity of using video-level temporal aggregation to smooth out individual corrupted frames.
