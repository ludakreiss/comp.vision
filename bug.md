# Comprehensive Technical Audit: 27 Identified Bugs & Codebase Defects

> **Project**: Deepfake Detection Under Compression & Resizing  
> **Workspace**: `/home/utn/uzis83et/CV`  
> **Audit Status**: Complete (27 Defect Reports)

---

## 1. Model Architecture & Spatial-Frequency Fusion (`model.py`)

### Bug 1: Spatial-Frequency Cross-Attention Non-Contiguous Memory Reshape
* **Location:** [model.py:L145-L151](file:///home/utn/uzis83et/CV/model.py#L145-L151)
* **Code Snippet:**
  ```python
  out_attn = F.scaled_dot_product_attention(Q, K, V)
  context = out_attn.permute(0, 2, 1, 3).contiguous().view(B, N, self.embed_dim).permute(0, 2, 1).contiguous().view(B, self.embed_dim, H_rgb, W_rgb)
  ```
* **Root Cause:** `out_attn` produces shape `[B, heads, N, head_dim]`. The sequence of `.permute()` and `.view()` assumes linear memory order across spatial and attention dimensions. If spatial tensor dimensions $N = H \times W$ are permuted without re-ordering, channels risk being transposed out of sequence.
* **Impact:** Distorts spatial cross-attention feature alignment between spatial RGB maps and frequency residual maps.

### Bug 2: SRM High-Pass Preprocessing Scale Mismatch
* **Location:** [model.py:L282](file:///home/utn/uzis83et/CV/model.py#L282), [model.py:L291](file:///home/utn/uzis83et/CV/model.py#L291), [model.py:L302](file:///home/utn/uzis83et/CV/model.py#L302)
* **Code Snippet:**
  ```python
  x_raw = x * self.std + self.mean
  freq_x = self.srm(x_raw)
  ```
* **Root Cause:** `x_raw = x * self.std + self.mean` restores pixel values to $[0.0, 1.0]$. However, spatial derivative SRM filters in [MultiScaleSRMLayer](file:///home/utn/uzis83et/CV/model.py#L8-L108) were mathematically formulated for standard $[0, 255]$ pixel ranges.
* **Impact:** Applying fixed normalization divisors to $[0, 1]$ inputs squashes gradient magnitudes by a factor of 255, severely attenuating high-frequency noise residual traces.

### Bug 3: Spatial Feature Resolution Mismatch in SFCA Interpolation
* **Location:** [model.py:L137-L140](file:///home/utn/uzis83et/CV/model.py#L137-L140)
* **Code Snippet:**
  ```python
  if freq_map.shape[2:] != (H_rgb, W_rgb):
      f_map = F.interpolate(freq_map, size=(H_rgb, W_rgb), mode='bilinear', align_corners=False)
  ```
* **Root Cause:** `freq_convs` applies 4 strided convolutions, forcing frequency maps to $16 \times 16$. Backbones such as `resnet18` or `convnext_tiny` output $8 \times 8$ feature maps at $256 \times 256$ input resolution.
* **Impact:** Interpolating `freq_map` bilinear to $8 \times 8$ causes spatial resolution degradation and misalignment with key projections.

### Bug 4: Redundant Backbone Instantiation in Frequency-Only Branch
* **Location:** [model.py:L189-L221](file:///home/utn/uzis83et/CV/model.py#L189-L221)
* **Root Cause:** When `branch_mode == "freq"`, `self.rgb_backbone` is fully instantiated with pretrained weights and frozen parameters in `__init__`.
* **Impact:** Consumes unnecessary GPU memory and initialization overhead even though RGB backbone features are completely bypassed in `forward()`.

### Bug 27 (CRITICAL): Non-Zero-Sum SRM $7 \times 7$ High-Pass Filter Coefficients
* **Location:** [model.py:L47-L74](file:///home/utn/uzis83et/CV/model.py#L47-L74)
* **Empirical Log Evidence:** Fails `test_srm_zero_sum_kernels_and_frozen_weights` unit test (`AssertionError: f1_7 sum is 0.1017, expected 0.0`).
* **Root Cause:** High-pass residual filters in spatial forensic models must satisfy the zero-sum constraint ($\sum_{i,j} W_{i,j} = 0$) so that flat image regions produce zero response. All three $7 \times 7$ SRM filter kernels in `model.py` violate this property:
  1. `f1_7`: Center element is set to `+15` instead of `+12`. Total filter sum = **$+3.0$**.
  2. `f2_7`: Center row coefficients `[2, -4, 6, -4, 6, -4, 2]` sum to $+4$. Total filter sum = **$+4.0$**.
  3. `f3_7`: Center row `[2, 4, 6, -24, 6, 4, 2]` sums to $+6$. Total filter sum = **$+6.0$**.
* **Impact:** Converts the $7 \times 7$ SRM frequency branch into DC-leaky low-pass filters. Low-frequency scene content (lighting, skin tone, background colors) leaks directly into frequency feature maps, defeating spatial-frequency noise isolation.

---

## 2. Training Loop, Optimizer & Checkpointing (`train.py`)

### Bug 5: Classifier Parameter Group Misassignment on Checkpoint Resume
* **Location:** [train.py:L707-L712](file:///home/utn/uzis83et/CV/train.py#L707-L712)
* **Root Cause:** `build_optimizer` splits model parameters into 4 groups (`decay_backbone`, `no_decay_backbone`, `decay_classifier`, `no_decay_classifier`). When resuming training, dynamically added parameter groups default to `current_backbone_lr` (`4e-5`) and `config.WEIGHT_DECAY` (`1e-2`).
* **Impact:** Misassigns classifier head parameters to the lower backbone learning rate (`4e-5` instead of `1e-3`).

### Bug 6: `IndexError` in `sync_scheduler_param_groups`
* **Location:** [train.py:L283-L290](file:///home/utn/uzis83et/CV/train.py#L283-L290)
* **Code Snippet:**
  ```python
  while hasattr(sub, "base_lrs") and len(sub.base_lrs) < target_len:
      sub.base_lrs.append(optimizer.param_groups[len(sub.base_lrs)]["lr"])
  ```
* **Root Cause:** `sync_scheduler_param_groups` accesses `optimizer.param_groups[len(sub.base_lrs)]`. If new parameter groups are added during progressive unfreezing and `len(sub.base_lrs)` equals `len(optimizer.param_groups)`, indexing triggers `IndexError: list index out of range`.

### Bug 7: PyTorch AMP GradScaler `update()` Called Without `step()` During Accumulation
* **Location:** [train.py:L336-L341](file:///home/utn/uzis83et/CV/train.py#L336-L341)
* **Root Cause:** During gradient accumulation steps (`is_accumulating == True`), `scaler.unscale_()` and `scaler.step()` are bypassed. If `scaler.update()` is called without `scaler.step()`, PyTorch `GradScaler` emits runtime warnings or crashes on skipped steps.

### Bug 8: Unhandled Zero-Division Risk in Loss & Metric Computation
* **Location:** [train.py:L367](file:///home/utn/uzis83et/CV/train.py#L367), [train.py:L420](file:///home/utn/uzis83et/CV/train.py#L420)
* **Code Snippet:**
  ```python
  metrics["loss"] = running_loss / len(all_labels) if all_labels else 0.0
  ```
* **Root Cause:** If `train_loader` or `evaluate_model` is executed with `limit_batches=0` or an empty dataset slice, `len(all_labels)` is 0, raising `ZeroDivisionError: float division by zero`.

### Bug 9: EMA Shadow Weight Out-of-Sync Risk during Progressive Unfreezing
* **Location:** [train.py:L789-L790](file:///home/utn/uzis83et/CV/train.py#L789-L790)
* **Root Cause:** If `apply_shadow()` is called before `register_new_parameters()` runs after progressive unfreezing, `ema.backup` lacks keys for newly unfrozen layers.

### Bug 23: Learning Rate Decay Discrepancy on Progressive Unfreezing
* **Location:** [train.py:L775-L787](file:///home/utn/uzis83et/CV/train.py#L775-L787) vs [train.py:L283-L290](file:///home/utn/uzis83et/CV/train.py#L283-L290)
* **Root Cause:** When layers are unfrozen midway through training (e.g. epoch 6), `optimizer.add_param_group()` adds a group with the *decayed current backbone LR* (`3.2e-5`). `sync_scheduler_param_groups()` appends this already-decayed LR to `scheduler.base_lrs`, causing `CosineAnnealingLR` to treat `3.2e-5` as the initial base LR for newly unfrozen layers.
* **Impact:** Newly unfrozen layers decay from a lower initial learning rate than the rest of the backbone.

### Bug 24: Unscaled AMP GradScaler in LRFinder Range Test
* **Location:** [train.py:L106-L132](file:///home/utn/uzis83et/CV/train.py#L106-L132)
* **Root Cause:** `LRFinder.range_test()` executes `with torch.amp.autocast(device_type="cuda")` without `GradScaler`.
* **Impact:** On FP16 CUDA hardware, underflowing gradients during LR sweeps cause loss to explode to NaN, aborting the LR finder test early.

### Bug 25: Double Loss Penalty in `DeepfakeLoss`
* **Location:** [train.py:L48-L63](file:///home/utn/uzis83et/CV/train.py#L48-L63)
* **Code Snippet:**
  ```python
  if self.class_weights is not None:
      bce_loss = bce_loss * weight_t
  ...
  loss = alpha_t * focal_weight * bce_loss
  ```
* **Root Cause:** When `class_weights` is provided alongside `loss_type="focal"`, `bce_loss` is multiplied by `weight_t` at line 53, and then multiplied by `alpha_t` and `focal_weight` at line 61.
* **Impact:** Double-penalizes class imbalance, destabilizing gradient updates on minority samples.

---

## 3. Dataset Topology, Splitting & Manifest Generation (`dataset.py`)

### Bug 21 (CRITICAL): Celeb-DF Identity Graph Collapse via Sequence Number Collisions
* **Location:** [dataset.py:L78-L84](file:///home/utn/uzis83et/CV/dataset.py#L78-L84)
* **Code Snippet:**
  ```python
  is_celebdf = (id1.startswith('id') and id1[2:].isdigit()) and (id2.startswith('id') and id2[2:].isdigit() or id2.isdigit())
  if is_ffpp or is_celebdf:
      uf.union(id1, id2)
  ```
* **Root Cause:** Celeb-DF real videos follow `idX_YYYY.mp4` (`idX` = celeb ID, `YYYY` = video index, e.g. `id0_0000.mp4`). In `build_connected_groups()`, `is_celebdf` evaluates `or id2.isdigit()`. Because `id2` (`'0000'`) is numeric, `uf.union('id0', '0000')`, `uf.union('id1', '0000')`, and `uf.union('id2', '0000')` link distinct celeb identities together via the shared sequence index `'0000'`.
* **Impact:** Collapses all Celeb-DF identities into **1 single connected component**, causing `assign_group_splits` to fail with `ValueError: Dataset has too few unique source videos to produce three non-empty splits` or placing all data in train.

### Bug 10: Stratified Group Split Leakage on Single-Group Buckets
* **Location:** [dataset.py:L147-L156](file:///home/utn/uzis83et/CV/dataset.py#L147-L156)
* **Root Cause:** In `_split_bucket()`, if a label bucket contains $n=1$ group, it returns `list(bucket), list(bucket), list(bucket)` (assigning the group to train, val, AND test simultaneously).
* **Impact:** Triggers `ValueError: Split leakage detected` at lines 169–173 whenever subset manifests are processed.

### Bug 11: Celeb-DF Manifest Directory Hierarchy Parser Assumption
* **Location:** [dataset.py:L231](file:///home/utn/uzis83et/CV/dataset.py#L231)
* **Root Cause:** `rel_video_path = f"{category_name}/{item.parent.name}.mp4"` assumes image crops are stored in subfolders named after the video ID. If crops reside directly in the category folder, `item.parent.name` resolves to the category directory instead of the video ID.

---

## 4. Evaluation & Statistical Metrics (`evaluate.py` & `metrics_utils.py`)

### Bug 22: Non-Deterministic `sample_id` Hash Fallback in Distortion Tiers
* **Location:** [evaluate.py:L32-L34](file:///home/utn/uzis83et/CV/evaluate.py#L32-L34)
* **Code Snippet:**
  ```python
  seed_str = f"{global_seed}_{sample_id}_{tier_cfg.get('jpeg_quality')}..."
  seed_val = int(hashlib.md5(seed_str.encode("utf-8")).hexdigest()[:8], 16)
  ```
* **Root Cause:** When `image_path` is empty, `TierDistortionModifier` falls back to `str(id(img))`. In Python, `id(object)` returns the memory address of the object, which varies across process runs.
* **Impact:** Random distortion numbers generated for color jitter and noise differ across execution runs, destroying evaluation determinism.

### Bug 26: Redundant Multi-Pass Chained Tier Attribute Shadowing
* **Location:** [evaluate.py:L37-L58](file:///home/utn/uzis83et/CV/evaluate.py#L37-L58) vs [config.py:L108-L109](file:///home/utn/uzis83et/CV/config.py#L108-L109)
* **Root Cause:** For chained distortion matrix tiers (`"screenshot_recompress"`, `"social_media_pipeline"`), `apply_advanced_tier_distortion` executes the `chain` loop and returns early (`return current_img`). Top-level dict keys in `config.DEGRADATION_TIERS` (such as `"resize_scale": 0.50` or `"jpeg_quality": 70`) are completely un-evaluated.

### Bug 12: Missing `model_variant` Argument in Celeb-DF Benchmark Loader
* **Location:** [evaluate.py:L301](file:///home/utn/uzis83et/CV/evaluate.py#L301), [evaluate.py:L312](file:///home/utn/uzis83et/CV/evaluate.py#L312)
* **Root Cause:** `run_celebdf_eval()` calls `build_model(config.MODEL_NAME, pretrained=False)` without passing `model_variant=config.MODEL_VARIANT`. It defaults to `"fusion"`.
* **Impact:** If the checkpoint was trained under `"rgb_only"` or `"fusion_no_attn"`, `load_state_dict()` crashes with a `RuntimeError: Error(s) in loading state_dict`.

### Bug 13: Unhandled Single-Class Exception in Bootstrap ROC-AUC Resampling
* **Location:** [metrics_utils.py:L162-L163](file:///home/utn/uzis83et/CV/metrics_utils.py#L162-L163)
* **Root Cause:** In `paired_bootstrap_test`, `roc_auc_score(sub_true, sub_std)` is calculated on random bootstrap samples. Although `len(np.unique(sub_true)) < 2` is checked on `sub_true`, small subsample slices can still sample only positive or negative instances, throwing `ValueError: Only one class present in y_true`.

### Bug 14: Adaptive ECE Quantile Binning Collapse
* **Location:** [metrics_utils.py:L34-L38](file:///home/utn/uzis83et/CV/metrics_utils.py#L34-L38)
* **Root Cause:** When predictions are heavily confident (e.g. many `0.00` or `1.00` probabilities), `np.quantile` yields duplicate bin boundaries. `np.unique` truncates `bin_boundaries`. If `len(bin_boundaries) == 2`, `n_bins` becomes 1, which can lead to index bounds misalignment.

---

## 5. Transforms, Preprocessing & Augmentations (`transforms.py` & `config.py`)

### Bug 15: Image Resolution Overriding Filter Artifacts
* **Location:** [config.py:L47](file:///home/utn/uzis83et/CV/config.py#L47) vs [train.py:L548-L553](file:///home/utn/uzis83et/CV/train.py#L548-L553)
* **Root Cause:** Face extraction crops faces at $256 \times 256$. If `train.py` dynamically resizes images to $224 \times 224$ during dataloading, the bicubic interpolation step acts as a low-pass filter, erasing subtle high-frequency grid artifacts and SRM noise residuals.

### Bug 16: Asymmetric Motion Blur Kernel on Even Filter Sizes
* **Location:** [transforms.py:L91-L94](file:///home/utn/uzis83et/CV/transforms.py#L91-L94)
* **Code Snippet:**
  ```python
  kernel[int((size - 1) / 2), :] = np.ones(size)
  ```
* **Root Cause:** `int((size - 1) / 2)` for even kernel sizes (e.g. `size = 4`) computes index `1`, resulting in an asymmetric 2D filter kernel that introduces spatial pixel shift artifacts into augmented frames.

---

## 6. Face Extraction, Downloader Scripts & Environment (`extract_faces.py`, `download_celebdf.py`, `tests/`)

### Bug 17: `TypeError` Risk in `choose_dominant_face`
* **Location:** [extract_faces.py:L34](file:///home/utn/uzis83et/CV/extract_faces.py#L34)
* **Root Cause:** If MTCNN returns `None` entries in `probabilities`, calling `float(probability)` without explicit `None` checks raises `TypeError: float() argument must be a string or a real number, not 'NoneType'`.

### Bug 18: Unclosed OpenCV Video Capture File Handles
* **Location:** [extract_faces.py:L123-L198](file:///home/utn/uzis83et/CV/extract_faces.py#L123-L198)
* **Root Cause:** `cv2.VideoCapture` is initialized without a `try...finally` block. If an unhandled exception occurs during face extraction, video file handles remain open.

### Bug 19: `gdown` Python API Parameter Deprecation in Download Script
* **Location:** [download_celebdf.py:L39](file:///home/utn/uzis83et/CV/download_celebdf.py#L39)
* **Root Cause:** Modern `gdown` releases modified argument signatures for `gdown.download()`, causing keyword argument failures when invoked on updated environments.

### Bug 20: Test Suite Module Path Collection Failure
* **Location:** [tests/test_dataset.py:L8](file:///home/utn/uzis83et/CV/tests/test_dataset.py#L8), [tests/test_pipeline.py:L12](file:///home/utn/uzis83et/CV/tests/test_pipeline.py#L12)
* **Root Cause:** Running `pytest` from the repository root fails during collection unless `PYTHONPATH=.` is explicitly set or a `pytest.ini` config file is present.
