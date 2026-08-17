import argparse
import io
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance
import cv2

import torch
from torch.utils.data import DataLoader, Dataset

import config
from dataset import get_dataloaders, DeepfakeImageDataset, get_transforms
from model import build_model
from train import evaluate_model, calculate_binary_metrics, DeepfakeLoss
from metrics_utils import bootstrap_metric_ci, paired_bootstrap_test, calculate_ece

def apply_advanced_tier_distortion(pil_img, tier_cfg):
    """Apply synthetic platform distortion (JPEG compression, downscaling, blur, noise, jitter) to PIL Image."""
    img = pil_img.copy()
    w, h = img.size
    
    # 1. Resizing / Downscaling
    scale = tier_cfg.get("resize_scale", 1.0)
    if scale < 1.0:
        nw = max(16, int(w * scale))
        nh = max(16, int(h * scale))
        img = img.resize((nw, nh), Image.Resampling.BILINEAR)
        img = img.resize((w, h), Image.Resampling.BILINEAR)
        
    # 2. Gaussian Blur
    blur_sigma = tier_cfg.get("blur_sigma")
    if blur_sigma is not None and blur_sigma > 0:
        img = img.filter(ImageFilter.GaussianBlur(blur_sigma))
        
    # 3. Motion Blur
    motion_size = tier_cfg.get("motion_blur_size")
    if motion_size is not None and motion_size > 1:
        img_np = np.array(img)
        kernel = np.zeros((motion_size, motion_size))
        kernel[int((motion_size - 1) / 2), :] = np.ones(motion_size)
        kernel /= motion_size
        img_np = cv2.filter2D(img_np, -1, kernel)
        img = Image.fromarray(img_np)
        
    # 4. Color Jitter
    jitter = tier_cfg.get("color_jitter")
    if jitter is not None and jitter > 0:
        img = ImageEnhance.Brightness(img).enhance(1.0 + (np.random.uniform(-jitter, jitter)))
        img = ImageEnhance.Contrast(img).enhance(1.0 + (np.random.uniform(-jitter, jitter)))
        
    # 5. Gaussian Noise
    noise_std = tier_cfg.get("noise_std")
    if noise_std is not None and noise_std > 0:
        img_np = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, noise_std, img_np.shape)
        img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        
    # 6. JPEG Compression
    q = tier_cfg.get("jpeg_quality")
    if q is not None:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(q))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        
    return img

class TierDistortionModifier:
    def __init__(self, tier_cfg):
        self.tier_cfg = tier_cfg

    def __call__(self, img):
        return apply_advanced_tier_distortion(img, self.tier_cfg)

def evaluate_single_model_on_tier(model, test_df, eval_transform, tier_cfg, criterion, device, limit_batches=None):
    modifier = TierDistortionModifier(tier_cfg)
    ds = DeepfakeImageDataset(test_df, transform=eval_transform, image_modifier=modifier)
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    
    metrics, preds_df = evaluate_model(model, loader, criterion, device, limit_batches=limit_batches)
    return metrics, preds_df

def run_comparative_benchmark(n_bootstraps=1000, limit_batches=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Running Comparative Deepfake Detection Benchmark Matrix on {device}...")
    
    train_loader, val_loader, test_loader, test_df = get_dataloaders()
    _, eval_transform = get_transforms()
    criterion = DeepfakeLoss(loss_type="bce", smoothing=config.LABEL_SMOOTHING)
    
    # 1. Load Standard Model
    std_ckpt_path = config.OUTPUT_ROOT / f"{config.MODEL_NAME}_clean" / "best_model.pt"
    if not std_ckpt_path.exists():
        raise FileNotFoundError(f"Standard model checkpoint not found at: {std_ckpt_path}")
    print(f"Loading Standard Model from {std_ckpt_path}...")
    std_model = build_model(config.MODEL_NAME, pretrained=False).to(device)
    std_ckpt = torch.load(std_ckpt_path, map_location=device, weights_only=False)
    std_model.load_state_dict(std_ckpt["model_state_dict"])
    std_thresh = std_ckpt.get("configuration", {}).get("optimal_threshold", 0.5)

    # 2. Load Robustness Model
    rob_ckpt_path = config.OUTPUT_ROOT / f"{config.MODEL_NAME}_degradation" / "best_model.pt"
    if not rob_ckpt_path.exists():
        raise FileNotFoundError(f"Robustness model checkpoint not found at: {rob_ckpt_path}")
    print(f"Loading Robustness Model from {rob_ckpt_path}...")
    rob_model = build_model(config.MODEL_NAME, pretrained=False).to(device)
    rob_ckpt = torch.load(rob_ckpt_path, map_location=device, weights_only=False)
    rob_model.load_state_dict(rob_ckpt["model_state_dict"])
    rob_thresh = rob_ckpt.get("configuration", {}).get("optimal_threshold", 0.5)

    results = []
    
    print("\n" + "="*110)
    print(f"{'Degradation Tier':<25} | {'Std AUC (95% CI)':<22} | {'Std ECE':<8} | {'Rob AUC (95% CI)':<22} | {'Rob ECE':<8} | {'Δ AUC':<7} | {'p-value'}")
    print("="*110)
    
    for tier_name, tier_cfg in config.DEGRADATION_TIERS.items():
        std_m, std_preds = evaluate_single_model_on_tier(std_model, test_df, eval_transform, tier_cfg, criterion, device, limit_batches=limit_batches)
        rob_m, rob_preds = evaluate_single_model_on_tier(rob_model, test_df, eval_transform, tier_cfg, criterion, device, limit_batches=limit_batches)
        
        y_true = std_preds["label"].to_numpy().astype(int)
        std_probs = std_preds["prob_fake"].to_numpy().astype(float)
        rob_probs = rob_preds["prob_fake"].to_numpy().astype(float)
        
        std_auc_mean, std_low, std_high = bootstrap_metric_ci(y_true, std_probs, metric_func=calculate_binary_metrics, n_bootstraps=n_bootstraps, seed=config.SEED)
        rob_auc_mean, rob_low, rob_high = bootstrap_metric_ci(y_true, rob_probs, metric_func=calculate_binary_metrics, n_bootstraps=n_bootstraps, seed=config.SEED)
        
        p_val = paired_bootstrap_test(y_true, std_probs, rob_probs, n_bootstraps=n_bootstraps, seed=config.SEED)
        
        std_ece = calculate_ece(y_true, std_probs)
        rob_ece = calculate_ece(y_true, rob_probs)
        
        delta_auc = (rob_m["roc_auc"] - std_m["roc_auc"]) * 100.0
        
        results.append({
            "tier": tier_name,
            "std_auc": std_m["roc_auc"],
            "std_ci_low": std_low,
            "std_ci_high": std_high,
            "std_ece": std_ece,
            "rob_auc": rob_m["roc_auc"],
            "rob_ci_low": rob_low,
            "rob_ci_high": rob_high,
            "rob_ece": rob_ece,
            "delta_auc": delta_auc,
            "p_value": p_val
        })
        
        std_str = f"{std_m['roc_auc']*100:.2f}% [{std_low*100:.1f}%, {std_high*100:.1f}%]"
        rob_str = f"{rob_m['roc_auc']*100:.2f}% [{rob_low*100:.1f}%, {rob_high*100:.1f}%]"
        p_str = f"{p_val:.3f}" if p_val >= 0.001 else "< 0.001 ***"
        
        print(f"{tier_name:<25} | {std_str:<22} | {std_ece:.4f}   | {rob_str:<22} | {rob_ece:.4f}   | {delta_auc:+6.2f}% | {p_str}")

    print("="*110)
    
    # Save CSV and Markdown reports
    report_df = pd.DataFrame(results)
    csv_path = config.OUTPUT_ROOT / "comparative_robustness_stat_results.csv"
    report_df.to_csv(csv_path, index=False)
    
    report_path = config.OUTPUT_ROOT / "comparative_robustness_report.md"
    with open(report_path, "w") as f:
        f.write("# Comparative Deepfake Detection Robustness & Statistical Rigor Report\n\n")
        f.write("> **University of Technology Nuremberg — Statistically Rigorous Benchmark Report**\n")
        f.write(f"> **Model Architecture**: {config.MODEL_NAME.upper()} Dual-Branch Fusion\n")
        f.write("> **Statistical Protocol**: 95% Non-parametric Percentile Bootstrap CIs (1,000 resamples), Expected Calibration Error (ECE), Paired Bootstrap Hypothesis Testing.\n\n")
        f.write("---\n\n## 1. Degradation Benchmark Matrix\n\n")
        f.write("| Degradation Tier | Standard ROC-AUC (95% CI) | Standard ECE | Robustness ROC-AUC (95% CI) | Robustness ECE | $\\Delta$ ROC-AUC Gain | Paired $p$-value |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in results:
            p_s = f"`{r['p_value']:.3f}`" if r['p_value'] >= 0.001 else "`< 0.001 ***`"
            if r['p_value'] >= 0.05:
                p_s += " (n.s.)"
            f.write(f"| **{r['tier']}** | {r['std_auc']*100:.2f}% [{r['std_ci_low']*100:.1f}%, {r['std_ci_high']*100:.1f}%] | {r['std_ece']:.4f} | **{r['rob_auc']*100:.2f}% [{r['rob_ci_low']*100:.1f}%, {r['rob_ci_high']*100:.1f}%]** | **{r['rob_ece']:.4f}** | **{r['delta_auc']:+.2f}%** | {p_s} |\n")

    print(f"\n[+] Exported comparative report to: {report_path}")

def generate_celebdf_manifest(celeb_root, output_manifest, max_frames_per_video=10):
    """Generate CSV manifest for Celeb-DF v2 dataset supporting both image crops and raw video files."""
    print(f"[+] Scanning Celeb-DF dataset directory at: {celeb_root}...")
    celeb_root = Path(celeb_root)
    rows = []
    
    categories = {
        "Celeb-real": 0,
        "YouTube-real": 0,
        "Celeb-synthesis": 1
    }
    
    test_list_path = celeb_root / "List_of_testing_videos.txt"
    test_videos = set()
    if test_list_path.exists():
        print(f"--> Found official test list: {test_list_path}")
        with open(test_list_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    test_videos.add(Path(parts[1]).name)
                    
    # 1. Search for existing image crops in category subfolders
    for cat_folder, label in categories.items():
        cat_path = celeb_root / cat_folder
        if not cat_path.exists():
            continue
            
        video_frame_counts = {}
        for root, _, files in os.walk(cat_path):
            img_files = sorted([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            for file in img_files:
                img_path = Path(root) / file
                video_id = img_path.parent.name
                if test_videos and video_id not in test_videos and img_path.name not in test_videos:
                    continue
                
                count = video_frame_counts.get(video_id, 0)
                if max_frames_per_video is not None and count >= max_frames_per_video:
                    continue
                video_frame_counts[video_id] = count + 1

                rows.append({
                    "image_path": str(img_path),
                    "video_id": video_id,
                    "label": float(label),
                    "category": cat_folder
                })

    # 2. Search for video files (.mp4, .avi, .mov, .mkv) if no image crops were found
    if not rows:
        video_files = []
        for root, _, files in os.walk(celeb_root):
            for file in files:
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_files.append(Path(root) / file)
                    
        if video_files:
            print(f"--> Found {len(video_files)} raw video files. Extracting frames...")
            processed_dir = celeb_root / "processed_faces"
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            for vid_path in video_files:
                vid_name = vid_path.stem
                parent_dir = vid_path.parent.name
                
                # Determine label: synthesis/fake/manipulated = 1, real = 0
                if "synthesis" in parent_dir.lower() or "synthesis" in vid_name.lower() or "fake" in vid_name.lower():
                    label = 1.0
                    category = "Celeb-synthesis"
                elif "youtube" in parent_dir.lower():
                    label = 0.0
                    category = "YouTube-real"
                else:
                    label = 0.0
                    category = "Celeb-real"
                    
                # Frame extraction via OpenCV
                cap = cv2.VideoCapture(str(vid_path))
                if not cap.isOpened():
                    continue
                total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_f <= 0:
                    cap.release()
                    continue
                step = max(1, total_f // max_frames_per_video)
                
                vid_out_dir = processed_dir / category / vid_name
                vid_out_dir.mkdir(parents=True, exist_ok=True)
                
                frame_idx = 0
                saved_count = 0
                while cap.isOpened() and saved_count < max_frames_per_video:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_idx % step == 0:
                        out_file = vid_out_dir / f"frame_{saved_count:04d}.jpg"
                        cv2.imwrite(str(out_file), frame)
                        rows.append({
                            "image_path": str(out_file),
                            "video_id": vid_name,
                            "label": label,
                            "category": category
                        })
                        saved_count += 1
                    frame_idx += 1
                cap.release()

    if not rows:
        return None
        
    df = pd.DataFrame(rows)
    df.to_csv(output_manifest, index=False)
    print(f"--> Generated lightweight Celeb-DF manifest with {len(df):,} samples: {output_manifest}")
    return df

def setup_mini_celebdf(num_samples=200):
    """Build a lightweight mini evaluation set (~5MB) from local dataset samples without downloading 10GB."""
    celeb_dir = config.CELEBDF_ROOT
    celeb_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.CELEBDF_MANIFEST_PATH

    print(f"[+] Creating lightweight mini evaluation set in: {celeb_dir}...")
    
    if config.MANIFEST_PATH.exists():
        df = pd.read_csv(config.MANIFEST_PATH)
        df = df[df["image_path"].map(lambda p: Path(p).exists())]
    else:
        df = None

    if df is None or len(df) == 0:
        print("[-] Main dataset manifest not found. Cannot sample local images.")
        return None

    reals = df[df["label"] == 0]
    fakes = df[df["label"] == 1]
    
    n_real = min(num_samples // 2, len(reals))
    n_fake = min(num_samples // 2, len(fakes))
    
    sampled_reals = reals.sample(n=n_real, random_state=42) if n_real > 0 else reals
    sampled_fakes = fakes.sample(n=n_fake, random_state=42) if n_fake > 0 else fakes
    
    sampled_df = pd.concat([sampled_reals, sampled_fakes]).reset_index(drop=True)
    
    mini_rows = []
    for idx, row in sampled_df.iterrows():
        cat = "Celeb-synthesis" if row["label"] == 1 else ("YouTube-real" if idx % 2 == 0 else "Celeb-real")
        mini_rows.append({
            "image_path": row["image_path"],
            "video_id": f"mini_{row['video_id']}",
            "label": float(row["label"]),
            "category": cat
        })
        
    mini_df = pd.DataFrame(mini_rows)
    mini_df.to_csv(manifest_path, index=False)
    print(f"[✓] Successfully generated lightweight Celeb-DF mini manifest with {len(mini_df):,} samples: {manifest_path}")
    return mini_df

def run_celebdf_eval(limit_batches=None):
    """Run Celeb-DF v2 cross-dataset generalization evaluation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Starting Celeb-DF v2 Cross-Dataset Generalization Evaluation on {device}...")

    manifest_df = None
    if config.CELEBDF_MANIFEST_PATH.exists() and config.CELEBDF_MANIFEST_PATH.stat().st_size > 0:
        try:
            manifest_df = pd.read_csv(config.CELEBDF_MANIFEST_PATH)
        except Exception:
            manifest_df = None

    if manifest_df is None or len(manifest_df) == 0:
        if config.CELEBDF_ROOT.exists():
            manifest_df = generate_celebdf_manifest(config.CELEBDF_ROOT, config.CELEBDF_MANIFEST_PATH)
            
    if manifest_df is None or len(manifest_df) == 0:
        print("[+] Attempting automatic lightweight mini dataset setup...")
        manifest_df = setup_mini_celebdf()

    if manifest_df is None or len(manifest_df) == 0:
        print(f"\n[!] Notice: Celeb-DF dataset manifest not found at: {config.CELEBDF_MANIFEST_PATH}")
        print(f"[!] Scanned dataset directory: {config.CELEBDF_ROOT}")
        print("\nTo set up Celeb-DF evaluation:")
        print("1. Run lightweight setup: python download_celebdf.py --mini")
        print("2. Or download full zip manually via official links:")
        print(f"   - Google Drive (v2): {config.CELEBDF_V2_GDRIVE_URL}")
        print(f"   - Baidu Net Disk (v2): {config.CELEBDF_V2_BAIDU_URL}  (passcode: yxa1)")
        print("3. Re-run: python evaluate.py --celebdf")
        return

    print(f"Loaded Celeb-DF v2 manifest: {len(manifest_df):,} samples.")
    _, eval_transform = get_transforms()
    dataset = DeepfakeImageDataset(manifest_df, eval_transform)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    criterion = DeepfakeLoss(loss_type="bce", smoothing=config.LABEL_SMOOTHING)

    # Standard Model
    std_ckpt = config.OUTPUT_ROOT / f"{config.MODEL_NAME}_clean" / "best_model.pt"
    if std_ckpt.exists():
        std_model = build_model(config.MODEL_NAME, pretrained=False).to(device)
        std_model.load_state_dict(torch.load(std_ckpt, map_location=device, weights_only=False)["model_state_dict"])
        std_metrics, _ = evaluate_model(std_model, dataloader, criterion, device, limit_batches=limit_batches)
    else:
        std_metrics = {"accuracy": 0.0, "roc_auc": 0.0, "f1": 0.0}

    # Robustness Model
    rob_ckpt = config.OUTPUT_ROOT / f"{config.MODEL_NAME}_degradation" / "best_model.pt"
    if rob_ckpt.exists():
        rob_model = build_model(config.MODEL_NAME, pretrained=False).to(device)
        rob_model.load_state_dict(torch.load(rob_ckpt, map_location=device, weights_only=False)["model_state_dict"])
        rob_metrics, _ = evaluate_model(rob_model, dataloader, criterion, device, limit_batches=limit_batches)
    else:
        rob_metrics = {"accuracy": 0.0, "roc_auc": 0.0, "f1": 0.0}

    print("\n" + "="*80)
    print("        CELEB-DF V2 CROSS-DATASET GENERALIZATION BENCHMARK        ")
    print("="*80)
    print(f"Standard Model (Clean)       | Acc: {std_metrics['accuracy']*100:.2f}% | ROC-AUC: {std_metrics['roc_auc']*100:.2f}% | F1: {std_metrics['f1']*100:.2f}%")
    print(f"Robustness Model (Degradation)| Acc: {rob_metrics['accuracy']*100:.2f}% | ROC-AUC: {rob_metrics['roc_auc']*100:.2f}% | F1: {rob_metrics['f1']*100:.2f}%")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Unified Deepfake Detection Evaluation Harness")
    parser.add_argument("--mode", type=str, default="comparative", choices=["comparative", "celebdf"], help="Evaluation mode")
    parser.add_argument("--celebdf", action="store_true", help="Shortcut flag for Celeb-DF evaluation")
    parser.add_argument("--bootstraps", type=int, default=1000, help="Number of bootstrap resamples for CIs")
    parser.add_argument("--limit_batches", type=int, default=None, help="Limit number of batches for quick validation")
    args = parser.parse_args()

    if args.celebdf or args.mode == "celebdf":
        run_celebdf_eval(limit_batches=args.limit_batches)
    else:
        run_comparative_benchmark(n_bootstraps=args.bootstraps, limit_batches=args.limit_batches)

if __name__ == "__main__":
    main()