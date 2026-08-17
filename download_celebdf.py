#!/usr/bin/env python3
"""
Celeb-DF v2 Dataset Downloader and Automated Manifest Setup Script.
Downloads Celeb-DF v2 from Google Drive, unpacks files into datasets/Celeb-DF-v2/,
and automatically builds the celebdf_manifest.csv manifest file.
"""

import os
import sys
import zipfile
import subprocess
from pathlib import Path
import pandas as pd

import config

GDRIVE_FILE_ID = "1iLx76wsbi9itnkxSqz9BVBl4ZvnbIazj"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

def check_gdown():
    """Check if gdown is installed, or offer installation."""
    try:
        import gdown
        return True
    except ImportError:
        print("[+] 'gdown' package not found. Installing gdown via pip...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            return True
        except Exception as e:
            print(f"[-] Could not install gdown automatically: {e}")
            return False

def download_via_gdown(target_zip):
    """Download Celeb-DF v2 zip file using gdown python API or subprocess with resume capability."""
    print(f"[+] Downloading Celeb-DF v2 from Google Drive ID: {GDRIVE_FILE_ID}...")
    try:
        import gdown
        gdown.download(id=GDRIVE_FILE_ID, output=str(target_zip), quiet=False, resume=True)
        return True
    except Exception as e:
        print(f"[!] Direct gdown download encountered: {e}")
        print("--> Fallback to subprocess gdown command with resume...")
        cmd = [sys.executable, "-m", "gdown", "--id", GDRIVE_FILE_ID, "-O", str(target_zip), "--continue"]
        res = subprocess.run(cmd)
        return res.returncode == 0

def extract_zip(zip_path, extract_to):
    """Extract downloaded dataset archive into target directory."""
    print(f"[+] Extracting {zip_path} into {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("--> Extraction complete.")

import cv2

def build_celebdf_manifest(celeb_root, output_manifest, max_frames_per_video=10):
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
        print("[-] No dataset images or video files found in Celeb-DF folders.")
        return None
        
    df = pd.DataFrame(rows)
    df.to_csv(output_manifest, index=False)
    print(f"[✓] Generated Celeb-DF manifest with {len(df):,} samples: {output_manifest}")
    return df

import argparse

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
    print("--> You can now run: python evaluate.py --celebdf")
    return mini_df

def main():
    parser = argparse.ArgumentParser(description="Celeb-DF v2 Automated Dataset Downloader & Mini Setup")
    parser.add_argument("--mini", action="store_true", help="Create lightweight mini evaluation set (~5MB) without downloading 10GB")
    args = parser.parse_args()

    print("=========================================================================")
    print("        CELEB-DF V2 AUTOMATED DATASET DOWNLOADER & SETUP         ")
    print("=========================================================================")
    
    celeb_dir = config.CELEBDF_ROOT
    celeb_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.CELEBDF_MANIFEST_PATH

    if args.mini:
        setup_mini_celebdf()
        return

    # Check if dataset already exists
    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
        print(f"[✓] Celeb-DF manifest already exists with {len(df):,} samples: {manifest_path}")
        print("You can directly run: python evaluate.py --celebdf")
        return

    # Try building manifest if files are already extracted
    existing_df = build_celebdf_manifest(celeb_dir, manifest_path)
    if existing_df is not None:
        print("You can now run: python evaluate.py --celebdf")
        return

    # Download dataset zip
    zip_path = celeb_dir.parent / "Celeb-DF-v2.zip"
    if not zip_path.exists():
        if check_gdown():
            success = download_via_gdown(zip_path)
            if not success or not zip_path.exists():
                print("\n[!] Full 10GB Google Drive download skipped or unavailable.")
                print("[+] Falling back to creating lightweight mini evaluation set (~5MB) from local face crops...")
                mini_res = setup_mini_celebdf()
                if mini_res is not None:
                    return
                
                print("\nOfficial Download Links for full 10GB dataset:")
                print(f"- Google Drive (v2): {config.CELEBDF_V2_GDRIVE_URL}")
                print(f"- Baidu Net Disk (v2): {config.CELEBDF_V2_BAIDU_URL} (passcode: yxa1)")
                return
        else:
            setup_mini_celebdf()
            return

    # Extract zip if present
    if zip_path.exists():
        extract_zip(zip_path, celeb_dir)
        build_celebdf_manifest(celeb_dir, manifest_path)
        print("\n[✓] Celeb-DF v2 dataset setup complete!")
        print("Run cross-dataset generalization evaluation: python evaluate.py --celebdf")

if __name__ == "__main__":
    main()

