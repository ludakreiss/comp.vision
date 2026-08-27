import cv2
import math
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
from facenet_pytorch import MTCNN

import config
from dataset import build_connected_groups

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

def list_videos(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def choose_dominant_face(boxes, probabilities, landmarks=None):
    if boxes is None or probabilities is None:
        return None, None, None

    candidates = []
    for idx, (box, probability) in enumerate(zip(boxes, probabilities)):
        if probability is None or probability < config.MIN_FACE_PROBABILITY:
            continue

        x1, y1, x2, y2 = box
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        area = width * height
        lms = landmarks[idx] if landmarks is not None else None
        candidates.append((area, float(probability), box, lms))

    if not candidates:
        return None, None, None

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, probability, box, lms = candidates[0]
    return box, probability, lms


def crop_face_with_margin(rgb_frame, box, margin_percent=0.30):
    frame_height, frame_width = rgb_frame.shape[:2]
    x1, y1, x2, y2 = box

    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    margin_w = int(width * margin_percent)
    margin_h = int(height * margin_percent)

    x1 = max(0, int(math.floor(x1)) - margin_w)
    y1 = max(0, int(math.floor(y1)) - margin_h)
    x2 = min(frame_width, int(math.ceil(x2)) + margin_w)
    y2 = min(frame_height, int(math.ceil(y2)) + margin_h)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = rgb_frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return Image.fromarray(crop)


def align_face_crop(rgb_frame, landmarks, target_size=256, margin_percent=0.30):
    left_eye = landmarks[0]
    right_eye = landmarks[1]

    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    eye_center = (float((left_eye[0] + right_eye[0]) / 2.0), float((left_eye[1] + right_eye[1]) / 2.0))

    desired_dist = target_size * 0.30
    actual_dist = np.sqrt(dx**2 + dy**2)
    scale = desired_dist / max(1e-6, actual_dist)

    target_eye_x = target_size * 0.5
    target_eye_y = target_size * (0.35 + margin_percent * 0.1)

    R = cv2.getRotationMatrix2D(eye_center, angle, scale)
    R[0, 2] += (target_eye_x - eye_center[0] * scale)
    R[1, 2] += (target_eye_y - eye_center[1] * scale)

    warped = cv2.warpAffine(rgb_frame, R, (target_size, target_size), flags=cv2.INTER_CUBIC)
    return Image.fromarray(warped)


def is_blurry(pil_image, threshold=100.0):
    gray = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < threshold


def extract_faces_from_video(video_path, output_directory, mtcnn, frame_interval=30, maximum_frames=12):
    video_path = Path(video_path)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    if not config.FORCE_REEXTRACT:
        existing_images = sorted(output_directory.glob("*.jpg"))
        if len(existing_images) >= maximum_frames:
            return [
                {
                    "image_path": str(path),
                    "face_probability": None,
                }
                for path in existing_images[:maximum_frames]
            ]

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"Could not open video: {video_path}")
        return []

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        capture.release()
        return []

    if config.SAMPLING_STRATEGY == "spaced":
        target_frames = np.linspace(5, total_frames - 6, maximum_frames, dtype=int).tolist()
    else:
        target_frames = [i * frame_interval for i in range(maximum_frames) if i * frame_interval < total_frames - 5]

    extracted = []

    for t_idx in target_frames:
        window_indices = [t_idx - 2, t_idx, t_idx + 2]
        window_boxes = []
        window_probs = []
        window_landmarks = []

        for w_idx in window_indices:
            w_idx = max(0, min(w_idx, total_frames - 1))
            capture.set(cv2.CAP_PROP_POS_FRAMES, w_idx)
            success, frame_bgr = capture.read()
            if not success:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)

            boxes, probabilities, landmarks = mtcnn.detect(frame_pil, landmarks=True)
            box, face_probability, lms = choose_dominant_face(boxes, probabilities, landmarks)
            if box is not None:
                window_boxes.append(box)
                window_probs.append(face_probability)
                if lms is not None:
                    window_landmarks.append(lms)

        if len(window_boxes) >= 2:
            avg_box = np.mean(window_boxes, axis=0)
            avg_prob = np.mean(window_probs)

            capture.set(cv2.CAP_PROP_POS_FRAMES, t_idx)
            success, frame_bgr = capture.read()
            if success:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                if config.ALIGN_FACES and len(window_landmarks) >= 2:
                    avg_landmarks = np.mean(window_landmarks, axis=0)
                    target_sz = getattr(config, "IMAGE_SIZE", 256)
                    face = align_face_crop(frame_rgb, avg_landmarks, target_size=target_sz, margin_percent=config.FACE_MARGIN_PERCENT)
                else:
                    face = crop_face_with_margin(frame_rgb, avg_box, config.FACE_MARGIN_PERCENT)

                if face is not None:
                    w, h = face.size
                    if min(w, h) < config.MIN_FACE_SIZE:
                        continue
                    if is_blurry(face, config.BLUR_THRESHOLD):
                        continue

                    output_path = output_directory / f"frame_{t_idx:06d}.jpg"
                    face.save(
                        output_path,
                        format="JPEG",
                        quality=95,
                        subsampling=0,
                    )
                    extracted.append({
                        "image_path": str(output_path),
                        "face_probability": float(avg_prob),
                    })

    capture.release()
    return extracted


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract faces from videos")
    parser.add_argument("--sanity", action="store_true", help="Run a quick sanity check (only process 2 videos per category)")
    args = parser.parse_args()

    if args.sanity:
        config.MAX_EXTRACTION_VIDEOS_PER_CATEGORY = 3
        config.MAX_VIDEOS_PER_CATEGORY = 3

    # Set seed for reproducibility
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    # Initialize paths
    EXPECTED_FOLDERS = {
        "original": config.DATASET_ROOT / "original_sequences" / "youtube" / "c23" / "videos",
        "Deepfakes": config.DATASET_ROOT / "manipulated_sequences" / "Deepfakes" / "c23" / "videos",
        "Face2Face": config.DATASET_ROOT / "manipulated_sequences" / "Face2Face" / "c23" / "videos",
        "FaceSwap": config.DATASET_ROOT / "manipulated_sequences" / "FaceSwap" / "c23" / "videos",
        "NeuralTextures": config.DATASET_ROOT / "manipulated_sequences" / "NeuralTextures" / "c23" / "videos",
    }

    print("Checking video directories...")
    video_rows = []
    for manipulation, folder in EXPECTED_FOLDERS.items():
        videos = list_videos(folder)
        if config.MAX_VIDEOS_PER_CATEGORY is not None:
            videos = videos[:config.MAX_VIDEOS_PER_CATEGORY]

        label = 0 if manipulation == "original" else 1
        for video_path in videos:
            video_id = video_path.stem
            video_rows.append({
                "video_path": str(video_path),
                "label": label,
                "manipulation": manipulation,
                "video_id": video_id,
            })
            
    group_map = build_connected_groups([r["video_id"] for r in video_rows])
    for row in video_rows:
        row["group_id"] = group_map.get(row["video_id"], row["video_id"])

    ffpp_videos = pd.DataFrame(video_rows)
    print(f"Total videos found: {len(ffpp_videos)}")
    if len(ffpp_videos) == 0:
        print("No videos found! Please run the download script first.")
        return

    # Balance the dataset (keep equal original vs manipulated if possible, matching notebook logic)
    real_videos = ffpp_videos[ffpp_videos["label"] == 0].copy()
    num_real_videos = len(real_videos)

    fake_parts = []
    for manipulation in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
        manip_df = ffpp_videos[ffpp_videos["manipulation"] == manipulation].copy()
        sample_size = min(max(1, num_real_videos // 4), len(manip_df))
        if sample_size > 0:
            manip_df = manip_df.sample(n=sample_size, random_state=config.SEED)
            fake_parts.append(manip_df)

    selected_videos = pd.concat([real_videos] + fake_parts, ignore_index=True)
    selected_videos = selected_videos.sample(frac=1, random_state=config.SEED).reset_index(drop=True)

    if config.MAX_EXTRACTION_VIDEOS_PER_CATEGORY is not None:
        selected_videos = selected_videos.groupby("manipulation", group_keys=False).apply(
            lambda g: g.sample(n=min(len(g), config.MAX_EXTRACTION_VIDEOS_PER_CATEGORY), random_state=config.SEED)
        ).reset_index(drop=True)

    print(f"Videos selected for face extraction: {len(selected_videos)}")
    print(selected_videos["manipulation"].value_counts())

    manifest_rows = []
    existing_records = {}

    # Load existing manifest if it exists to enable incremental updates
    if config.MANIFEST_PATH.exists() and not config.FORCE_REEXTRACT:
        try:
            old_manifest = pd.read_csv(config.MANIFEST_PATH)
            print(f"Found existing manifest at {config.MANIFEST_PATH}. Loading records...")
            for r in old_manifest.itertuples(index=False):
                vid = str(r.video_id)
                if vid not in existing_records:
                    existing_records[vid] = []
                existing_records[vid].append({
                    "image_path": r.image_path,
                    "label": int(r.label),
                    "manipulation": r.manipulation,
                    "video_id": vid,
                    "group_id": str(r.group_id),
                    "face_probability": r.face_probability,
                })
            print(f"Loaded existing face records for {len(existing_records)} videos.")
        except Exception as error:
            print(f"Could not load existing manifest: {error}. Rebuilding from scratch.")

    # Initialize MTCNN on GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing MTCNN on device: {device}")
    mtcnn = MTCNN(
        keep_all=True,
        min_face_size=40,
        post_process=False,
        device=device
    )

    print("Extracting faces...")
    for row in tqdm(selected_videos.itertuples(index=False), total=len(selected_videos)):
        vid = str(row.video_id)

        # Re-use existing face records if this video has already been processed
        if not config.FORCE_REEXTRACT and vid in existing_records:
            manifest_rows.extend(existing_records[vid])
            continue

        output_directory = config.FFPP_FACE_ROOT / row.manipulation / row.video_id
        extracted_faces = extract_faces_from_video(
            video_path=row.video_path,
            output_directory=output_directory,
            mtcnn=mtcnn,
            frame_interval=config.FRAME_INTERVAL,
            maximum_frames=config.MAX_FRAMES_PER_VIDEO,
        )

        for face in extracted_faces:
            manifest_rows.append({
                "image_path": face["image_path"],
                "label": int(row.label),
                "manipulation": row.manipulation,
                "video_id": vid,
                "group_id": str(row.group_id),
                "face_probability": face["face_probability"],
            })

    ffpp_manifest = pd.DataFrame(manifest_rows)
    # Save the split mapped manifest
    from dataset import assign_group_splits
    ffpp_manifest = assign_group_splits(ffpp_manifest, seed=config.SEED)
    
    ffpp_manifest.to_csv(config.MANIFEST_PATH, index=False)
    print(f"\nFace extraction complete. Extracted {len(ffpp_manifest)} faces.")
    print(f"Manifest saved to: {config.MANIFEST_PATH}")
    print(ffpp_manifest["label"].value_counts().rename({0: "Real", 1: "Fake"}))

if __name__ == "__main__":
    main()
