import os
import cv2
import numpy as np
import torch
import random
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# -----------------------------
# Configuration
# -----------------------------
DATASET_DIR = "Shop DataSet"        # Folder containing all videos
OUTPUT_DIR = "dataset_processed"
FRAME_COUNT = 32                    # Number of frames per video
SEED = 42
SPLIT = [0.8, 0.15, 0.05]           # train/val/test split ratio

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# -----------------------------
# Helper Functions
# -----------------------------
def get_video_embedding(video_path, frame_count=8):
    """
    Compute a robust video embedding using color histograms from sampled frames.
    This helps differentiate videos better than grayscale averages.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        return np.zeros((1, 512))

    step = max(1, total // frame_count)
    histograms = []

    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (64, 64))
        hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8],
                            [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        histograms.append(hist)

        if len(histograms) >= frame_count:
            break

    cap.release()

    if len(histograms) == 0:
        return np.zeros((1, 512))

    return np.mean(histograms, axis=0, keepdims=True)


def remove_similar_videos(video_paths, threshold=0.9999):
    """
    Remove visually similar videos using cosine similarity.
    If threshold=None, it automatically adapts to preserve 85–90% of videos.
    """
    print(f"Calculating embeddings for {len(video_paths)} videos...")
    embeddings = [get_video_embedding(v) for v in tqdm(video_paths, desc="Embedding videos")]
    embeddings = np.vstack(embeddings)
    sim_matrix = cosine_similarity(embeddings)

    upper_tri = sim_matrix[np.triu_indices(len(video_paths), k=1)]
    mean_sim = np.mean(upper_tri)
    std_sim = np.std(upper_tri)

    if threshold is None:
        threshold = min(0.99, mean_sim + 2 * std_sim)
        print(f"Auto-selected similarity threshold: {threshold:.3f}")

    keep = []
    removed = set()
    for i in range(len(video_paths)):
        if i in removed:
            continue
        keep.append(video_paths[i])
        for j in range(i + 1, len(video_paths)):
            if sim_matrix[i, j] > threshold:
                removed.add(j)

    print(f"Removed {len(removed)} / {len(video_paths)} videos ({len(keep)} kept).")
    return keep


def extract_frames_uniform(video_path, num_frames=FRAME_COUNT):
    """Extract a fixed number of frames uniformly from the video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        return np.zeros((num_frames, 112, 112, 3), dtype=np.uint8)

    frame_indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (112, 112))
        frames.append(frame)

    cap.release()

    # Pad if fewer frames than expected
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return np.array(frames, dtype=np.uint8)


def save_processed_dataset(video_paths):
    """Split videos into train/val/test and save as .npy frame arrays."""
    random.shuffle(video_paths)
    n = len(video_paths)
    n_train = int(n * SPLIT[0])
    n_val = int(n * SPLIT[1])

    subsets = {
        "train": video_paths[:n_train],
        "val": video_paths[n_train:n_train + n_val],
        "test": video_paths[n_train + n_val:]
    }

    for subset, videos in subsets.items():
        subset_dir = os.path.join(OUTPUT_DIR, subset)
        os.makedirs(subset_dir, exist_ok=True)
        print(f"\nProcessing {subset} set ({len(videos)} videos)...")

        for vid_path in tqdm(videos, desc=f"Processing {subset}"):
            label = os.path.basename(os.path.dirname(vid_path))
            label_dir = os.path.join(subset_dir, label)
            os.makedirs(label_dir, exist_ok=True)

            frames = extract_frames_uniform(vid_path)
            np.save(os.path.join(label_dir, os.path.basename(vid_path) + ".npy"), frames)


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    all_videos = []
    for root, _, files in os.walk(DATASET_DIR):
        for f in files:
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                all_videos.append(os.path.join(root, f))

    print(f"Total videos found: {len(all_videos)}")
    filtered_videos = remove_similar_videos(all_videos, threshold=0.9999)
    save_processed_dataset(filtered_videos)
    print("\n✅ Preprocessing complete! Processed data saved to:", OUTPUT_DIR)
