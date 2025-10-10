# dataset.py
import os
import cv2
import random
import numpy as np
from glob import glob
from typing import List, Tuple
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class VideoDataset(Dataset):
    """
    On-the-fly video dataset that samples a fixed number of frames from each video.
    Returns tensor shape: [C, T, H, W]
    """
    def __init__(self,
                 root_dir: str,
                 num_frames: int = 32,
                 img_size: int = 112,
                 split: str = "train",
                 split_ratios: Tuple[float,float,float] = (0.8,0.1,0.1),
                 shuffle: bool = True,
                 augment: bool = True):
        """
        root_dir: path to 'Shop DataSet'
        split: 'train'|'val'|'test'
        split_ratios: (train, val, test)
        """
        assert split in ("train","val","test")
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.img_size = img_size
        self.split = split
        self.augment = augment and (split == "train")

        # collect videos and labels
        classes = [("non shop lifters", 0), ("shop lifters", 1)]
        all_items = []
        for folder, label in classes:
            folder_path = os.path.join(root_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
                for p in glob(os.path.join(folder_path, ext)):
                    all_items.append((p, label))
        if shuffle:
            random.shuffle(all_items)

        # split
        n = len(all_items)
        t, v, te = split_ratios
        i1 = int(n * t)
        i2 = i1 + int(n * v)
        if split == "train":
            self.items = all_items[:i1]
        elif split == "val":
            self.items = all_items[i1:i2]
        else:
            self.items = all_items[i2:]

        # transforms
        if self.augment:
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        video_path, label = self.items[idx]
        frames = self._read_video(video_path)
        frames = self._sample_uniform(frames, self.num_frames)
        # apply transform per frame -> list of tensors [C,H,W]
        processed = [self.transform(f) for f in frames]
        # stack -> [T, C, H, W] then permute to [C, T, H, W]
        frames_tensor = torch.stack(processed).permute(1, 0, 2, 3)
        return frames_tensor, torch.tensor(label, dtype=torch.long)

    def _read_video(self, path: str) -> List[np.ndarray]:
        cap = cv2.VideoCapture(path)
        frames = []
        if not cap.isOpened():
            raise IOError(f"Could not open video: {path}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        if len(frames) == 0:
            raise ValueError(f"No frames found in video: {path}")
        return frames

    def _sample_uniform(self, frames: List[np.ndarray], k: int) -> List[np.ndarray]:
        n = len(frames)
        if n >= k:
            indices = np.linspace(0, n - 1, k, dtype=np.int32)
            return [frames[i] for i in indices]
        else:
            # pad by repeating last frame until k
            out = frames.copy()
            while len(out) < k:
                out.append(frames[-1])
            return out[:k]

    def get_class_distribution(self):
        # returns counts per class in this split
        counts = {}
        for _, label in self.items:
            counts[label] = counts.get(label, 0) + 1
        return counts
