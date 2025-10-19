import torch
import torch.nn as nn
from torchvision.models.video import r3d_18
import numpy as np
import os
from tqdm import tqdm
from preprocess_videos import extract_frames_uniform  # ✅ import the correct function

# ---------------- Configuration ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_video_classifier.pth"
TEST_PATH = "Shop DataSet/shop lifters/shop_lifter_21.mp4"  # Can be a folder OR a video file
NUM_CLASSES = 2  # shop lifters / non shop lifters
NUM_FRAMES = 32  # same as your preprocessing

# ---------------- Dataset ----------------
class VideoTestDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for label in os.listdir(root_dir):
            folder = os.path.join(root_dir, label)
            if not os.path.isdir(folder):
                continue
            for f in os.listdir(folder):
                if f.endswith(".npy"):
                    self.samples.append((os.path.join(folder, f), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = np.load(path)  # (T, H, W, C)
        frames = torch.tensor(frames).permute(3, 0, 1, 2).float() / 255.0
        return frames, label


# ---------------- Evaluation ----------------
def evaluate(model, loader):
    model.eval()
    total = 0
    with torch.no_grad():
        for x, _ in tqdm(loader, desc="Testing", leave=False):
            x = x.to(DEVICE)
            outputs = model(x)
            preds = outputs.argmax(1)
            total += len(preds)
    return total


def predict_single_video(model, video_path):
    frames = extract_frames_uniform(video_path, num_frames=NUM_FRAMES)  # ✅ reuse existing function
    x = torch.tensor(frames).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
    x = x.to(DEVICE)
    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        pred = outputs.argmax(1).item()
        confidence = probs[0, pred].item()
    return pred, confidence


# ---------------- Main ----------------
if __name__ == "__main__":
    print("🧠 Loading model...")
    model = r3d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    if os.path.isdir(TEST_PATH):
        print("📂 Testing on dataset...")
        test_ds = VideoTestDataset(TEST_PATH)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=4, shuffle=False)
        total = evaluate(model, test_loader)
        print(f"✅ Test completed on {len(test_ds)} samples.")
    elif TEST_PATH.lower().endswith((".mp4", ".avi", ".mov")):
        print(f"🎥 Testing single video: {TEST_PATH}")
        pred, conf = predict_single_video(model, TEST_PATH)
        label_names = ["non shop lifters", "shop lifters"]
        print(f"🎯 Prediction: {label_names[pred]} ({conf*100:.1f}% confidence)")
    else:
        print("❌ Invalid TEST_PATH. Please provide a folder or video file.")
