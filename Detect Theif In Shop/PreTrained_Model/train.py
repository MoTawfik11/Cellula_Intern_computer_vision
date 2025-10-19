import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r3d_18, R3D_18_Weights
import numpy as np
import os
from tqdm import tqdm
import random
import torchvision.transforms as T

# ---------------- Configuration ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
EPOCHS = 30
LR = 1e-4
DATASET_DIR = "dataset_processed/train"
VAL_DIR = "dataset_processed/val"
SEED = 42
PATIENCE = 5  # Early stopping

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ---------------- Dataset ----------------
class VideoDataset(Dataset):
    def __init__(self, root_dir, augment=False):
        self.samples = []
        self.labels = sorted(os.listdir(root_dir))
        self.augment = augment

        for label_idx, label in enumerate(self.labels):
            folder = os.path.join(root_dir, label)
            if not os.path.isdir(folder):
                continue
            for f in os.listdir(folder):
                if f.endswith(".npy"):
                    self.samples.append((os.path.join(folder, f), label_idx))

        # Augmentations (spatial only)
        self.transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.RandomResizedCrop((112, 112), scale=(0.8, 1.0))
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = np.load(path)  # (T, H, W, C)
        frames = torch.tensor(frames).permute(3, 0, 1, 2).float() / 255.0  # [C, T, H, W]

        if self.augment:
            # Apply frame-wise augmentation
            augmented = []
            for i in range(frames.shape[1]):
                f = self.transform(frames[:, i, :, :])
                augmented.append(f.unsqueeze(1))
            frames = torch.cat(augmented, dim=1)

        return frames, torch.tensor(label, dtype=torch.long)


# ---------------- Training and Evaluation ----------------
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x, y in tqdm(loader, desc="Training", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return total_loss / len(loader), correct / total


# ---------------- Main ----------------
if __name__ == "__main__":
    print("📂 Loading datasets...")
    train_ds = VideoDataset(DATASET_DIR, augment=True)
    val_ds = VideoDataset(VAL_DIR)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    print("🧠 Loading model...")
    weights = R3D_18_Weights.KINETICS400_V1
    model = r3d_18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(train_ds.labels))
    model.to(DEVICE)

    # Freeze backbone initially
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_acc = 0
    best_val_loss = float("inf")
    patience_counter = 0

    print("🚀 Starting training...")
    for epoch in range(EPOCHS):
        # Unfreeze all layers after 5 epochs for fine-tuning
        if epoch == 5:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=LR * 0.5)
            print("🔓 Unfrozen all layers for fine-tuning")

        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1}/{EPOCHS}] - "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.3f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_video_classifier.pth")
            print(f"💾 Saved best model (Val Acc: {best_val_acc:.3f})")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("⏹ Early stopping triggered.")
                break

    print("✅ Training completed. Best model saved as best_video_classifier.pth")
