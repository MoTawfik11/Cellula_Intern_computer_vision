# train.py
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import VideoDataset
from models import CNN_RNN, Simple3DCNN
from utils import seed_everything, compute_class_weights, metrics_from_preds

def train_one_model(cfg):
    device = cfg["device"]
    seed_everything(cfg["seed"])
    os.makedirs(cfg["save_dir"], exist_ok=True)

    # datasets
    train_ds = VideoDataset(cfg["data_root"], num_frames=cfg["num_frames"], img_size=cfg["img_size"], split="train", augment=True)
    val_ds   = VideoDataset(cfg["data_root"], num_frames=cfg["num_frames"], img_size=cfg["img_size"], split="val", augment=False)
    test_ds  = VideoDataset(cfg["data_root"], num_frames=cfg["num_frames"], img_size=cfg["img_size"], split="test", augment=False)

    # dataloaders
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], pin_memory=(device.startswith("cuda")))
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], pin_memory=(device.startswith("cuda")))

    # model selection
    if cfg["model_type"] == "cnn_rnn":
        model = CNN_RNN(backbone_pretrained=True, rnn_hidden=cfg["rnn_hidden"], num_classes=cfg["num_classes"])
        model_name = "cnn_rnn"
    elif cfg["model_type"] == "3dcnn":
        model = Simple3DCNN(num_classes=cfg["num_classes"])
        model_name = "3dcnn"
    else:
        raise ValueError("model_type must be 'cnn_rnn' or '3dcnn'")

    model = model.to(device)

    # weighted loss
    class_weights = compute_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    writer = SummaryWriter(log_dir=os.path.join(cfg["save_dir"], "runs", model_name))

    best_val_f1 = 0.0
    best_ckpt = None

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_losses = []
        preds_all = []
        labels_all = []

        pbar = tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch}/{cfg['epochs']}", ncols=120)
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            preds = logits.argmax(dim=1).cpu().numpy().tolist()
            preds_all.extend(preds)
            labels_all.extend(y.cpu().numpy().tolist())

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_metrics = metrics_from_preds(labels_all, preds_all)
        avg_loss = sum(epoch_losses) / len(epoch_losses)

        # validation
        model.eval()
        val_preds, val_trues = [], []
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_losses.append(loss.item())
                val_preds.extend(logits.argmax(1).cpu().numpy().tolist())
                val_trues.extend(y.cpu().numpy().tolist())

        val_metrics = metrics_from_preds(val_trues, val_preds)
        val_loss = sum(val_losses) / max(1, len(val_losses))

        # logging
        print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f} | Train F1: {train_metrics['f1']:.4f} | Val F1: {val_metrics['f1']:.4f}")
        writer.add_scalar(f"{model_name}/train/loss", avg_loss, epoch)
        writer.add_scalar(f"{model_name}/train/f1", train_metrics["f1"], epoch)
        writer.add_scalar(f"{model_name}/val/loss", val_loss, epoch)
        writer.add_scalar(f"{model_name}/val/f1", val_metrics["f1"], epoch)
        writer.flush()

        # save best by val f1
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "cfg": cfg
            }
            ckpt_path = os.path.join(cfg["save_dir"], f"best_{model_name}.pth")
            torch.save(best_ckpt, ckpt_path)
            print(f"✅ Saved best checkpoint: {ckpt_path}")

            # TorchScript export (trace with example)
            try:
                example_input = torch.randn(1, 3, cfg["num_frames"], cfg["img_size"], cfg["img_size"]).to(device)
                # tracing works for both models
                traced = torch.jit.trace(model, example_input)
                traced_path = os.path.join(cfg["save_dir"], f"{model_name}_traced.pt")
                traced.save(traced_path)
                print(f"✅ Exported TorchScript to {traced_path}")
            except Exception as e:
                print("Warning: TorchScript export failed:", e)

    writer.close()
    return best_ckpt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="Shop DataSet")
    parser.add_argument("--model", type=str, default="both", choices=["cnn_rnn","3dcnn","both"])
    args = parser.parse_args()

    base_cfg = {
        "data_root": args.data_root,
        "num_frames": 32,
        "img_size": 112,
        "batch_size": 8,
        "epochs": 30,
        "lr": 3e-4,
        "weight_decay": 1e-5,
        "num_workers": 4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 42,
        "rnn_hidden": 256,
        "num_classes": 2,
        "save_dir": "checkpoints",
    }

    if args.model in ("cnn_rnn","both"):
        cfg = base_cfg.copy()
        cfg["model_type"] = "cnn_rnn"
        print("Starting training for CNN+RNN")
        train_one_model(cfg)

    if args.model in ("3dcnn","both"):
        cfg = base_cfg.copy()
        cfg["model_type"] = "3dcnn"
        print("Starting training for 3D-CNN")
        train_one_model(cfg)
