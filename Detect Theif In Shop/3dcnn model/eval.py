# eval.py
import torch
from torch.utils.data import DataLoader
from models import CNN_RNN, Simple3DCNN
from dataset import VideoDataset
from utils import metrics_from_preds
import argparse
import os

def evaluate(model_path, model_type="cnn_rnn", data_root="Shop DataSet", batch_size=8, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = VideoDataset(data_root, num_frames=32, img_size=112, split="test", augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=(device.startswith("cuda")))
    if model_type == "cnn_rnn":
        model = CNN_RNN(backbone_pretrained=False)
    else:
        model = Simple3DCNN()
    model = model.to(device)
    ckpt = torch.load(model_path, map_location=device)
    # ckpt might be saved as dict with "model_state_dict"
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(1).cpu().numpy().tolist()
            y_pred.extend(preds)
            y_true.extend(y.numpy().tolist())

    metrics = metrics_from_preds(y_true, y_pred)
    print("Test metrics:", metrics)
    return metrics

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--model_type", default="cnn_rnn", choices=["cnn_rnn","3dcnn"])
    p.add_argument("--data_root", default="Shop DataSet")
    args = p.parse_args()
    evaluate(args.model_path, args.model_type, args.data_root)
