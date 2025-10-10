# inference.py
import torch
import argparse
from dataset import VideoDataset
from models import CNN_RNN, Simple3DCNN
import cv2
import numpy as np
import torchvision.transforms as T

def predict_video(model_path, model_type, video_path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    num_frames = 32
    img_size = 112
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        raise RuntimeError("No frames found in video")

    # uniform sampling
    if len(frames) >= num_frames:
        idxs = np.linspace(0, len(frames)-1, num_frames, dtype=int)
        sampled = [frames[i] for i in idxs]
    else:
        sampled = frames.copy()
        while len(sampled) < num_frames:
            sampled.append(frames[-1])

    processed = [transform(f) for f in sampled]  # T x C x H x W
    x = torch.stack(processed).permute(1,0,2,3).unsqueeze(0).to(device)  # 1,C,T,H,W

    if model_type == "cnn_rnn":
        model = CNN_RNN(backbone_pretrained=False).to(device)
    else:
        model = Simple3DCNN().to(device)

    ckpt = torch.load(model_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0,1].item()
        pred = logits.argmax(dim=1).item()

    label = "Shop Lifter" if pred == 1 else "No Theft"
    return label, prob


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--model_type", default="cnn_rnn", choices=["cnn_rnn","3dcnn"])
    p.add_argument("--video", required=True)
    args = p.parse_args()

    label, prob = predict_video(args.model_path, args.model_type, args.video)
    print(f"Prediction: {label} (score={prob:.4f})")
