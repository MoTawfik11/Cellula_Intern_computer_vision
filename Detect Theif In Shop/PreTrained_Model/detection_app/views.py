import os
import torch
import torch.nn as nn
from django.shortcuts import render
from .forms import VideoUploadForm
from .utils.preprocess_videos import extract_frames_uniform
from torchvision.models.video import r3d_18

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_video_classifier.pth"
NUM_CLASSES = 2
NUM_FRAMES = 32

# Load model once globally
model = r3d_18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

def predict_video(video_path):
    frames = extract_frames_uniform(video_path, num_frames=NUM_FRAMES)
    x = torch.tensor(frames).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
    x = x.to(DEVICE)
    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        pred = outputs.argmax(1).item()
        confidence = probs[0, pred].item()
    label_names = ["Non Shop Lifter", "Shop Lifter"]
    return label_names[pred], round(confidence * 100, 2)

def index(request):
    result = None
    confidence = None

    if request.method == "POST":
        form = VideoUploadForm(request.POST, request.FILES)
        if form.is_valid():
            video = form.cleaned_data["video"]
            temp_path = os.path.join("media", video.name)
            os.makedirs("media", exist_ok=True)
            with open(temp_path, "wb+") as dest:
                for chunk in video.chunks():
                    dest.write(chunk)

            result, confidence = predict_video(temp_path)

    else:
        form = VideoUploadForm()

    return render(request, "index.html", {
        "form": form,
        "result": result,
        "confidence": confidence
    })
