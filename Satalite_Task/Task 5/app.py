from flask import Flask, request, render_template
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import rasterio
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import base64  # <-- needed

app = Flask(__name__)

# ---- Load processor & trained model ----
processor = SegformerImageProcessor.from_pretrained("./final_model")
model = SegformerForSemanticSegmentation.from_pretrained("./final_model")
model.eval()

# ---- Load tif as RGB ----
def load_tif_as_rgb(file_stream):
    with rasterio.MemoryFile(file_stream) as memfile:
        with memfile.open() as src:
            img = np.stack([src.read(3), src.read(2), src.read(1)], axis=-1)
    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = img.astype(np.uint8)
    return Image.fromarray(img)

# ---- Colored mask ----
def generate_colored_mask(pred_seg):
    num_classes = model.config.num_labels
    cmap = plt.get_cmap("nipy_spectral", num_classes)
    mask_color = cmap(pred_seg / (num_classes - 1))[:, :, :3]
    mask_color = (mask_color * 255).astype(np.uint8)
    return Image.fromarray(mask_color)

# ---- Overlay mask on image ----
def overlay_mask_on_image(image, mask_color, alpha=0.5):
    return Image.blend(image.convert("RGBA"), mask_color.convert("RGBA"), alpha=alpha)

# ---- Convert PIL image to base64 string ----
def pil_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# ---- Home page ----
@app.route("/")
def home():
    return render_template("index.html")

# ---- Prediction ----
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    original_image = load_tif_as_rgb(file)

    # Preprocess
    inputs = processor(images=original_image, return_tensors="pt")

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Upsample
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=original_image.size[::-1],
        mode="bilinear",
        align_corners=False
    )

    pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

    # Colored mask and overlay
    mask_color = generate_colored_mask(pred_seg)
    overlay_img = overlay_mask_on_image(original_image, mask_color)

    # Convert to base64
    original_b64 = pil_to_base64(original_image)
    mask_b64 = pil_to_base64(mask_color)
    overlay_b64 = pil_to_base64(overlay_img)

    return render_template(
        "result.html",
        original_image=original_b64,
        mask_image=mask_b64,
        overlay_image=overlay_b64
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
