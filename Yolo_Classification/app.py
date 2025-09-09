import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image

# ---------------------------
# Config
# ---------------------------
MODEL_PATH = r"E:\\Computer Vision\\Cellula Tech\\Task 2\\runs\\classify\\train\\weights\\best.onnx"
CLASS_NAMES = ["CaS", "CoS", "Gum", "MC", "OC", "OLP", "OT"] 

# Load ONNX runtime session
session = ort.InferenceSession(MODEL_PATH, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

# ---------------------------
# Streamlit App
# ---------------------------
st.set_page_config(page_title="Teeth Disease Classifier", layout="centered")

st.title("🦷 Teeth Disease Classification (ONNX)")
st.write("Upload a teeth image and the model will classify it into disease categories.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)  # ✅ fix deprecation warning

    # Preprocess: resize → normalize → add batch dim
    img_size = (224, 224)
    img = image.resize(img_size)
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))  # HWC → CHW
    img_array = np.expand_dims(img_array, axis=0)   # add batch dimension

    # Run inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    preds = session.run([output_name], {input_name: img_array})[0]

    # Softmax for probabilities
    exp_preds = np.exp(preds[0])
    probs = exp_preds / np.sum(exp_preds)

    # Top-1 prediction
    top1_idx = int(np.argmax(probs))
    top1_label = CLASS_NAMES[top1_idx]
    top1_conf = probs[top1_idx] * 100

    st.subheader("Prediction")
    st.success(f"**{top1_label}** ({top1_conf:.2f}%)")

    # Top-3 predictions
    top3_idx = np.argsort(probs)[-3:][::-1]
    top3_labels = [CLASS_NAMES[i] for i in top3_idx]
    top3_conf = [probs[i] for i in top3_idx]

    st.subheader("Top-3 Predictions")
    st.bar_chart({label: conf for label, conf in zip(top3_labels, top3_conf)})
