import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2, preprocess_input, decode_predictions
)
from tensorflow.keras.preprocessing.image import img_to_array

# -------------------- Custom Background + Font Size --------------------
page_style = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #3e2723, #8d6e63, #cfd8dc);
    background-attachment: fixed;
    color: white;
    font-size: 20px !important;
}
h1 {font-size: 48px !important;}
h2 {font-size: 36px !important;}
h3 {font-size: 28px !important;}
p, div, span {font-size: 20px !important;}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #2c3e50, #4ca1af);
}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# -------------------- Load AI Model --------------------
model = MobileNetV2(weights="imagenet")

# -------------------- Title & Subtitle --------------------
st.title("✨ File Activities & Image Features Analyzer")
st.subheader("Upload an image to explore its objects, colors, and transformations")

# -------------------- File Upload --------------------
file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# -------------------- Color Extraction --------------------
def get_colors(image, k=3):
    img = np.array(image.convert("RGB"))
    img = cv2.resize(img, (100, 100))
    img = img.reshape((-1, 3))
    img = np.float32(img)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(img, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    counts = np.bincount(labels.flatten())
    total = sum(counts)

    colors = []
    for count, center in zip(counts, centers):
        percent = round((count / total) * 100, 2)
        rgb = tuple(map(int, center))
        colors.append((rgb, percent))
    return colors

# -------------------- Main Logic --------------------
if file:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    # -------- File Details --------
    st.header("📏 File Details")
    st.write(f"Mode: {image.mode}")
    st.write(f"Size: {image.size[0]} x {image.size[1]} pixels")
    st.write(f"Format: {image.format if image.format else 'Unknown'}")

    # -------- Object Detection --------
    img = image.resize((224, 224))
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)

    preds = model.predict(arr)
    labels = decode_predictions(preds, top=3)[0]

    st.header("🔍 Objects Detected")
    for l in labels:
        st.write(f"{l[1]} : {round(l[2] * 100, 2)} %")

    # -------- Color Analysis --------
    st.header("🎨 Dominant Colors")
    colors = get_colors(image)
    for rgb, percent in colors:
        st.markdown(
            f"<span style='color:rgb{rgb}; font-weight:bold;'>RGB{rgb}</span> "
            f"≈ {percent}% of image",
            unsafe_allow_html=True
        )

    # Pie chart visualization
    labels = [f"RGB{rgb}" for rgb, _ in colors]
    sizes = [percent for _, percent in colors]
    color_values = [tuple(v/255 for v in rgb) for rgb, _ in colors]

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=color_values, autopct='%1.1f%%', startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

    # -------- Grayscale Preview --------
    st.header("🌑 Grayscale Preview")
    gray = np.array(image.convert("L"))
    st.image(gray, caption="Grayscale", use_container_width=True)

    # -------- Edge Detection --------
    st.header("⚡ Edge Detection")
    threshold1 = st.slider("Lower Threshold", 50, 300, 100)
    threshold2 = st.slider("Upper Threshold", 50, 300, 200)
    edges = cv2.Canny(np.array(image), threshold1, threshold2)
    st.image(edges, caption=f"Edges (thresholds: {threshold1}, {threshold2})", use_container_width=True)