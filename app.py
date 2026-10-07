import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

st.set_page_config(
    page_title="CNN Image Classifier",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 CNN Image Classifier")
st.write("Upload an image and let the trained CNN predict its CIFAR-10 class.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/cifar10_cnn.keras")

try:
    model = load_model()
except Exception:
    st.error("Model not found. Run `python train.py` first.")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    resized = image.resize((32, 32))
    array = np.array(resized).astype("float32") / 255.0
    array = np.expand_dims(array, axis=0)

    probabilities = model.predict(array, verbose=0)[0]
    index = int(np.argmax(probabilities))

    st.success(f"Prediction: {CLASS_NAMES[index]}")
    st.metric("Confidence", f"{probabilities[index] * 100:.2f}%")

    st.subheader("Top Predictions")
    top_indices = np.argsort(probabilities)[-3:][::-1]

    for i in top_indices:
        st.write(f"**{CLASS_NAMES[i]}** — {probabilities[i] * 100:.2f}%")
        st.progress(float(probabilities[i]))
