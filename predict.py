import sys
import numpy as np
import tensorflow as tf
from PIL import Image

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

if len(sys.argv) != 2:
    print("Usage: python predict.py path/to/image.jpg")
    raise SystemExit(1)

image_path = sys.argv[1]

model = tf.keras.models.load_model("models/cifar10_cnn.keras")

image = Image.open(image_path).convert("RGB").resize((32, 32))
image_array = np.array(image).astype("float32") / 255.0
image_array = np.expand_dims(image_array, axis=0)

probabilities = model.predict(image_array, verbose=0)[0]
index = int(np.argmax(probabilities))

print(f"Prediction: {CLASS_NAMES[index]}")
print(f"Confidence: {probabilities[index] * 100:.2f}%")
