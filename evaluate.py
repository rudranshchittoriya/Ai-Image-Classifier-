import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

model = tf.keras.models.load_model("models/cifar10_cnn.keras")
(_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_test = x_test.astype("float32") / 255.0
y_test = y_test.flatten()

predictions = np.argmax(model.predict(x_test, verbose=0), axis=1)

print(classification_report(y_test, predictions, target_names=CLASS_NAMES))

cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)

fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
plt.title("CIFAR-10 Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
