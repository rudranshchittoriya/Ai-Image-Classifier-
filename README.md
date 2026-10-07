# 🧠 CNN Image Classifier

A beginner-friendly Deep Learning project that uses a Convolutional Neural Network (CNN) with TensorFlow/Keras to classify images from the CIFAR-10 dataset.

## Features
- Automatic CIFAR-10 dataset download
- CNN model training
- Training/validation accuracy and loss graphs
- Model evaluation on the test set
- Single-image prediction
- Confusion matrix
- Saved trained model
- Streamlit demo interface

## Tech Stack
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- Pillow

## Project Structure
```text
dl_cnn_image_classifier/
├── app.py
├── train.py
├── predict.py
├── evaluate.py
├── requirements.txt
├── .gitignore
├── README.md
└── models/
    └── .gitkeep
```

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/dl-cnn-image-classifier.git
cd dl-cnn-image-classifier
```

### 2. Create a virtual environment
```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

macOS/Linux:
```bash
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model
```bash
python train.py
```

The trained model will be saved in `models/cifar10_cnn.keras`.

### 5. Evaluate the model
```bash
python evaluate.py
```

### 6. Run the web app
```bash
streamlit run app.py
```

## CIFAR-10 Classes
The model predicts:
`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`

## GitHub Upload
```bash
git init
git add .
git commit -m "Initial CNN image classifier"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/dl-cnn-image-classifier.git
git push -u origin main
```

## Portfolio Value
This project demonstrates practical understanding of CNNs, image preprocessing, model training, evaluation, and deployment. It is a good first Deep Learning portfolio project for a BTech AI/ML student.

## Author
Rudransh Chittoriya 

B.tech Ai/Ml
