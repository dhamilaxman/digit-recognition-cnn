# Handwritten Digit Recognition with CNN

A deep learning web application that recognizes handwritten digits (0-9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

## 🎯 Project Details

| Field | Details |
|-------|---------|
| **University** | Invertis University, CSED |
| **Program** | B.Tech (CSE) 2024 |
| **Mentor** | Pradeep Singh |
| **Team Members** | Aman Pal , Laxman Singh , Vijay Mishra |
| **Project ID** | DL-001 |
| **Algorithm** | Convolutional Neural Network (CNN) |
| **Dataset** | MNIST (60,000 training + 10,000 test images, 28×28 grayscale) |

## 🧠 Model Architecture

| Layer | Output Shape | Parameters |
|-------|-------------|------------|
| Conv2D (32 filters, 3×3) | (None, 26, 26, 32) | 320 |
| MaxPooling2D (2×2) | (None, 13, 13, 32) | 0 |
| Conv2D (64 filters, 3×3) | (None, 11, 11, 64) | 18,496 |
| MaxPooling2D (2×2) | (None, 5, 5, 64) | 0 |
| Flatten | (None, 1600) | 0 |
| Dense (128, ReLU) | (None, 128) | 204,928 |
| Dense (10, Softmax) | (None, 10) | 1,290 |

**Total Parameters:** 225,034

## 🚀 How It Works

1. Draw a digit (0-9) on the canvas
2. Click **Predict**
3. The CNN model processes your drawing and predicts the digit

## 🛠️ Tech Stack

- **Backend:** Python, Flask, TensorFlow/Keras
- **Frontend:** HTML5 Canvas, JavaScript
- **Deployment:** Render (Gunicorn)

## 📦 Local Setup

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000 in your browser.

## 📁 Project Structure

```
digit-web/
├── app.py                  # Flask web server + prediction API
├── digit_model.keras       # Trained CNN model
├── requirements.txt        # Python dependencies
├── render.yaml             # Render deployment config
├── templates/
│   └── index.html          # Frontend UI
└── README.md
```
