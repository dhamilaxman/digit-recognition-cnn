import base64
import io
import os

import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, render_template, request
from PIL import Image

app = Flask(__name__)

# Load ONNX models (lightweight, no TensorFlow needed at runtime)
digit_session = ort.InferenceSession("digit_model.onnx")
digit_input_name = digit_session.get_inputs()[0].name

validator_session = ort.InferenceSession("validator.onnx")
validator_input_name = validator_session.get_inputs()[0].name


def preprocess(data_url):
    """Turn the canvas drawing (white on black) into a 28x28 MNIST-style input."""
    raw = base64.b64decode(data_url.split(",", 1)[1])
    img = Image.open(io.BytesIO(raw)).convert("L")

    # Crop tightly around the digit
    bbox = img.point(lambda p: 255 if p > 40 else 0).getbbox()
    if bbox is None:
        return None
    img = img.crop(bbox)

    # Fit into 20x20 without stretching, place on a 28x28 canvas
    scale = 20.0 / max(img.size)
    new_size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    img = img.resize(new_size, Image.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(img, ((28 - new_size[0]) // 2, (28 - new_size[1]) // 2))

    # Centre by centre of mass, like MNIST digits
    arr = np.array(canvas, dtype=np.float32)
    total = arr.sum()
    ys, xs = np.indices(arr.shape)
    cy = (ys * arr).sum() / total
    cx = (xs * arr).sum() / total
    arr = np.roll(arr, (int(round(13.5 - cy)), int(round(13.5 - cx))), axis=(0, 1))

    return (arr / 255.0).reshape(1, 28, 28, 1)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    image = payload.get("image")
    arr = preprocess(image) if image else None
    if arr is None:
        return jsonify(error="Draw a digit first."), 400

    # Ink density check (filters out tiny specks or excessive scribbles)
    active_pixels = int(np.sum(arr > 0.2))
    if active_pixels < 35:
        return jsonify(
            error="Drawing is too faint or small. Please draw a clear digit."
        ), 400
    if active_pixels > 250:
        return jsonify(
            error="Drawing is too dense or complex. Please draw a single digit (0-9)."
        ), 400

    # 1. Gatekeeper Validator: Checks if the drawing is an alphabet or symbol
    digit_score = float(
        validator_session.run(None, {validator_input_name: arr})[0][0][0]
    )

    # 2. Digit Classifier CNN
    result = digit_session.run(None, {digit_input_name: arr})
    probs = result[0][0]
    confidence = float(np.max(probs))
    digit = int(np.argmax(probs))

    # Single vertical line (digit 1) exception
    is_valid_1 = (digit == 1 and confidence > 0.65)

    if digit_score < 0.35 and not is_valid_1:
        return jsonify(
            error="Alphabet or non-digit symbol detected. Please draw a digit (0-9)."
        ), 400

    # Low confidence check
    if confidence < 0.40:
        return jsonify(
            error="Not recognized clearly as a digit. Please draw a clear digit (0-9)."
        ), 400

    return jsonify(digit=digit, confidence=round(confidence * 100, 1))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)






