import base64
import io
import os

import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, render_template, request
from PIL import Image

app = Flask(__name__)

# Load ONNX model (lightweight, no TensorFlow needed)
session = ort.InferenceSession("digit_model.onnx")
input_name = session.get_inputs()[0].name


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

    result = session.run(None, {input_name: arr})
    probs = result[0][0]
    sorted_probs = np.sort(probs)[::-1]
    confidence = float(sorted_probs[0])
    second_confidence = float(sorted_probs[1])
    margin = confidence - second_confidence
    digit = int(np.argmax(probs))

    # Reject non-digits, alphabets, or ambiguous drawings
    # High threshold (95%) and margin requirement (>85%) ensures only clear digits pass
    if confidence < 0.95 or margin < 0.85:
        return jsonify(
            error="Not recognized as a digit. Please draw a clear digit (0-9)."
        ), 400

    return jsonify(digit=digit, confidence=round(confidence * 100, 1))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)






