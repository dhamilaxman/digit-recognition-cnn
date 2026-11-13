import os, string, random, shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from keras import layers, models
from keras.datasets import mnist
import tf2onnx

fonts = [os.path.join('C:/Windows/Fonts', f) for f in os.listdir('C:/Windows/Fonts') if f.endswith('.ttf')]

def render_sample(char, font_path, size=180, angle=0):
    img = Image.new('L', (280, 280), 0)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, size)
        draw.text((70, 40), char, fill=255, font=font)
    except: return None
    if angle != 0:
        img = img.rotate(angle)
    bbox = img.point(lambda p: 255 if p > 40 else 0).getbbox()
    if not bbox: return None
    cropped = img.crop(bbox)
    scale = 20.0 / max(cropped.size)
    new_size = (max(1, round(cropped.width * scale)), max(1, round(cropped.height * scale)))
    resized = cropped.resize(new_size, Image.LANCZOS)
    canvas = Image.new('L', (28, 28), 0)
    canvas.paste(resized, ((28 - new_size[0]) // 2, (28 - new_size[1]) // 2))
    arr = np.array(canvas, dtype=np.float32)
    total = arr.sum()
    if total == 0: return None
    ys, xs = np.indices(arr.shape)
    cy = (ys * arr).sum() / total
    cx = (xs * arr).sum() / total
    arr = np.roll(arr, (int(round(13.5 - cy)), int(round(13.5 - cx))), axis=(0, 1))
    return arr / 255.0

def main():
    print("1. Generating balanced font digits and font letters...")
    font_digits = []
    font_letters = []

    # 8,000 font digits
    while len(font_digits) < 8000:
        d = random.choice(string.digits)
        f = random.choice(fonts)
        s = render_sample(d, f, random.randint(140, 210), random.randint(-18, 18))
        if s is not None: font_digits.append(s)

    # 8,000 font letters (uppercase and lowercase)
    letters_pool = list('ABCDEFGHIJKMNPQRTUVWXYZ') + list('abcdefghikmnpqrtuvwxy')
    while len(font_letters) < 8000:
        l = random.choice(letters_pool)
        f = random.choice(fonts)
        s = render_sample(l, f, random.randint(140, 210), random.randint(-18, 18))
        if s is not None: font_letters.append(s)

    # 2,000 shapes/scribbles
    shapes = []
    while len(shapes) < 2000:
        img = Image.new('L', (280, 280), 0)
        draw = ImageDraw.Draw(img)
        t = random.choice(['cross', 'triangle', 'scribble', 'star'])
        w = random.randint(16, 26)
        if t == 'cross':
            draw.line([(50, 50), (230, 230)], fill=255, width=w)
            draw.line([(50, 230), (230, 50)], fill=255, width=w)
        elif t == 'triangle':
            draw.polygon([(140, 40), (40, 240), (240, 240)], outline=255, width=w)
        elif t == 'scribble':
            pts = [(random.randint(40, 240), random.randint(40, 240)) for _ in range(5)]
            draw.line(pts, fill=255, width=w)
        elif t == 'star':
            draw.line([(140, 40), (140, 240)], fill=255, width=w)
            draw.line([(40, 140), (240, 140)], fill=255, width=w)
        bbox = img.point(lambda p: 255 if p > 40 else 0).getbbox()
        if not bbox: continue
        cropped = img.crop(bbox)
        scale = 20.0 / max(cropped.size)
        new_size = (max(1, round(cropped.width * scale)), max(1, round(cropped.height * scale)))
        resized = cropped.resize(new_size, Image.LANCZOS)
        canvas = Image.new('L', (28, 28), 0)
        canvas.paste(resized, ((28 - new_size[0]) // 2, (28 - new_size[1]) // 2))
        arr = np.array(canvas, dtype=np.float32)
        total = arr.sum()
        if total == 0: continue
        ys, xs = np.indices(arr.shape)
        cy = (ys * arr).sum() / total
        cx = (xs * arr).sum() / total
        arr = np.roll(arr, (int(round(13.5 - cy)), int(round(13.5 - cx))), axis=(0, 1))
        shapes.append(arr / 255.0)

    (x_train, _), _ = mnist.load_data()
    mnist_digits = (x_train[:8000] / 255.0).astype(np.float32)

    # 16,000 Digits (8k MNIST + 8k Font digits)
    x_digits = np.vstack([mnist_digits, np.array(font_digits, dtype=np.float32)]).reshape(-1, 28, 28, 1)
    y_digits = np.ones((len(x_digits), 1), dtype=np.float32)

    # 10,000 Non-digits (8k letters + 2k shapes)
    x_nondigits = np.vstack([np.array(font_letters, dtype=np.float32), np.array(shapes, dtype=np.float32)]).reshape(-1, 28, 28, 1)
    y_nondigits = np.zeros((len(x_nondigits), 1), dtype=np.float32)

    x_all = np.vstack([x_digits, x_nondigits])
    y_all = np.vstack([y_digits, y_nondigits])
    idx = np.random.permutation(len(x_all))
    x_all, y_all = x_all[idx], y_all[idx]

    print(f"2. Dataset: {len(x_all)} total samples (16k digits, 10k non-digits).")
    m = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    m.fit(x_all, y_all, epochs=5, batch_size=64, validation_split=0.15, verbose=1)

    saved_model_dir = 'C:/Users/dhami/Deep Learning Project/digit-web/saved_validator'
    if os.path.exists(saved_model_dir):
        shutil.rmtree(saved_model_dir)
    m.export(saved_model_dir)
    print("SavedModel exported.")

if __name__ == '__main__':
    main()
