import os, string, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from keras import layers, models
from keras.datasets import mnist
import tf2onnx
import shutil

def get_valid_fonts():
    font_dir = 'C:/Windows/Fonts'
    valid = []
    test_chars = 'ABCDEF'
    for f in os.listdir(font_dir):
        if not f.lower().endswith('.ttf'):
            continue
        p = os.path.join(font_dir, f)
        try:
            font = ImageFont.truetype(p, 60)
            img = Image.new('L', (100, 100), 0)
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), test_chars, fill=255, font=font)
            valid.append(p)
        except Exception:
            pass
    return valid

def main():
    print("1. Loading full MNIST digits...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # 25,000 real digits
    x_digits = (x_train[:25000] / 255.0).astype(np.float32).reshape(-1, 28, 28, 1)
    y_digits = np.ones((len(x_digits), 1), dtype=np.float32)

    print("2. Generating non-digits (letters & non-digit symbols)...")
    fonts = get_valid_fonts()
    non_digits = []
    
    # Distinct alphabet letters (excluding ambiguous lookalikes O/0, l/1, Z/2, S/5)
    chars = list('ABCDEFGHIJKMNPQRTUVWXYZ') + list('abcdefghikmnpqrtuvwxy')
    
    # Generate 18,000 letters
    while len(non_digits) < 18000:
        char = random.choice(chars)
        font_path = random.choice(fonts)
        angle = random.randint(-18, 18)
        size = random.randint(140, 210)
        try:
            font = ImageFont.truetype(font_path, size)
            img = Image.new('L', (280, 280), 0)
            draw = ImageDraw.Draw(img)
            draw.text((random.randint(50, 80), random.randint(30, 50)), char, fill=255, font=font)
            if angle != 0:
                img = img.rotate(angle)
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
            non_digits.append(arr / 255.0)
        except Exception:
            continue

    # Generate 7,000 shapes, scribbles and symbols
    print("3. Generating shapes and scribbles...")
    while len(non_digits) < 25000:
        img = Image.new('L', (280, 280), 0)
        draw = ImageDraw.Draw(img)
        shape_type = random.choice(['cross', 'triangle', 'scribble', 'star', 'arrow', 'box'])
        w = random.randint(16, 26)
        if shape_type == 'cross':
            draw.line([(50, 50), (230, 230)], fill=255, width=w)
            draw.line([(50, 230), (230, 50)], fill=255, width=w)
        elif shape_type == 'triangle':
            draw.polygon([(140, 40), (40, 240), (240, 240)], outline=255, width=w)
        elif shape_type == 'scribble':
            pts = [(random.randint(40, 240), random.randint(40, 240)) for _ in range(random.randint(4, 7))]
            draw.line(pts, fill=255, width=w)
        elif shape_type == 'star':
            draw.line([(140, 40), (140, 240)], fill=255, width=w)
            draw.line([(40, 140), (240, 140)], fill=255, width=w)
            draw.line([(60, 60), (220, 220)], fill=255, width=w)
        elif shape_type == 'arrow':
            draw.line([(140, 50), (140, 230)], fill=255, width=w)
            draw.line([(140, 50), (90, 100)], fill=255, width=w)
            draw.line([(140, 50), (190, 100)], fill=255, width=w)
        elif shape_type == 'box':
            draw.rectangle([(50, 50), (230, 230)], outline=255, width=w)
        
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
        non_digits.append(arr / 255.0)

    x_nondigits = np.array(non_digits, dtype=np.float32).reshape(-1, 28, 28, 1)
    y_nondigits = np.zeros((len(x_nondigits), 1), dtype=np.float32)

    x_all = np.vstack([x_digits, x_nondigits])
    y_all = np.vstack([y_digits, y_nondigits])
    idx = np.random.permutation(len(x_all))
    x_all, y_all = x_all[idx], y_all[idx]

    print(f"4. Dataset prepared: {len(x_all)} total samples (25k digits, 25k non-digits).")
    
    # Lightweight CNN architecture
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("5. Training Validator CNN...")
    model.fit(x_all, y_all, epochs=5, batch_size=64, validation_split=0.15, verbose=1)

    # Clean previous saved_validator
    saved_model_dir = 'C:/Users/dhami/Deep Learning Project/digit-web/saved_validator'
    if os.path.exists(saved_model_dir):
        shutil.rmtree(saved_model_dir)
        
    model.export(saved_model_dir)
    print("SavedModel exported.")

if __name__ == '__main__':
    main()
