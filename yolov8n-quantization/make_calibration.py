
import cv2
import glob
import numpy as np
import os

IMAGE_DIR = "coco300/images/"
OUTPUT = "calibration_640.npy"
SIZE = 640
MAX_IMAGES = 500

files = sorted(glob.glob(IMAGE_DIR + "/*.jpg"))

if len(files) < 300:
    raise RuntimeError(
        f"Only {len(files)} images found. Need at least 300."
    )

samples = []

for path in files[:MAX_IMAGES]:
    img = cv2.imread(path)

    if img is None:
        continue

    h, w = img.shape[:2]

    # YOLO letterbox preprocessing
    scale = min(SIZE / h, SIZE / w)

    nh = int(round(h * scale))
    nw = int(round(w * scale))

    resized = cv2.resize(img, (nw, nh))

    canvas = np.full(
        (SIZE, SIZE, 3), 114, dtype=np.uint8
    )

    top = (SIZE - nh) // 2
    left = (SIZE - nw) // 2

    canvas[top:top+nh, left:left+nw] = resized

    # BGR -> RGB
    canvas = cv2.cvtColor(
        canvas, cv2.COLOR_BGR2RGB
    )

    # Normalize to [0,1]
    canvas = canvas.astype(np.float32) / 255.0

    samples.append(canvas)

data = np.stack(samples)

np.save(OUTPUT, data)

print("Saved:", OUTPUT)
print("Shape:", data.shape)
print("Dtype:", data.dtype)
print("Range:", data.min(), data.max())

