import cv2
import os
import glob
import numpy as np

# CONFIGURATION
INPUT_FOLDER = "dataset/flir_radiometric"  # Where your .jpgs are
OUTPUT_FOLDER = "yolo_dataset/images/train" # Where they go for training

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

print(f"🔄 Converting images from {INPUT_FOLDER}...")

# Get all images
files = glob.glob(os.path.join(INPUT_FOLDER, "*.jpg"))

for i, file in enumerate(files):
    try:
        # 1. Read Image
        img = cv2.imread(file)
        if img is None: continue

        # 2. Convert to Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 3. Enhance Contrast (CLAHE)
        # This makes the "Hot Eye" pop out against the cold face
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # 4. Save to YOLO folder
        filename = os.path.basename(file)
        save_path = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(save_path, enhanced)

    except Exception as e:
        print(f"Skipped {file}: {e}")

print(f"✅ Processed {len(files)} images. Ready for Annotation!")
