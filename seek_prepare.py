import cv2
import os
import glob
import numpy as np

# ================= CONFIGURATION =================
# 1. Put your friend's raw Seek images here
INPUT_FOLDER = "seek_raw"      

# 2. Output folder (Your main training folder)
OUTPUT_FOLDER = "yolo_dataset/images/train" 

# 3. Naming Prefix
PREFIX = "seek_cow"  # Result: seek_cow_1.jpg, seek_cow_2.jpg
# =================================================

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Get all images (jpg, png, jpeg)
files = []
for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG']:
    files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))

print(f"📂 Found {len(files)} Seek images. Processing...")

for i, file in enumerate(files):
    try:
        # 1. Read the Colorful Image
        img = cv2.imread(file)
        if img is None: continue

        # 2. Convert to Grayscale (Purple -> Gray)
        # This makes the hottest parts (white/yellow) become bright white pixels
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 3. Enhance Contrast (CLAHE)
        # This matches the look of your FLIR data
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # 4. Save with new name (seek_cow_1.jpg)
        new_filename = f"{PREFIX}_{i+1}.jpg"
        save_path = os.path.join(OUTPUT_FOLDER, new_filename)
        
        cv2.imwrite(save_path, enhanced)
        print(f"✅ Converted: {os.path.basename(file)} -> {new_filename}")
        
    except Exception as e:
        print(f"❌ Error processing {file}: {e}")

print(f"\n🎉 Done! You now have consistent Grayscale images in '{OUTPUT_FOLDER}'.")
print("👉 Next Step: Open labelImg and annotate these new 'seek_cow' images.")
