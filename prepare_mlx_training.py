import cv2
import os
import glob
import shutil

# ================= CONFIGURATION =================
# 1. Your original dataset
ORIGINAL_DATASET = "yolo_dataset"

# 2. The new "MLX Style" dataset we will create
NEW_DATASET = "yolo_dataset_mlx"
# =================================================

def pixelate_image(img_path, save_path):
    img = cv2.imread(img_path)
    if img is None: return

    # 1. Force Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Downscale to 32x24 (MLX Resolution)
    tiny = cv2.resize(gray, (32, 24), interpolation=cv2.INTER_AREA)

    # 3. Upscale back to 320x320 (Nearest Neighbor for "Blocky" look)
    # This keeps the image size compatible with your existing label files!
    blocky = cv2.resize(tiny, (320, 320), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(save_path, blocky)

def main():
    print(f"🚀 Creating MLX Dataset in '{NEW_DATASET}'...")

    # 1. Copy the entire folder structure (labels, data.yaml, etc.)
    if os.path.exists(NEW_DATASET):
        shutil.rmtree(NEW_DATASET)
    shutil.copytree(ORIGINAL_DATASET, NEW_DATASET)
    print("✅ Copied original dataset structure.")

    # 2. Overwrite images with "Pixelated" versions
    for split in ['train', 'val']:
        img_folder = os.path.join(NEW_DATASET, 'images', split)
        images = glob.glob(os.path.join(img_folder, "*"))
        
        print(f"   Processing {len(images)} images in '{split}'...")
        
        for img_file in images:
            pixelate_image(img_file, img_file) # Overwrite the file

    print("\n🎉 Done! You now have a 'pixelated' dataset.")
    print(f"👉 Labels are preserved. You are ready to train on '{NEW_DATASET}'.")

if __name__ == "__main__":
    main()
