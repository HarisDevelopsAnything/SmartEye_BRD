import os
import random
import shutil

# CONFIGURATION
# Your project root
BASE_DIR = os.path.abspath("yolo_dataset")

IMAGES_TRAIN = os.path.join(BASE_DIR, "images/train")
LABELS_TRAIN = os.path.join(BASE_DIR, "labels/train")

IMAGES_VAL = os.path.join(BASE_DIR, "images/val")
LABELS_VAL = os.path.join(BASE_DIR, "labels/val")

# Create val folders if not exist
os.makedirs(IMAGES_VAL, exist_ok=True)
os.makedirs(LABELS_VAL, exist_ok=True)

# 1. Get list of all images
images = [f for f in os.listdir(IMAGES_TRAIN) if f.endswith('.jpg')]
total_images = len(images)
val_count = int(total_images * 0.2)  # 20% for validation

print(f"📂 Found {total_images} total images.")
print(f"🚚 Moving {val_count} images to Validation folder...")

# 2. Randomly select files
random.shuffle(images)
val_files = images[:val_count]

for img_file in val_files:
    # Construct paths
    src_img = os.path.join(IMAGES_TRAIN, img_file)
    dst_img = os.path.join(IMAGES_VAL, img_file)
    
    # Matching label file (e.g., cow_1.jpg -> cow_1.txt)
    label_file = img_file.replace(".jpg", ".txt")
    src_label = os.path.join(LABELS_TRAIN, label_file)
    dst_label = os.path.join(LABELS_VAL, label_file)
    
    # Move Image
    shutil.move(src_img, dst_img)
    
    # Move Label (if it exists)
    if os.path.exists(src_label):
        shutil.move(src_label, dst_label)
    else:
        print(f"⚠️ Warning: No label found for {img_file}")

print("✅ Split Complete! You are ready to train.")
