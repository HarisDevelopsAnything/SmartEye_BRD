import os
import glob

# ================= CONFIGURATION =================
# Path to your training images
FOLDER_PATH = "yolo_dataset/images/train" 

# The new base name (e.g., cow_1.jpg)
NEW_PREFIX = "cow" 
# =================================================

def main():
    # 1. Verify folder exists
    if not os.path.exists(FOLDER_PATH):
        print(f"❌ Error: Folder '{FOLDER_PATH}' not found.")
        return

    # 2. Get all image files
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(FOLDER_PATH, ext)))
    
    # Sort them so they are renamed in a consistent order
    files.sort()
    
    if not files:
        print("⚠️  No images found. Did you run the cleaner script?")
        return

    print(f"📂 Found {len(files)} images. Renaming started...")

    # 3. Rename Loop
    for i, old_path in enumerate(files):
        # Get file extension (usually .jpg)
        _, ext = os.path.splitext(old_path)
        
        # Create new filename: cow_1.jpg, cow_2.jpg ...
        new_filename = f"{NEW_PREFIX}_{i+1}{ext.lower()}"
        new_path = os.path.join(FOLDER_PATH, new_filename)
        
        # Rename
        try:
            os.rename(old_path, new_path)
        except OSError as e:
            print(f"❌ Error renaming {old_path}: {e}")

    print(f"✅ Success! All images are now named '{NEW_PREFIX}_n.jpg'")

if __name__ == "__main__":
    main()
