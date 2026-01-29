from ultralytics import YOLO
import cv2
import os
import glob
import numpy as np

# ================= CONFIGURATION =================
# 1. Path to your best trained model
MODEL_PATH = "runs/detect/cow_eye_final/weights/best.pt"

# 2. Where your raw test images are
INPUT_FOLDER = "test_input"

# 3. Where to save the results (with boxes)
OUTPUT_FOLDER = "test_output"
# =================================================

def preprocess_image(image_path):
    """
    Reads an image and converts it to the exact Grayscale format
    used during training.
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return None

    # Step 1: Force Grayscale (Removes Purple/Orange colors)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: Enhance Contrast (CLAHE)
    # This makes the "Hot Eye" pop out, just like in your training set
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # Step 3: Convert back to 3-Channel BGR
    # YOLO expects 3 channels (even if they are all identical gray values)
    model_input = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    return model_input

def main():
    # Create folders if they don't exist
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Error: Folder '{INPUT_FOLDER}' does not exist.")
        print("   -> Create it and put some test images inside!")
        return

    # Load the Model
    print(f"🚀 Loading model from: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"❌ Could not load model: {e}")
        return

    # Get all images
    files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG']:
        files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))

    if not files:
        print("⚠️  No images found in 'test_input'.")
        return

    print(f"📂 Found {len(files)} images. processing...")

    count = 0
    for file in files:
        filename = os.path.basename(file)
        
        # 1. Preprocess (Gray + CLAHE)
        processed_img = preprocess_image(file)
        if processed_img is None: continue

        # 2. Run Inference
        # conf=0.25 means "Only show boxes if 25% sure it's an eye"
        results = model(processed_img, conf=0.25, verbose=False)

        # 3. Draw the Box
        # .plot() draws the bounding box and label on the image
        result_img = results[0].plot()

        # 4. Save to Output
        save_path = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(save_path, result_img)
        
        print(f"✅ Processed: {filename}")
        count += 1

    print(f"\n🎉 Done! Check the '{OUTPUT_FOLDER}' folder to see the results.")

if __name__ == "__main__":
    main()
