import os
import glob
import cv2
from ultralytics import YOLO

# CONFIGURATION
INPUT_FOLDER = "test_input"
MODEL_PATH = "runs/detect/cow_eye_final/weights/best.pt"

print("----- DIAGNOSTIC START -----")

# 1. Check Model File
if not os.path.exists(MODEL_PATH):
    print(f"❌ CRITICAL ERROR: Model file not found at {MODEL_PATH}")
    exit()
print(f"✅ Model file exists.")

# 2. Check Input Folder
if not os.path.exists(INPUT_FOLDER):
    print(f"❌ CRITICAL ERROR: Input folder '{INPUT_FOLDER}' does not exist.")
    exit()

# 3. List Files
files = glob.glob(os.path.join(INPUT_FOLDER, "*"))
print(f"📂 Found {len(files)} files in '{INPUT_FOLDER}'")

if len(files) == 0:
    print("❌ ERROR: The folder is empty! Put some .jpg files in there.")
    exit()

# 4. Load Model
print("⏳ Loading YOLO model... (This usually takes 5-10 seconds)")
try:
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Model Load Failed: {e}")
    exit()

# 5. Process First Image Only (for testing)
first_file = files[0]
print(f"\n--- Testing First File: {os.path.basename(first_file)} ---")

# Try to read image
img = cv2.imread(first_file)
if img is None:
    print("❌ ERROR: cv2.imread returned None. The image format might be unsupported or corrupt.")
else:
    print(f"✅ Image loaded. Dimensions: {img.shape}")
    
    # Force preprocessing to match training (Gray -> CLAHE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    input_tensor = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    # Run Prediction with extremely low confidence
    print("🔎 Running inference with conf=0.01 (1%)...")
    results = model(input_tensor, conf=0.05, verbose=True)
    
    # Check boxes
    box_count = len(results[0].boxes)
    print(f"📊 Result: Found {box_count} bounding boxes.")
    
    if box_count > 0:
        print(f"📍 Box Coordinates: {results[0].boxes.xywh.cpu().numpy()}")
        results[0].save(filename="diagnostic_result.jpg")
        print("💾 Saved visualization to 'diagnostic_result.jpg'")
    else:
        print("⚠️ No objects detected. The model sees nothing.")

print("\n----- DIAGNOSTIC END -----")
