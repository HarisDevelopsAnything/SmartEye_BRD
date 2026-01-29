from ultralytics import YOLO
import cv2
import glob
import random
import os

# 1. Load your trained model
model = YOLO('runs/detect/cow_eye_final/weights/best.pt')

# 2. Pick a random image from your validation set
val_images = glob.glob('yolo_dataset/images/val/*.jpg')
if not val_images:
    print("❌ No images found in val folder.")
    exit()

img_path = random.choice(val_images)
print(f"👀 Testing on: {os.path.basename(img_path)}")

# 3. Run Prediction
results = model(img_path)

# 4. Show Result
# YOLO has a built-in method to plot the boxes
res_plotted = results[0].plot()

# Save it so you can look at it
cv2.imwrite("test_result.jpg", res_plotted)
print("📸 Saved result to 'test_result.jpg'. Check it out!")
