from ultralytics import YOLO

# 1. Load the BEST model you just trained
# (YOLO automatically saved it in the 'runs' folder)
model = YOLO('runs/detect/cow_eye_final/weights/best.pt') 

print("🔄 Re-exporting with correct Thermal Calibration...")

# 2. Export with the 'data' argument
# This forces it to look at YOUR thermal images while compressing
model.export(
    format='tflite', 
    int8=True, 
    data='/home/haris/cow_project/yolo_dataset/data.yaml'  # <--- The Magic Fix
)

print("✅ Fixed! Your new TFLite model is optimized for HEAT.")
