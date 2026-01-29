from ultralytics import YOLO
import torch
import os

# 1. Force GPU
torch.cuda.set_device(0)

# 2. Load Model (Nano version is best for Raspberry Pi)
model = YOLO('yolov8n.pt') 

print(f"🚀 Starting Training on {torch.cuda.get_device_name(0)}...")

# 3. Train
results = model.train(
    data='/home/haris/cow_project/yolo_dataset/data.yaml', 
    epochs=100,      # 100 epochs ensures it learns well
    imgsz=320,       # 320x320 is faster on Pi than 640
    batch=16,        # If you get "Out of Memory", change to 8
    device=0,
    name='cow_eye_final',
    patience=20      # Stop early if it stops improving
)

print("✅ Training Done! Exporting to TFLite...")

# 4. Export for Raspberry Pi (Int8 Quantized)
path = model.export(format="tflite", int8=True)
print(f"🎉 Model saved at: {path}")
