from ultralytics import YOLO

# 1. Load your BEST High-Res model (Transfer Learning)
# We start with the smart brain that knows what a cow is.
model = YOLO('runs/detect/cow_eye_final/weights/best.pt') 

print("🚀 Starting Optimized MLX Fine-Tuning...")

# 2. Train with "Low-Res Safe" Settings
results = model.train(
    data='/home/haris/cow_project/yolo_dataset_mlx/data_mlx.yaml', 
    
    # --- CRITICAL CHANGES ---
    epochs=100,          # Give it more time to relearn
    imgsz=128,           # Lower resolution matches the MLX sensor better than 320
    batch=16,
    
    # --- DISABLING AUGMENTATIONS (The Magic Fix) ---
    mosaic=0.0,          # ❌ Turn off Mosaic (It breaks blocky images)
    mixup=0.0,           # ❌ Turn off Mixup (It confuses heat signatures)
    copy_paste=0.0,      # ❌ Turn off Copy-Paste
    
    # --- GENTLE GEOMETRY ---
    degrees=5.0,         # Only slight rotation (don't spin the blocks too much)
    translate=0.1,       # Only slight shifting
    scale=0.1,           # Only slight zooming
    
    # --- LEARNING ---
    lr0=0.005,           # Higher initial learning rate to adapt quickly
    lrf=0.01,            # Final learning rate
    
    device=0,
    name='cow_eye_mlx_improved'
)

print("✅ Improved Training Done!")

# 3. Export the final optimized model
model.export(
    format="tflite", 
    int8=True, 
    data='/home/haris/cow_project/yolo_dataset_mlx/data_mlx.yaml'
)
