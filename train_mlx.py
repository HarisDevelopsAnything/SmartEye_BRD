from ultralytics import YOLO

# 1. Load your ALREADY TRAINED model (Transfer Learning)
# Instead of 'yolov8n.pt', we use your cow brain.
model = YOLO('runs/detect/cow_eye_final/weights/best.pt') 

print("🚀 Fine-tuning model for MLX90640 (Blocky Images)...")

results = model.train(
    data='/home/haris/cow_project/yolo_dataset_mlx/data_mlx.yaml', 
    epochs=50,       # Fewer epochs needed because it already knows what a cow is
    imgsz=320,
    batch=16,
    device=0,
    name='cow_eye_mlx_finetune',
    lr0=0.001        # Lower learning rate to gently adjust the weights
)

print("✅ Fine-tuning Done!")
model.export(format="tflite", int8=True, data='/home/haris/cow_project/yolo_dataset_mlx/data_mlx.yaml')
