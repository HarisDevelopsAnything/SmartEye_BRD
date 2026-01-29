from ultralytics import YOLO
import cv2
import math
import time

# ================= CONFIGURATION =================
# Path to your HIGH RES model (The one with 75% accuracy)
MODEL_PATH = "runs/detect/cow_eye_mlx_improved/weights/best.pt"

# Confidence Threshold (Adjust if it's too shy or too noisy)
CONF_THRESHOLD = 0.40

# Camera ID (0 is usually the default laptop webcam)
CAMERA_ID = 0
# =================================================

def main():
    print(f"🚀 Loading High-Res Model: {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Open Webcam
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(3, 640) # Width
    cap.set(4, 480) # Height

    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    print("✅ Webcam started. Press 'Q' to quit.")
    print("ℹ️  Applying Grayscale + CLAHE to simulate thermal input...")

    while True:
        success, img = cap.read()
        if not success:
            break

        # --- STEP 1: SIMULATE THERMAL LOOK ---
        # Your model doesn't know what 'color' is. It expects high-contrast gray.
        
        # Convert to Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This makes bright spots pop out, mimicking a heat signature.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # Convert back to BGR (Model expects 3 channels, even if gray)
        input_frame = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        # --- STEP 2: INFERENCE ---
        results = model(input_frame, conf=CONF_THRESHOLD, verbose=False)

        # --- STEP 3: VISUALIZATION ---
        # We draw boxes on the 'input_frame' (the gray one) so you see what the AI sees.
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 

                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100

                # Class Name
                cls = int(box.cls[0])
                label = f"Eye {conf}"

                # Draw Box (Green)
                cv2.rectangle(input_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                # Draw Label Background (Green) & Text (White)
                t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(input_frame, (x1, y1), c2, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.putText(input_frame, label, (x1, y1 - 2), 0, 0.5, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        # Show the processed "Thermal-Simulated" feed
        cv2.imshow('Cow Eye Detector (High Res)', input_frame)

        # Press 'Q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
