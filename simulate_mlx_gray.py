import cv2
import os
import glob

# ================= CONFIGURATION =================
INPUT_FOLDER = "test_input"        # Original High-Res images
OUTPUT_FOLDER = "test_input_mlx"   # Output folder for MLX simulation
# =================================================

def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # Get all images
    files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG']:
        files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))

    print(f"found {len(files)} images. Simulating Grayscale MLX90640 (32x24)...")

    for file in files:
        filename = os.path.basename(file)
        
        # 1. Read Original Image
        img = cv2.imread(file)
        if img is None: continue

        # 2. Convert to Grayscale FIRST
        # This strips the purple/orange Ironbow colors, leaving just raw intensity.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 3. Downscale to 32x24 (The MLX90640 Resolution)
        # We use INTER_AREA to average the pixels, simulating a larger sensor pixel.
        mlx_tiny = cv2.resize(gray, (32, 24), interpolation=cv2.INTER_AREA)

        # 4. Upscale back to 320x320 (For Visualization/Model Compatibility)
        # We use INTER_NEAREST to keep it "Blocky" (Minecraft style).
        # If we used linear interpolation, it would just look blurry.
        # We want the model to see the hard blocks, because that's what the real MLX output looks like.
        mlx_blocky = cv2.resize(mlx_tiny, (320, 320), interpolation=cv2.INTER_NEAREST)

        # 5. Save the result
        save_path = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(save_path, mlx_blocky)
        
        print(f"✅ Processed: {filename}")

    print(f"\n🎉 Done! Check '{OUTPUT_FOLDER}'.")
    print("👉 These images represent what the MLX90640 'sees'.")

if __name__ == "__main__":
    main()
