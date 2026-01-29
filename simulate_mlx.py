import cv2
import os
import glob

# ================= CONFIGURATION =================
INPUT_FOLDER = "test_input"       # Your original high-res images
OUTPUT_FOLDER = "test_input_mlx"  # Where to save the 32x24 simulations
# =================================================

def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # Get all images
    files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG']:
        files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))

    print(f"found {len(files)} images. Simulating MLX90640 (32x24)...")

    for file in files:
        filename = os.path.basename(file)
        
        # 1. Read Original Image
        img = cv2.imread(file)
        if img is None: continue

        # 2. Downscale to 32x24 (MLX Resolution)
        # INTER_AREA is best for shrinking images without aliasing
        mlx_tiny = cv2.resize(img, (32, 24), interpolation=cv2.INTER_AREA)

        # 3. Upscale back to 320x320 (Optional - For Visualization ONLY)
        # We save the TINY version for testing, but if you want to see
        # how "blocky" it is for the AI, we can upscale it with Nearest Neighbor.
        # This keeps the sharp "Minecraft blocks" look.
        mlx_blocky = cv2.resize(mlx_tiny, (320, 320), interpolation=cv2.INTER_NEAREST)

        # 4. Save the result
        # We save the upscaled 'blocky' version so the batch_test script 
        # doesn't have to struggle resizing a 32px tiny dot.
        # This represents exactly the information loss of the MLX90640.
        save_path = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(save_path, mlx_blocky)
        
        print(f"✅ Converted: {filename}")

    print(f"\n🎉 Done! Images saved in '{OUTPUT_FOLDER}'.")
    print("👉 Now run 'batch_test.py' and change INPUT_FOLDER to 'test_input_mlx'.")

if __name__ == "__main__":
    main()
