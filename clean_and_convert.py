import flirimageextractor
import numpy as np
import cv2
import os
import glob

# CONFIGURATION
# 1. Input: Your raw FLIR images (with the overlay)
INPUT_FOLDER = "dataset/flir_radiometric"  

# 2. Output: Clean images for YOLO (Overlay removed)
OUTPUT_FOLDER = "yolo_dataset/images/train"

# 3. Path to ExifTool (Arch Linux default)
EXIFTOOL_PATH = "/usr/bin/exiftool"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def normalize_to_image(thermal_data):
    """
    Converts raw temperature data (e.g., 25.0 - 40.0) into a 0-255 grayscale image.
    """
    # Normalize to 0-1 range
    min_val = np.min(thermal_data)
    max_val = np.max(thermal_data)
    norm = (thermal_data - min_val) / (max_val - min_val)
    
    # Scale to 0-255 and convert to integer
    img_8bit = (norm * 255).astype(np.uint8)
    return img_8bit

def main():
    print(f"🚀 Initializing FLIR Extractor...")
    try:
        flir = flirimageextractor.FlirImageExtractor(exiftool_path=EXIFTOOL_PATH)
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    files = glob.glob(os.path.join(INPUT_FOLDER, "*.jpg"))
    print(f"📂 Found {len(files)} images. cleaning them now...")

    for i, file in enumerate(files):
        filename = os.path.basename(file)
        save_path = os.path.join(OUTPUT_FOLDER, filename)
        
        try:
            # 1. Extract the Raw Thermal Matrix (This has NO OVERLAY)
            flir.process_image(file)
            thermal_np = flir.get_thermal_np()
            
            # 2. Convert to a standard Image (Grayscale)
            clean_img = normalize_to_image(thermal_np)
            
            # 3. Enhance Contrast (Optional but recommended for eyes)
            # This makes the hot eye pop out against the cold background
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            final_img = clahe.apply(clean_img)
            
            # 4. Save
            cv2.imwrite(save_path, final_img)
            print(f"✅ Cleaned: {filename}")
            
        except Exception as e:
            print(f"⚠️  Skipping {filename}: Not radiometric or corrupt.")

    print(f"\n🎉 Done! Check the folder '{OUTPUT_FOLDER}'.")
    print("The images inside will be Black & White and completely FREE of text/overlays.")

if __name__ == "__main__":
    main()
