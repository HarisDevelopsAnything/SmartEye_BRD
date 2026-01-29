import cv2
import os
import numpy as np
import glob

# ================= USER SETTINGS =================
# Adjust these slightly if the red box doesn't cover the text perfectly
BOX_WIDTH_RATIO = 0.16   # The text box is ~16% of the image width
BOX_HEIGHT_RATIO = 0.07  # The text box is ~7% of the image height
BOX_OFFSET_Y = 0.04      # The box starts 4% down from the center

# Input/Output Folders
INPUT_FOLDER = "seek_raw"
OUTPUT_FOLDER = "seek_clean"
DEBUG_FOLDER = "dataset_check_alignment" # LOOK IN HERE FIRST!
# =============================================

def create_seek_mask(h, w):
    """ Creates a precise mask for the Seek Thermal overlay. """
    mask = np.zeros((h, w), dtype="uint8")
    cX, cY = w // 2, h // 2

    # 1. THE RETICLE (Center Crosshair)
    # It's small, so we erase a small circle in the dead center
    cv2.circle(mask, (cX, cY), int(w * 0.02), 255, -1)

    # 2. THE TEMPERATURE BUBBLE (Rectangle below center)
    box_w = int(w * BOX_WIDTH_RATIO)
    box_h = int(h * BOX_HEIGHT_RATIO)
    
    # Calculate top-left corner of the box
    box_x = int(cX - (box_w / 2))
    box_y = int(cY + (h * BOX_OFFSET_Y))

    # Draw the rectangle on the mask (White = Erase this)
    cv2.rectangle(mask, (box_x, box_y), (box_x + box_w, box_y + box_h), 255, -1)
    
    return mask

def process_images():
    # Make directories
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(DEBUG_FOLDER, exist_ok=True)

    files = glob.glob(os.path.join(INPUT_FOLDER, "*"))
    print(f"🚀 Found {len(files)} images. Processing...")

    for i, file_path in enumerate(files):
        img = cv2.imread(file_path)
        if img is None: continue

        h, w = img.shape[:2]
        
        # 1. Create the Eraser Mask
        mask = create_seek_mask(h, w)

        # 2. SAVE DEBUG IMAGE (Only for the first 5 images)
        # This draws a RED box over the original so you can check alignment
        if i < 5:
            debug_img = img.copy()
            # Set pixels to RED where the mask is white
            debug_img[mask == 255] = [0, 0, 255] 
            debug_name = os.path.join(DEBUG_FOLDER, f"debug_{os.path.basename(file_path)}")
            cv2.imwrite(debug_name, debug_img)

        # 3. ERASE (Inpainting)
        # Radius 3 is cleaner than 5 for text boxes
        clean_img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

        # 4. Save Result
        out_name = os.path.join(OUTPUT_FOLDER, os.path.basename(file_path))
        cv2.imwrite(out_name, clean_img)

    print(f"\n✅ DONE!")
    print(f"⚠️  CRITICAL: Open the '{DEBUG_FOLDER}' folder now.")
    print(f"    - If the RED BOX covers the text perfectly, you are good.")
    print(f"    - If the text peeks out, increase 'BOX_WIDTH_RATIO' or 'BOX_HEIGHT_RATIO' in the script.")

if __name__ == "__main__":
    process_images()
