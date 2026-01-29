import os
import subprocess
import json
import numpy as np
import pandas as pd
import flirimageextractor

# ================= CONFIGURATION =================
# We scan this folder AND all subfolders inside it
FOLDER_PATH = "dataset"

# Output file
OUTPUT_FILE = "training_dataset.csv"

# ExifTool Path (Arch Linux)
EXIFTOOL_PATH = "/usr/bin/exiftool"
# =================================================

def get_ambient_temp(filepath):
    cmd = [EXIFTOOL_PATH, '-ReflectedApparentTemperature', '-AtmosphericTemperature', '-j', filepath]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        meta = json.loads(result.stdout.decode('utf-8'))[0]
        ambient = meta.get('AtmosphericTemperature', meta.get('ReflectedApparentTemperature', 20.0))
        if isinstance(ambient, str):
            ambient = float(ambient.replace(' C', ''))
        return ambient
    except:
        return 20.0

def main():
    print(f"🚀 Scanning recursively inside: {FOLDER_PATH}...")

    try:
        flir = flirimageextractor.FlirImageExtractor(exiftool_path=EXIFTOOL_PATH)
    except Exception as e:
        print(f"❌ Error initializing FLIR tool: {e}")
        return

    data = []

    # os.walk is the magic command that digs into subfolders
    for root, dirs, files in os.walk(FOLDER_PATH):
        for file in files:
            if file.lower().endswith('.jpg'):
                filepath = os.path.join(root, file)

                try:
                    # 1. Get Thermal Data
                    flir.process_image(filepath)
                    thermal_np = flir.get_thermal_np()

                    # 2. Get Metadata
                    ambient_temp = get_ambient_temp(filepath)

                    # 3. Calculate Stats
                    max_eye_temp = np.max(thermal_np)
                    avg_temp = np.mean(thermal_np)

                    # 4. Save
                    data.append({
                        "filename": file,
                        "max_eye_temp": round(max_eye_temp, 2),
                        "ambient_temp": round(ambient_temp, 2),
                        "label": 0  # <--- REMEMBER TO EDIT THIS LATER!
                    })
                    print(f"✅ {file}: Max={max_eye_temp:.2f}°C")

                except:
                    # If it fails, it's probably a Seek image or normal photo (skip it)
                    pass

    # Save CSV
    if data:
        df = pd.DataFrame(data)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n🎉 Done! Found {len(data)} valid FLIR images.")
        print(f"💾 Data saved to: {os.path.abspath(OUTPUT_FILE)}")
    else:
        print("\n❌ No valid radiometric images found in any subfolder.")

if __name__ == "__main__":
    main()
