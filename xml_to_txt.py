import os
import glob
import xml.etree.ElementTree as ET

# ================= CONFIGURATION =================
# Path to your labels folder (where the XMLs are)
LABELS_FOLDER = "yolo_dataset/labels/train"

# Dictionary to map class names to ID
# You only have 'eye', so it maps to ID 0
CLASS_MAP = {
    "eye": 0
}
# =================================================

def convert(size, box):
    """ Converts PascalVOC (xmin, ymin...) to YOLO (x, y, w, h) """
    dw = 1./size[0]
    dh = 1./size[1]
    
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def main():
    # Find all XML files
    xml_files = glob.glob(os.path.join(LABELS_FOLDER, "*.xml"))
    
    if not xml_files:
        print("❌ No XML files found. Check your path!")
        return

    print(f"🔄 Converting {len(xml_files)} XML files to YOLO format...")

    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # 1. Get Image Size
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        # 2. Create the TXT filename
        # e.g., cow_1.xml -> cow_1.txt
        txt_filename = os.path.splitext(xml_file)[0] + ".txt"
        
        with open(txt_filename, 'w') as out_file:
            for obj in root.iter('object'):
                cls = obj.find('name').text
                
                # If you accidentally capitalized "Eye", this fixes it
                if cls not in CLASS_MAP and cls.lower() in CLASS_MAP:
                    cls = cls.lower()

                if cls not in CLASS_MAP:
                    print(f"⚠️ Warning: Unknown class '{cls}' in {xml_file}")
                    continue
                
                cls_id = CLASS_MAP[cls]
                
                # 3. Get Bounding Box
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), 
                     float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                
                # 4. Convert and Write
                bb = convert((w,h), b)
                out_file.write(f"{cls_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")

    print("✅ Conversion Complete!")
    print("👉 You can now delete the .xml files if you want to keep it clean.")

if __name__ == "__main__":
    main()
