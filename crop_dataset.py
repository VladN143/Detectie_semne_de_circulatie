import os
import cv2
import glob

LABELS_FOLDER = r"D:\Licenta\proiect Licenta\Traffic_sign_detection_data\labels\train"

IMAGES_FOLDER = r"D:\Licenta\proiect Licenta\Traffic_sign_detection_data\images\train" 

OUTPUT_FOLDER = r"dataset_final_crop"

IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG']

def process_dataset():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    txt_files = glob.glob(os.path.join(LABELS_FOLDER, "*.txt"))
    
    print(f"Am găsit {len(txt_files)} fișiere text în folderul de etichete.")
    print(f"Caut imaginile corespunzătoare în: {IMAGES_FOLDER} ...")
    
    total_crops = 0
    images_processed = 0
    images_not_found = 0
    
    for txt_path in txt_files:
        filename = os.path.basename(txt_path)
        
        if filename == "classes.txt":
            continue
            
        base_name = os.path.splitext(filename)[0]
      
        image_path = None
        for ext in IMG_EXTENSIONS:
            potential_path = os.path.join(IMAGES_FOLDER, base_name + ext)
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if image_path is None:
           
            if images_not_found < 5:
                print(f"EROARE: Am fișierul {filename}, dar NU găsesc imaginea: {base_name}.jpg/png în folderul de imagini.")
            images_not_found += 1
            continue
              
        img = cv2.imread(image_path)
        if img is None:
            continue
            
        h_img, w_img, _ = img.shape
        images_processed += 1
        
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5: continue
                
                try:
                    class_id = int(float(parts[0]))
                    x_c = float(parts[1]) * w_img
                    y_c = float(parts[2]) * h_img
                    w_box = float(parts[3]) * w_img
                    h_box = float(parts[4]) * h_img
                    
                    x_min = max(0, int(x_c - w_box / 2))
                    y_min = max(0, int(y_c - h_box / 2))
                    x_max = min(w_img, int(x_c + w_box / 2))
                    y_max = min(h_img, int(y_c + h_box / 2))
                    
                    crop_img = img[y_min:y_max, x_min:x_max]
                    
                    if crop_img.size > 0:
                        class_dir = os.path.join(OUTPUT_FOLDER, str(class_id))
                        if not os.path.exists(class_dir):
                            os.makedirs(class_dir)
                        
                        save_name = f"{base_name}_{i}.jpg"
                        cv2.imwrite(os.path.join(class_dir, save_name), crop_img)
                        total_crops += 1
                        
                except Exception as e:
                    pass

    print(f"\n--- RAPORT FINAL ---")
    print(f"Fișiere text găsite: {len(txt_files)}")
    print(f"Imagini găsite și procesate: {images_processed}")
    print(f"Imagini lipsă: {images_not_found}")
    print(f"Semne decupate total: {total_crops}")
    
    if images_processed == 0:
        print("\nATENȚIE: Tot 0 imagini. Verifică dacă folderul IMAGES_FOLDER conține pozele!")

if __name__ == "__main__":
    process_dataset()