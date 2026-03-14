import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

DATA_DIR = r"D:\Licenta\proiect Licenta\dataset_final_crop" 

IMG_HEIGHT = 32
IMG_WIDTH = 32
CHANNELS = 3 

NUM_CLASSES = 43 

def load_data(data_dir, num_classes):
    """
    Încarcă imaginile și etichetele din foldere.
    Returnează listele brute de imagini și etichete.
    """
    images = []
    labels = []
    
    print("Se încarcă imaginile...")
    
    for i in range(num_classes):
        path = os.path.join(data_dir, str(i))
        if not os.path.exists(path):
            print(f"Atenție: Folderul {path} nu există! Trecem peste.")
            continue
            
        classes_images = os.listdir(path)
        
        for img_name in classes_images:
            try:
                img_path = os.path.join(path, img_name)
                image = cv2.imread(img_path)
               
                image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                
                images.append(image)
                labels.append(i)
            except Exception as e:
                print(f"Eroare la încărcarea imaginii {img_name}: {e}")

    images = np.array(images)
    labels = np.array(labels)
    
    print(f"Încărcare completă: {len(images)} imagini găsite pentru {num_classes} clase.")
    return images, labels

def preprocess_for_cnn(images):
    """
    Pentru CNN:
    1. Păstrăm culorile (RGB).
    2. Normalizăm pixelii: valorile devin între 0 și 1 (împarțim la 255).
       Acest pas ajută rețeaua neuronală să învețe mai repede.
    """
    print("Preprocesare pentru CNN (Normalizare)...")
    images_cnn = images.astype("float32") / 255.0
    
    # normaliazre la media rgb

    return images_cnn

def preprocess_for_hog(images):
    """
    Pentru SVM clasic folosim HOG:
    1. Convertim la Grayscale (HOG lucrează pe gradientul de intensitate, culoarea contează mai puțin).
    2. Extragem trăsăturile HOG -> transformăm imaginea într-un vector de numere.
    """
    print("Preprocesare pentru SVM (Extragere HOG)...")
    hog_features = []
    
    for image in images:
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        features = hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys',
                       visualize=False, transform_sqrt=True)
        
        hog_features.append(features)
        
    return np.array(hog_features)

if __name__ == "__main__":
    
    raw_images, raw_labels = load_data(DATA_DIR, NUM_CLASSES)
    
    if len(raw_images) > 0:
       
        X_cnn = preprocess_for_cnn(raw_images)
        
        X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(
            X_cnn, raw_labels, test_size=0.2, random_state=42
        )
        print(f"Date CNN pregătite: Train shape {X_train_cnn.shape}")

        X_hog = preprocess_for_hog(raw_images)
       
        X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(
            X_hog, raw_labels, test_size=0.2, random_state=42
        )
        print(f"Date SVM (HOG) pregătite: Train shape {X_train_svm.shape}")
        
        print("Gata de antrenare!")
    else:
        print("Nu s-au găsit imagini. Verifică calea folderului DATA_DIR.")