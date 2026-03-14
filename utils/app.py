import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import joblib

# IMPORTURI NOI PENTRU PYTORCH
import torch
import torch.nn as nn
import torch.nn.functional as F

# Importăm funcțiile de preprocesare existente
from preprocessing import preprocess_for_hog, NUM_CLASSES

#1. DEFINIREA MODELULUI (Copiată exact din train_CNN.py)
class TrafficSignNet(nn.Module):
    def __init__(self, num_classes):
        super(TrafficSignNet, self).__init__()
        
        self.strat1=nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.strat2=nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.strat3=nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU()
        )
        
        self.dense=nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x): 
        x1=self.strat1(x)
        x2=self.strat2(x1)
        x3=self.strat3(x2)
        x4=self.dense(x3)
        return x4

#CONFIGURĂRI GUI
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Dicționarul de clase (COMPLETEAZĂ-L CU CLASELE TALE!)
CLASSES = {
    0: "Limita 20",
    1: "Limita 30",
    2: "Limita 50",
    3: "Limita 60",
    4: "Limita 70",
    5: "Limita 80",
    6: "Iesire limita 80",
    7: "Limita 100",
    8: "Limita 120",
    9: "Interis depasire",
    10: "Interis depasire camioane",
    11: "Intersectie",
    12: "Prioritate",
    13: "Cedeaza trecerea",
    14: "STOP",
    15: "Interzis ambele sensuri",
    16: "Interzis camioane",
    17: "Interzis",
    18: "Alte pericole",
    19: "Curba la st.",
    20: "Curba la dr.",
    21: "Curba dubla prima dr.",
    22: "Drum denivelat",
    23: "Drum alunecos",
    24: "Drum ingustat pe dr.",
    25: "Atentie lucrari",
    26: "Semafor",
    27: "Presemnalizare trecere pietoni",
    28: "Copii",
    29: "Biciclisti",
    30: "Zapada",
    31: "Animale",
    32: "Sfarsit restrictii",
    33: "Drea. Obligatoriu",
    34: "Stg. Obligatoriu",
    35: "Inainte",
    36: "Inainte sau la st.",
    37: "Inainte sau la dr.",
    38: "Patreaza Dr.",
    39: "Ocolire",
    40: "Sens giratoriu",
    41: "Iesire zona interzis depasire",
    42: "Iesire zona interzis depasire camioane",
}

class TrafficSignApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Sistem Avansat Detecție & Recunoaștere Auto")
        self.geometry("1100x700")

        self.cnn_model = None
        self.svm_model = None
        self.original_cv_image = None
        self.processed_cv_image = None

        # Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        #SIDEBAR
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        ctk.CTkLabel(self.sidebar, text="Meniu Principal", font=("Arial", 20, "bold")).pack(pady=20)

        self.btn_load = ctk.CTkButton(self.sidebar, text="Încarcă Imagine Scenă", command=self.load_image)
        self.btn_load.pack(pady=10, padx=20)

        ctk.CTkLabel(self.sidebar, text="-----------------").pack(pady=10)

        self.btn_detect = ctk.CTkButton(self.sidebar, text="🔍 Detectează & Compară", 
                                        fg_color="#E59400", hover_color="#B87700",
                                        height=40, font=("Arial", 14, "bold"),
                                        command=self.process_full_image)
        self.btn_detect.pack(pady=20, padx=20)

        self.log_box = ctk.CTkTextbox(self.sidebar, width=220, height=300)
        self.log_box.pack(pady=20, padx=10)
        self.log_box.insert("0.0", "Jurnal activități...\n")

        #ZONA IMAGINE
        self.image_frame = ctk.CTkFrame(self, fg_color="#2b2b2b")
        self.image_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        self.img_label = ctk.CTkLabel(self.image_frame, text="Așteptare imagine...")
        self.img_label.pack(expand=True, fill="both")

        # Încărcare Modele
        self.load_models()

    def log(self, message):
        self.log_box.insert("end", f"> {message}\n")
        self.log_box.see("end")

    def load_models(self):
        try:
            self.log("Se încarcă modelele...")
            
            #Încărcare CNN PyTorch
            # 1. Instanțiem clasa
            self.cnn_model = TrafficSignNet(NUM_CLASSES)
            # 2. Încărcăm greutățile (weights)
            self.cnn_model.load_state_dict(torch.load('traffic_sign_CNN.pth', map_location=torch.device('cpu')))
            # 3. CRUCIAL: Modul 'eval' oprește Dropout-ul la testare
            self.cnn_model.eval() 
            
            #Încărcare SVM
            self.svm_model = joblib.load('traffic_sign_svm.pkl')
            
            self.log("Modele încărcate cu succes (PyTorch + SVM).")
        except Exception as e:
            self.log(f"Eroare modele: {e}")

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg;*.png;*.jpeg")])
        if file_path:
            self.original_cv_image = cv2.imread(file_path)
            self.display_image(self.original_cv_image)
            self.log("Imagine încărcată.")

    def display_image(self, cv_img):
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img.thumbnail((800, 600))
        self.current_tk_image = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
        self.img_label.configure(image=self.current_tk_image, text="")

    def detect_signs_regions(self, image):
        regions = []
        
        # Un mic blur pentru a netezi pixelii
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Măști de culoare - le facem un pic mai permisive la lumină (Saturație de la 70 în sus)
        mask_r1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
        mask_r2 = cv2.inRange(hsv, np.array([160, 70, 50]), np.array([180, 255, 255]))
        mask_red = cv2.bitwise_or(mask_r1, mask_r2)
        
        mask_blue = cv2.inRange(hsv, np.array([90, 70, 50]), np.array([140, 255, 255]))

        combined_mask = cv2.bitwise_or(mask_red, mask_blue)

        # Curățare zgomot (Morfologie)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # FILTRUL 1: Mărimea. Scădem la 300 ca să prindă și semnele mai îndepărtate
            if area > 300: 
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h
                
                # FILTRUL 2: Proporțiile. Un semn e de obicei echilibrat (aprox pătrat)
                # Îi dăm o marjă mai mare (0.6 - 1.4) pentru poze făcute din unghi
                if 0.6 <= aspect_ratio <= 1.4:
                    
                    # FILTRUL 3: Soliditatea (cât de plin e conturul)
                    # O formă neregulată (ca un om în geacă roșie) va avea soliditate mică
                    hull = cv2.convexHull(cnt)
                    hull_area = cv2.contourArea(hull)
                    if hull_area > 0:
                        solidity = float(area) / hull_area
                        
                        # Un semn rotund/triunghiular e cam 50%-100% plin.
                        if solidity > 0.4:
                            regions.append((x, y, w, h))
                                
        return regions

    def process_full_image(self):
        if self.original_cv_image is None:
            self.log("Încarcă o imagine întâi!")
            return

        self.log("Începere scanare...")
        output_img = self.original_cv_image.copy()
        
        boxes = self.detect_signs_regions(output_img)
        self.log(f"Detectate {len(boxes)} potențiale semne.")

        for (x, y, w, h) in boxes:
            pad = 5
            roi = self.original_cv_image[max(0, y-pad):min(output_img.shape[0], y+h+pad),
                                         max(0, x-pad):min(output_img.shape[1], x+w+pad)]
            
            if roi.size == 0: continue

#debug imagini
            # cv2.imwrite(f"debug_roi_{x}_{y}.jpg", roi)

            #1. PREDICȚIE CNN (PyTorch)
            try:
                # Resize și Normalizare
                roi_resized = cv2.resize(roi, (32, 32))
                roi_norm = roi_resized.astype("float32") / 255.0
                
                # Transformare pentru PyTorch: (H, W, C) -> (C, H, W)
                roi_transposed = np.transpose(roi_norm, (2, 0, 1))
                
                # Convertire la Tensor și adăugare dimensiune Batch -> (1, 3, 32, 32)
                roi_tensor = torch.from_numpy(roi_transposed).unsqueeze(0)
                
                # Inferență fără a calcula gradienți (economisește memorie)
                with torch.no_grad():

                    outputs = self.cnn_model(roi_tensor)
                    # Aplicăm Softmax pentru a transforma output-ul în probabilități
                    probs = F.softmax(outputs, dim=1)
                    conf_cnn_tensor, predicted = torch.max(probs, 1)
                    
                    idx_cnn = predicted.item()
                    conf_cnn = conf_cnn_tensor.item() * 100
                    
                label_cnn = CLASSES.get(idx_cnn, "?")
            except Exception as e:
                self.log(f"Eroare CNN: {e}")
                label_cnn = "Err"
                conf_cnn = 0

            if conf_cnn < 90.0:
                continue

            #2. PREDICȚIE SVM
            try:
                roi_hog_input = cv2.resize(roi, (32, 32))
                feats = preprocess_for_hog(np.array([roi_hog_input])) 
                idx_svm = self.svm_model.predict(feats)[0]
                label_svm = CLASSES.get(idx_svm, "?")
            except:
                label_svm = "Err"

            #3. DESENARE REZULTATE 
            cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            text_cnn = f"CNN: {label_cnn} ({int(conf_cnn)}%)"
            text_svm = f"SVM: {label_svm}"
            
            cv2.rectangle(output_img, (x, y-40), (x+w+50, y), (0, 0, 0), -1)
            cv2.putText(output_img, text_cnn, (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(output_img, text_svm, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            self.log(f"Găsit la ({x},{y}): C={label_cnn} | S={label_svm}")

        self.display_image(output_img)
        self.log("Procesare completă.")

if __name__ == "__main__":
    app = TrafficSignApp()
    app.mainloop()