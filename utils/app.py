import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import numpy as np
import cv2
import joblib
import torch
import torch.nn.functional as F
from ultralytics import YOLO
import os


from preprocessing import preprocess_for_hog, NUM_CLASSES


ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

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
    9: "Interzis depasire", 
    10: "Interzis depasire camioane", 
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
    21: "Curba dubla", 
    22: "Drum denivelat", 
    23: "Drum alunecos",
    24: "Drum ingustat", 
    25: "Atentie lucrari", 
    26: "Presemnalizare semafor",
    27: "Trecere pietoni", 
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
        self.geometry("1200x750") 

        self.cnn_model = None
        self.svm_model = None
        self.yolo_detector = None
        self.original_cv_image = None

        
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        #1. SIDEBAR
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

        #2. TABVIEW (ZONA PRINCIPALĂ DIN DREAPTA)
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        self.tabview.add("Scanare Imagine")
        self.tabview.add("Performanță Modele")
        
        tab_scan = self.tabview.tab("Scanare Imagine")
        tab_scan.grid_columnconfigure(0, weight=1)
        tab_scan.grid_rowconfigure(0, weight=1) 
        
        self.img_label = ctk.CTkLabel(tab_scan, text="Așteptare imagine...")
        self.img_label.grid(row=0, column=0, sticky="nsew")

        self.results_frame = ctk.CTkScrollableFrame(tab_scan, orientation="horizontal", height=150)
        self.results_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        # Configurare Tab 2: Grafice 
        self.setup_graphs_tab()

        self.load_models()

    def setup_graphs_tab(self):
        tab = self.tabview.tab("Performanță Modele")
        
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
     
        scroll_frame = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll_frame.grid(row=0, column=0, sticky="nsew")
        
        scroll_frame.grid_columnconfigure(0, weight=1)
        
        # ZONA CNN
        ctk.CTkLabel(scroll_frame, text="Rezultate Antrenare CNN (PyTorch)", 
                     font=("Arial", 20, "bold"), text_color="#00FF00").grid(row=0, column=0, pady=(20, 10))
        
        self.lbl_graph_cnn = ctk.CTkLabel(scroll_frame, text="Grafic CNN lipsă")
        self.lbl_graph_cnn.grid(row=1, column=0, padx=20, pady=(0, 40)) 
        
        # ZONA SVM 
        ctk.CTkLabel(scroll_frame, text="Rezultate Antrenare SVM", 
                     font=("Arial", 20, "bold"), text_color="#E59400").grid(row=2, column=0, pady=(10, 10))
        
        self.lbl_graph_svm = ctk.CTkLabel(scroll_frame, text="Grafic SVM lipsă")
        self.lbl_graph_svm.grid(row=3, column=0, padx=20, pady=(0, 30))
        
        self.load_graphs()

    def load_graphs(self):
        cnn_path = "grafic CNN.png" 
        svm_path = "grafic SVM.png" 
        
        max_size = (850, 500) 
        
        if os.path.exists(cnn_path):
            img_cnn = Image.open(cnn_path)
            img_cnn.thumbnail(max_size, Image.Resampling.LANCZOS) 
            tk_cnn = ctk.CTkImage(light_image=img_cnn, dark_image=img_cnn, size=img_cnn.size)
            self.lbl_graph_cnn.configure(image=tk_cnn, text="")
        else:
            self.log(f"Nu am găsit imaginea: {cnn_path}")
            
        if os.path.exists(svm_path):
            img_svm = Image.open(svm_path)
            img_svm.thumbnail(max_size, Image.Resampling.LANCZOS)
            tk_svm = ctk.CTkImage(light_image=img_svm, dark_image=img_svm, size=img_svm.size)
            self.lbl_graph_svm.configure(image=tk_svm, text="")
        else:
            self.log(f"Nu am găsit imaginea: {svm_path}")

    def clear_previous_results(self):
       
        for widget in self.results_frame.winfo_children():
            widget.destroy()

    def add_detection_to_ui(self, roi_cv, label_cnn, conf_cnn, label_svm):
        roi_rgb = cv2.cvtColor(roi_cv, cv2.COLOR_BGR2RGB)
        roi_pil = Image.fromarray(roi_rgb)
        roi_pil = roi_pil.resize((80, 80)) 
        tk_roi = ctk.CTkImage(light_image=roi_pil, dark_image=roi_pil, size=(80, 80))

        card = ctk.CTkFrame(self.results_frame, fg_color="#3d3d3d", border_width=2, border_color="#555555")
        card.pack(side="left", padx=10, pady=10)
        
        img_lbl = ctk.CTkLabel(card, image=tk_roi, text="")
        img_lbl.pack(pady=5, padx=5)

        cnn_frame = ctk.CTkFrame(card, fg_color="transparent")
        cnn_frame.pack(fill="x", padx=5)
        ctk.CTkLabel(cnn_frame, text="CNN:", font=("Arial", 10, "bold"), text_color="#00FF00").pack(side="left")
        ctk.CTkLabel(cnn_frame, text=f"{label_cnn} ({int(conf_cnn)}%)", font=("Arial", 10)).pack(side="left", padx=5)

        svm_frame = ctk.CTkFrame(card, fg_color="transparent")
        svm_frame.pack(fill="x", padx=5, pady=(0, 5))
        ctk.CTkLabel(svm_frame, text="SVM:", font=("Arial", 10, "bold"), text_color="#E59400").pack(side="left")
        ctk.CTkLabel(svm_frame, text=f"{label_svm}", font=("Arial", 10)).pack(side="left", padx=5)

        img_lbl.image = tk_roi

    def log(self, message):
        self.log_box.insert("end", f"> {message}\n")
        self.log_box.see("end")

    def load_models(self):
        try:
            self.log("Se încarcă modelele...")
            self.yolo_detector = YOLO('D:/Licenta/proiect Licenta/runs/detect/train8/weights/best.pt')
            
            from train_cnn import TrafficSignNet

            self.cnn_model = TrafficSignNet(NUM_CLASSES)
            self.cnn_model.load_state_dict(torch.load('traffic_sign_CNN.pth', map_location=torch.device('cpu')))
            self.cnn_model.eval() 
            
            self.svm_model = joblib.load('traffic_sign_svm.pkl')
            self.log("Modele încărcate cu succes (PyTorch + SVM).")
        except Exception as e:
            self.log(f"Eroare modele: {e}")

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Imagini", "*.jpg;*.jpeg;*.png;*.webp;*.bmp"), ("Toate fișierele", "*.*")]
        )
        
        if file_path:
            try:
                img_array = np.fromfile(file_path, np.uint8)
                self.original_cv_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if self.original_cv_image is None:
                    self.log("Eroare: Fișierul selectat nu este o imagine validă!")
                    return

                self.clear_previous_results()
                self.tabview.set("Scanare Imagine")
                self.display_image(self.original_cv_image)
                self.log(f"Imagine încărcată: {file_path.split('/')[-1]}")
                    
            except Exception as e:
                self.log(f"Eroare la deschiderea imaginii: {e}")

    def display_image(self, cv_img):
        if len(cv_img.shape) == 3 and cv_img.shape[2] == 4:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2BGR)
            
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img.thumbnail((800, 500)) 
        self.current_tk_image = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
        self.img_label.configure(image=self.current_tk_image, text="")

    def detect_signs_regions(self, image):
        regions = []
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.yolo_detector(image_rgb, conf=0.3, verbose=False)
    
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                w = x2 - x1
                h = y2 - y1
                regions.append((x1, y1, w, h))
        return regions

    def process_full_image(self):
        if self.original_cv_image is None:
            self.log("Încarcă o imagine întâi!")
            return

        self.log("Începere scanare...")
        
        self.clear_previous_results()
        
        output_img = self.original_cv_image.copy()
        boxes = self.detect_signs_regions(output_img)
        self.log(f"Detectate {len(boxes)} potențiale semne.")

        for (x, y, w, h) in boxes:
            pad = 5
            roi = self.original_cv_image[max(0, y-pad):min(output_img.shape[0], y+h+pad),
                                         max(0, x-pad):min(output_img.shape[1], x+w+pad)]
            if roi.size == 0: continue

            if roi.shape[0] < 5 or roi.shape[1] < 5:
                continue

            try:
                roi_resized = cv2.resize(roi, (32, 32))
                roi_norm = roi_resized.astype("float32") / 255.0
                roi_transposed = np.transpose(roi_norm, (2, 0, 1))
                roi_tensor = torch.from_numpy(roi_transposed).unsqueeze(0)
                
                with torch.no_grad():
                    outputs = self.cnn_model(roi_tensor)
                    probs = F.softmax(outputs, dim=1)
                    conf_cnn_tensor, predicted = torch.max(probs, 1)
                    
                    idx_cnn = predicted.item()
                    conf_cnn = conf_cnn_tensor.item() * 100
                    
                label_cnn = CLASSES.get(idx_cnn, "?")
            except Exception as e:
                self.log(f"Eroare CNN: {e}")
                label_cnn = "Err"
                conf_cnn = 0

            try:
                roi_hog_input = cv2.resize(roi, (32, 32))
                feats = preprocess_for_hog(np.array([roi_hog_input])) 
                idx_svm = self.svm_model.predict(feats)[0]
                label_svm = CLASSES.get(idx_svm, "?")
            except:
                label_svm = "Err"

            self.add_detection_to_ui(roi, label_cnn, conf_cnn, label_svm)

            cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text_cnn = f"CNN: {label_cnn} ({int(conf_cnn)}%)"
            text_svm = f"SVM: {label_svm}"
            
            cv2.rectangle(output_img, (x, y-40), (x+max(w, 180), y), (0, 0, 0), -1)
            cv2.putText(output_img, text_cnn, (x, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(output_img, text_svm, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            self.log(f"Găsit la ({x},{y}): C={label_cnn} | S={label_svm}")

        self.display_image(output_img)
        self.log("Procesare completă.")

if __name__ == "__main__":
    app = TrafficSignApp()
    app.mainloop()