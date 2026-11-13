from ultralytics import YOLO

def train_my_yolo():
    print("Incarcare model YOLOv8 Nano...")
    
    model = YOLO('yolov8n.pt') 

    print("Incepere antrenare...")
    
    model.train(
        data='dataset_train_yolo.yaml',   # Fisierul de configurare pe care tocmai l-ai facut
        epochs=30,          # De cate ori trece prin tot setul de date
        imgsz=640,          # Redimensioneaza imaginile mari la 640x640 pt antrenare
        batch=8,           # Cate imagini ia odata (scade la 8 daca iei eroare de RAM)
        workers=2,          
        device='0'        # '0' daca ai GPU instalat cu CUDA
    )

if __name__ == '__main__':
    train_my_yolo()