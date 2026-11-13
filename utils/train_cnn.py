import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torchvision import transforms
from torch.utils.data import Dataset
import os
from sklearn.metrics import multilabel_confusion_matrix
from torchvision import models
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, classification_report

# Importăm datele folosind funcția ta existentă
from preprocessing import load_data, DATA_DIR, NUM_CLASSES
from app import CLASSES

#1.DEFINIREA MODELULUI
# class TrafficSignNet(nn.Module):
#     def __init__(self, num_classes):
#         super(TrafficSignNet, self).__init__()
#         # Input: 3 x 32 x 32

#                 # Strat 1: Conv (32 filtre) + ReLU + MaxPool
#         self.strat1=nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Dropout(0.1),
#         )
        
#         # Strat 2: Conv (64 filtre) + ReLU + MaxPool
#         self.strat2=nn.Sequential(
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Dropout(0.1)
#         )
        
#         # Strat 3: Conv (64 filtre) + ReLU
#         self.strat3=nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Dropout(0.1)
#         )
        
#         # Strat Flatten + Dense
#         # Trebuie să calculăm dimensiunea după convoluții.
#         # 32x32 -> (conv1) -> 30x30 -> (pool1) -> 15x15
#         # 15x15 -> (conv2) -> 13x13 -> (pool2) -> 6x6
#         # 6x6   -> (conv3) -> 4x4
#         # 64 canale * 4 * 4 = 1024 neuroni
#         self.dense=nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(64 * 4 * 4, 64),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(64, num_classes)
#         )
#     def forward(self, x): 
#         x1=self.strat1(x)
#         x2=self.strat2(x1)
#         x3=self.strat3(x2)
#         x4=self.dense(x3)
#         return x4


#==================================================

# 1. Definirea arhitecturii CNN ResNet-18 Adaptat
class TrafficSignNet(nn.Module):
    def __init__(self, num_classes):
        super(TrafficSignNet, self).__init__()
        
        # Încărcăm arhitectura ResNet18 (fără ponderi pre-antrenate, o învățăm de la zero)
        self.resnet = models.resnet18(weights=None)
        
        # ADAPTARE PENTRU IMAGINI 32x32 
        # ResNet original așteaptă poze de 224x224. Pentru poze de 32x32, 
        # trebuie să micșorăm filtrul primului strat și să scoatem MaxPool-ul inițial
        # ca să nu pierdem informația instantaneu.
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity() 
        
        # Schimbăm ultimul strat (care original scoate 1000 de clase) 
        # să scoată exact numărul nostru de clase (43)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x): 
        
        return self.resnet(x)
    
#==================================================
    
# Această clasă ne permite să aplicăm transformări pe loc, în timpul antrenării
class TrafficSignDataset(Dataset):
    def __init__(self, tensors_x, tensors_y, transform=None):
        self.x = tensors_x
        self.y = tensors_y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = self.x[idx]
        label = self.y[idx]
        
        if self.transform:
            # Transformările PyTorch lucrează perfect pe tensori
            image = self.transform(image)
            
        return image, label
    
def plot_history(train_acc, val_acc, train_loss, val_loss):

    #Generează graficele comparative la finalul antrenării.

    epochs_range = range(len(train_acc))

    plt.figure(figsize=(12, 4))
    
        # Grafic Acuratețe
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

        # Grafic Pierdere (Loss)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.savefig("grafic CNN.png")
    print("Graficul a fost salvat ca 'grafic CNN.png'")
    plt.show()

def plot_confusion_matrix_custom(y_true, y_pred):
    print("Se generează matricea de confuzie...")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(16, 12))
    
    # Folosesc seaborn pentru a face un heatmap frumos
    # annot=False pentru că la 43 de clase ar deveni prea aglomerat textul în căsuțe
    sns.heatmap(cm, annot=False, cmap='Blues', cbar=True)
    
    plt.title('Matricea de Confuzie - PyTorch CNN', fontsize=16)
    plt.xlabel('Clasa Prezisă (Ce a crezut modelul că este)', fontsize=14)
    plt.ylabel('Clasa Adevărată (Ce era de fapt)', fontsize=14)
    
    plt.savefig("confusion_matrix_CNN.png", bbox_inches='tight')
    print("Graficul a fost salvat ca 'confusion_matrix_CNN.png'")
    plt.show()

def plot_per_class_confusion_matrices(y_true, y_pred, num_classes):
    print("\nSe generează cele 43 de matrici de confuzie individuale...")
    
    # Creez folderul dacă nu există deja
    folder_name = "Matrici_Per_Clasa"
    os.makedirs(folder_name, exist_ok=True)
    
    # multilabel_confusion_matrix generează un array de matrici 2x2 pentru fiecare clasă
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    
    for i in range(num_classes):
        # Extrag matricea 2x2 pentru clasa i
        matrix_2x2 = mcm[i]
        
        plt.figure(figsize=(6, 5))
        
        # Etichetele pentru axele matricei 2x2
        labels = [f'Nu e clasa {i}', f'Este clasa {i}']
        
        #  Heatmap-ul cu numerele afișate (annot=True)
        sns.heatmap(matrix_2x2, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=labels, yticklabels=labels)
        
        plt.title(f'Matrice de Confuzie - Clasa {i}', fontsize=14)
        plt.xlabel('Predicție CNN', fontsize=12)
        plt.ylabel('Realitate', fontsize=12)
        
        file_path = os.path.join(folder_name, f'matrice_clasa_{i:02d}.png')
        plt.savefig(file_path, bbox_inches='tight')
        plt.close() 
        
    print(f"Toate cele {num_classes} grafice au fost salvate în folderul '{folder_name}'!")


def train_model():
    #2.PREGĂTIRE DATE 
    print("--- Încărcare date ---")
    raw_images, raw_labels = load_data(DATA_DIR, NUM_CLASSES)
    
    # Normalizare (0-1)
    X = raw_images.astype("float32") / 255.0
    y = raw_labels

    # Calcul ponderi
    print("Se calculează ponderile pentru clasele dezechilibrate...")
    clase_unice = np.unique(y)
    ponderi = compute_class_weight(class_weight='balanced', classes=clase_unice, y=y)
    
    class_weights_tensor = torch.tensor(ponderi, dtype=torch.float)
    
    print("Ponderi calculate!")
    
    # PyTorch vrea canalele primele (N, C, H, W)
    X = np.transpose(X, (0, 3, 1, 2)) # Mutăm axa 3 pe poziția 1
    
    # Împărțire
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 🚀 DEFINIM TRANSFORMĂRILE (Data Augmentation)
    train_transform = transforms.Compose([
        # Rotim maxim 10 grade în loc de 15 (să nu amețim rețeaua)
        transforms.RandomRotation(10), 
        # AM SCOS `RandomAffine(translate=...)` pentru că tăia marginile semnelor
        # Lăsăm doar modificarea de lumină care e super utilă
        transforms.ColorJitter(brightness=0.2, contrast=0.2)
    ])
    
    # Nu aplic augmentare pe setul de testare! Acela trebuie să rămână curat.
    
    # Creăm Tensorii de bază...
    x_train_tensor = torch.from_numpy(X_train)
    y_train_tensor = torch.from_numpy(y_train).long()
    x_test_tensor = torch.from_numpy(X_test)
    y_test_tensor = torch.from_numpy(y_test).long()

    train_data = TrafficSignDataset(x_train_tensor, y_train_tensor, transform=train_transform)
    test_data = TrafficSignDataset(x_test_tensor, y_test_tensor, transform=None) 

   
    print("Se configurează WeightedRandomSampler pentru oversampling...")
    
    # 1. Număr câte imagini avem în fiecare clasă în setul de antrenare
    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    
    # 2. Calculez ponderea (invers proporțional cu numărul de imagini)
    weight = 1. / class_sample_count
    
    # 3. Atribui fiecărei imagini din setul de antrenare ponderea clasei din care face parte
    samples_weight = np.array([weight[t] for t in y_train])
    samples_weight = torch.from_numpy(samples_weight).double()
    
    # 4. Creez Sampler-ul
    sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)
    

    # Când folosești sampler, TREBUIE să scoți shuffle=True (sampler-ul le amestecă oricum)
    train_loader = DataLoader(train_data, batch_size=64, sampler=sampler)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    #3.CONFIGURARE ANTRENARE
    model = TrafficSignNet(NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 40
    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []
    
    print("--- Începe antrenarea (PyTorch) ---")
    for epoch in range(epochs):
        model.train() # Punem modelul în mod antrenare (activează Dropout)
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad() # Resetăm gradienții
            
            outputs = model(inputs) # Forward pass
            loss = criterion(outputs, labels) # Calculăm eroarea
            
            loss.backward() # Backward pass (calculăm gradienții)
            optimizer.step() # Actualizăm greutățile
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = correct / total

        #2. FAZA DE VALIDARE
        model.eval() # Oprim Dropout-ul
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad(): # Nu calculăm gradienți (economisim memorie și timp)
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        epoch_val_loss = val_loss / len(test_loader)
        epoch_val_acc = correct_val / total_val
        
        # Salvăm metricile în liste (înmulțim cu 100 pentru acuratețe ca să arate ca în TF)
        train_loss_hist.append(epoch_train_loss)
        train_acc_hist.append(epoch_train_acc * 100)
        val_loss_hist.append(epoch_val_loss)
        val_acc_hist.append(epoch_val_acc * 100)
        
        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%")
    
    # Afișare Grafice Identice cu TF
    # Împărțim la 100 înapoi doar pentru grafic, ca să arate axa Y de la 0.0 la 1.0, cum o aveai tu
    plot_history([x/100 for x in train_acc_hist], [x/100 for x in val_acc_hist], 
                 train_loss_hist, val_loss_hist)
    
    model.eval() # Ne asigurăm că e în mod de testare
    all_preds = []
    all_labels = []
    
    print("--- Se evaluează setul de test pentru Matricea de Confuzie ---")
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            # Adăugăm predicțiile și etichetele reale în liste
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ------------------------------------------------------------

    print("\n--- Raport de Clasificare (Precizie, Recall, F1-Score) ---")
    
    raport = classification_report(all_labels, all_preds)
    print(raport)
    with open("raport_clasificare.txt", "w") as f:
        f.write("Raport de Clasificare (Precizie, Recall, F1-Score)\n\n")
        f.write(raport)
    print("Raportul a fost salvat ca 'raport_clasificare.txt'\n")
    # ------------------------------------------------------------
            
    # Apelăm funcția de desenare
    plot_confusion_matrix_custom(all_labels, all_preds)

    plot_per_class_confusion_matrices(all_labels, all_preds, NUM_CLASSES)

    #4. SALVARE
    # În PyTorch salvăm dicționarul de stări (state_dict), nu tot obiectul
    torch.save(model.state_dict(), 'traffic_sign_CNN.pth')
    print("Model salvat: traffic_sign_CNN.pth")

    # torch.save(model.state_dict(), 'traffic_sign_CNN_resnet18.pth')
    # print("Model salvat: traffic_sign_CNN_resnet18.pth")
    
    # Plotting rapid
    # plt.plot(train_acc_history)
    # plt.title('Training Accuracy (PyTorch)')
    # plt.show()

if __name__ == "__main__":
    train_model()