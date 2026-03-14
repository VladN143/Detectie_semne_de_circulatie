import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt

# Importăm datele folosind funcția ta existentă
from preprocessing import load_data, DATA_DIR, NUM_CLASSES

#1.DEFINIREA MODELULUI
class TrafficSignNet(nn.Module):
    def __init__(self, num_classes):
        super(TrafficSignNet, self).__init__()
        # Input: 3 x 32 x 32

                # Strat 1: Conv (32 filtre) + ReLU + MaxPool
        self.strat1=nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Strat 2: Conv (64 filtre) + ReLU + MaxPool
        self.strat2=nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # Strat 3: Conv (64 filtre) + ReLU
        self.strat3=nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU()
        )
        
        # Strat Flatten + Dense
        # Trebuie să calculăm dimensiunea după convoluții.
        # 32x32 -> (conv1) -> 30x30 -> (pool1) -> 15x15
        # 15x15 -> (conv2) -> 13x13 -> (pool2) -> 6x6
        # 6x6   -> (conv3) -> 4x4
        # 64 canale * 4 * 4 = 1024 neuroni
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
    
    # CRUCIAL: PyTorch vrea canalele primele (N, C, H, W)
    X = np.transpose(X, (0, 3, 1, 2)) # Mutăm axa 3 pe poziția 1
    
    # Împărțire
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convertim în Tensori PyTorch
    train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).long())
    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test).long())
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    #3.CONFIGURARE ANTRENARE
    model = TrafficSignNet(NUM_CLASSES)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 15
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

        # === 2. FAZA DE VALIDARE (TESTARE) ===
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

    # --- 4. SALVARE ---
    # În PyTorch salvăm dicționarul de stări (state_dict), nu tot obiectul
    torch.save(model.state_dict(), 'traffic_sign_CNN.pth')
    print("Model salvat: traffic_sign_CNN.pth")
    
    # Plotting rapid
    # plt.plot(train_acc_history)
    # plt.title('Training Accuracy (PyTorch)')
    # plt.show()

if __name__ == "__main__":
    train_model()