import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, learning_curve 
from sklearn.metrics import accuracy_score, classification_report

from preprocessing import load_data, preprocess_for_hog, DATA_DIR, NUM_CLASSES

def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    
    plt.figure()
    plt.title(title)
    plt.xlabel("Număr de exemple de antrenare")
    plt.ylabel("Acuratețe (Score)")

    print(f"Generare learning curve pentru {title}...")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='accuracy'
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Scor Antrenare (Train)")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Scor Validare (Test/CV)")

    plt.legend(loc="best")
    
    plt.savefig("grafic SVM.png")
    print("Graficul a fost salvat ca 'grafic SVM.png'")
    plt.show()

def train_svm_model():
    print("--- 1. Încărcare date pentru SVM ---")
    raw_images, raw_labels = load_data(DATA_DIR, NUM_CLASSES)
    
    if len(raw_images) == 0:
        print("Eroare: Nu s-au găsit imagini.")
        return

    print(f"Am încărcat {len(raw_images)} imagini.")

    print("--- 2. Extragere trăsături HOG (poate dura puțin)... ---")
    X_hog = preprocess_for_hog(raw_images)
    y = np.array(raw_labels)

    print(f"Dimensiune date HOG: {X_hog.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(X_hog, y, test_size=0.2, random_state=42)

    svm_clf = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)

   
    print("--- Generare Grafice (Learning Curves) ---")
    plot_learning_curve(svm_clf, "Curba de Învățare SVM (HOG)", X_train, y_train, cv=3, n_jobs=-1)

    print("--- 3. Antrenare SVM Finală... ---")
    svm_clf.fit(X_train, y_train)
    print("Antrenare finalizată!")

    print("--- 4. Evaluare Model ---")
    y_pred = svm_clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\nACURATEȚE FINALĂ TEST: {acc * 100:.2f}%")
    
    print("\nRaport Detaliat:")
    print(classification_report(y_test, y_pred))

    model_filename = 'traffic_sign_svm.pkl'
    joblib.dump(svm_clf, model_filename)
    print(f"Model salvat ca: {model_filename}")

if __name__ == "__main__":
    train_svm_model()