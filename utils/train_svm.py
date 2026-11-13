import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import pandas as pd  

from preprocessing import load_data, preprocess_for_hog, DATA_DIR, NUM_CLASSES

def plot_grid_search_results(grid_search):
    """
    Generează grafice detaliate bazate pe rezultatele obținute de GridSearchCV.
    Salvează un grafic general cu bare și un heatmap dedicat kernelului RBF.
    """
    print("Se generează graficele pentru analiza hiperparametrilor din GridSearchCV...")
    
    results = pd.DataFrame(grid_search.cv_results_)
    
    def create_param_label(row):
        if row['param_kernel'] == 'rbf':
            return f"RBF (C={row['param_C']}, $\gamma$={row['param_gamma']})"
        else:
            return f"Linear (C={row['param_C']})"
            
    results['param_label'] = results.apply(create_param_label, axis=1)
    
    # Sortăm rezultatele descrescător după scorul mediu de testare
    results = results.sort_values(by='mean_test_score', ascending=True)
    
    plt.figure(figsize=(12, 6))
    bars = plt.barh(results['param_label'], results['mean_test_score'], 
                    xerr=results['std_test_score'], color='skyblue', edgecolor='black', alpha=0.8)
    

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.005, bar.get_y() + bar.get_height()/2, f'{width*100:.2f}%', 
                 va='center', ha='left', fontsize=9, weight='bold')
                 
    plt.axvline(x=grid_search.best_score_, color='red', linestyle='--', label=f'Cel mai bun scor ({grid_search.best_score_*100:.2f}%)')
    plt.xlim(0, 1.1)  # Lăsăm spațiu pentru textul procentual
    plt.title('Analiza Comparativă a Parametrilor SVM (GridSearchCV)', fontsize=14, pad=15)
    plt.xlabel('Acuratețe Medie (Cross-Validation Score)', fontsize=12)
    plt.ylabel('Combinație Hiperparametri', fontsize=12)
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    
    plt.savefig("gridsearch_svm_performante_globale.png", bbox_inches='tight')
    print("Graficul bar plot salvat ca 'gridsearch_svm_performante_globale.png'")
    plt.close()
    
    #Heatmap dedicat kernelului RBF 
    rbf_results = results[results['param_kernel'] == 'rbf'].copy()
    
    if not rbf_results.empty:
        # Pivotez tabelul pentru a obține o matrice strânsă C x Gamma
        # Forțez conversia tipurilor pentru a evita erori de indexare la plotare
        rbf_results['param_C'] = rbf_results['param_C'].astype(float)
        rbf_results['param_gamma'] = rbf_results['param_gamma'].astype(str)
        
        scores_matrix = rbf_results.pivot(index='param_C', columns='param_gamma', values='mean_test_score')
        
        plt.figure(figsize=(8, 6))
        # Generăm harta termică folosind seaborn
        sns.heatmap(scores_matrix, annot=True, fmt=".4f", cmap="YlOrBr", cbar=True,
                    linewidths=0.5, linecolor='gray', annot_kws={"size": 11, "weight": "bold"})
        
        plt.title('Optimizarea Kernelului RBF: Influența Parametrilor C și $\gamma$', fontsize=13, pad=15)
        plt.xlabel('Parametru $\gamma$ (Gamma)', fontsize=11)
        plt.ylabel('Parametru de Regularizare C', fontsize=11)
        
        plt.savefig("gridsearch_svm_heatmap_rbf.png", bbox_inches='tight')
        print("Graficul heatmap salvat ca 'gridsearch_svm_heatmap_rbf.png'")
        plt.close()


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
    plt.close()

def plot_confusion_matrix_svm(y_true, y_pred):
    print("Se generează matricea de confuzie globală pentru SVM...")
    clase_existente = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=clase_existente)
    
    plt.figure(figsize=(16, 12))
    sns.heatmap(cm, annot=False, cmap='Oranges', cbar=True) 
    
    plt.title('Matricea de Confuzie - SVM (HOG Optimizat)', fontsize=16)
    plt.xlabel('Clasa Prezisă', fontsize=14)
    plt.ylabel('Clasa Adevărată', fontsize=14)
    
    plt.savefig("confusion_matrix_SVM.png", bbox_inches='tight')
    print("Graficul a fost salvat ca 'confusion_matrix_SVM.png'")
    plt.close()

def plot_class_specific_confusion_matrices(y_true, y_pred, output_dir="matrici_clase_svm"):
    """
    Generează o matrice de confuzie binară (One-vs-All) pentru fiecare clasă
    și o salvează într-un folder specificat.
    """
    print(f"Se generează matricele de confuzie pentru fiecare clasă în folderul '{output_dir}'...")
    
    # Creăm folderul dacă nu există
    os.makedirs(output_dir, exist_ok=True)
    
    clase_existente = np.unique(y_true)
    
    for clasa in clase_existente:
   
        y_true_bin = (y_true == clasa).astype(int)
        y_pred_bin = (y_pred == clasa).astype(int)
        
        
        cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False,
                    xticklabels=[f'Nu e clasa {clasa}', f'Este clasa {clasa}'],
                    yticklabels=[f'Nu e clasa {clasa}', f'Este clasa {clasa}'])
        
        plt.title(f'Matrice de Confuzie - Clasa {clasa} (SVM)', fontsize=14)
        plt.xlabel('Predicție SVM', fontsize=12)
        plt.ylabel('Realitate', fontsize=12)
        
        
        file_path = os.path.join(output_dir, f"matrice_confuzie_clasa_{clasa}.png")
        plt.savefig(file_path, bbox_inches='tight')
        plt.close()

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
    
    X_train, X_test, y_train, y_test = train_test_split(X_hog, y, test_size=0.2, random_state=42, stratify=y)

    print("--- 3. Optimizare parametri SVM cu GridSearchCV ---")
    param_grid = [
        {'C': [1, 10, 50], 'kernel': ['rbf'], 'gamma': ['scale', 0.01]},
        {'C': [1, 10], 'kernel': ['linear']}
    ]

    base_svm = SVC(probability=True, class_weight='balanced', random_state=42)

    grid_search = GridSearchCV(estimator=base_svm, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2)

    print("Începe căutarea celor mai buni parametri. Așteaptă...")
    grid_search.fit(X_train, y_train)

    print("\n=========================================")
    print("🏆 CEL MAI BUN MODEL A FOST GĂSIT!")
    print(f"Cea mai bună acuratețe în antrenare: {grid_search.best_score_ * 100:.2f}%")
    print(f"Combinația câștigătoare de parametri: {grid_search.best_params_}")
    print("=========================================\n")

    plot_grid_search_results(grid_search)

    best_svm_clf = grid_search.best_estimator_

    print("--- Generare Grafic Learning Curve pentru modelul optim ---")
    plot_learning_curve(best_svm_clf, "Curba de Învățare SVM (Optimizat)", X_train, y_train, cv=3, n_jobs=-1)

    print("--- 4. Evaluare Model Pe Setul De Test ---")
    y_pred = best_svm_clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\nACURATEȚE FINALĂ TEST: {acc * 100:.2f}%")

    # ------------------------------------------------------------------
    print("\nRaport Detaliat (Precizie, Recall, F1-Score):")
    raport = classification_report(y_test, y_pred)
    print(raport)
    
    
    with open("raport_clasificare_svm.txt", "w") as f:
        f.write("Raport de Clasificare SVM (Precizie, Recall, F1-Score)\n\n")
        f.write(raport)
    print("Raportul a fost salvat ca 'raport_clasificare_svm.txt'\n")
    # ------------------------------------------------------------------
    
    print("\nRaport Detaliat:")
    print(classification_report(y_test, y_pred))
    
    
    plot_confusion_matrix_svm(y_test, y_pred)
    
    
    plot_class_specific_confusion_matrices(y_test, y_pred)

    model_filename = 'traffic_sign_svm.pkl'
    joblib.dump(best_svm_clf, model_filename)
    print(f"\nModelul optim a fost salvat ca: {model_filename}")

if __name__ == "__main__":
    train_svm_model()