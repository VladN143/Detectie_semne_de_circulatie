

# Detecția Automată a Semnelor de Circulație în Imagini Color ale Scenelor de Trafic

Acest proiect reprezintă lucrarea mea de licență în cadrul Facultății de Electronică, Telecomunicații și Tehnologia Informației (Universitatea Națională de Știință și Tehnologie POLITEHNICA BUCUREȘTI).

Scopul principal este dezvoltarea unui sistem software rapid și robust pentru localizarea și clasificarea indicatoarelor rutiere, destinat sistemelor autonome de asistență pentru șoferi (ADAS). Sistemul abordează provocări din mediul real, cum ar fi variațiile de lumină, condițiile meteo nefavorabile și estomparea prin mișcare (motion blur).

##  Arhitectura Sistemului (Design Hibrid)

Proiectul folosește o arhitectură hibridă împărțită în două stadii distincte:

1. **Stadiul 1: Localizare Spațială Precisă**
* Utilizează algoritmul **YOLOv8 (Nano)** într-o abordare „One-Stage”.


* Analizează întreaga scenă de trafic dintr-o singură trecere și izolează instantaneu regiunile de interes (ROI), generând casetele de încadrare și decupând automat imaginile pentru clasificare.




2. **Stadiul 2: Clasificare Semantică Fină**
* Decupajele obținute sunt redimensionate la $32\times32$ pixeli și trecute printr-un clasificator pentru decizia finală.


* Au fost implementate și comparate două abordări: o rețea neuronală convoluțională (**CNN / ResNet-18**) și o metodă clasică bazată pe descriptori de formă (**HOG + SVM**).





##  Setul de Date și Preprocesarea

* **Volum date:** Modelul a fost antrenat pe un set de date format din 3.742 de imagini color (împărțite 80% pentru antrenare și 20% pentru validare).


* **Gestionarea dezechilibrului claselor:** Setul a prezentat un dezechilibru masiv (ex. clasa „Cedează Trecerea” avea 241 imagini, în timp ce „Atenție animale” avea doar 6 imagini). Pentru a rezolva problema, au fost utilizate tehnici de **Eșantionare Ponderată** (alocarea de greutăți mai mari claselor mici) și **Augmentare de Date** (rotații, modificări de contrast și luminozitate).



##  Abordări Comparative și Performanțe

Proiectul a comparat o metodă clasică (SVM) cu una bazată pe Deep Learning (CNN):

* **Modelul CNN (ResNet-18 adaptat):**
* Utilizează extragerea automată a trăsăturilor, minimizarea funcției de pierdere Cross-Entropy prin optimizatorul Adam și tehnica Dropout pentru a preveni overfitting-ul.


* A obținut performanțe superioare, cu o **Acuratețe de validare de 97.50%**, o **Precizie de 97.0%** și un **Scor F1 de 95.0%**.


* Tratează excelent diferențierea detaliilor interioare fine ale semnelor.




* **Modelul SVM + HOG:**
* Utilizează descriptori HOG pentru analiza geometrică a contururilor și un clasificator SVM cu nucleu RBF (C=50) optimizat prin GridSearchCV.


* A obținut o **Precizie de 86.0%** și un **Scor F1 de 85.0%**.


* A demonstrat probleme de varianță (100% pe antrenare, dar 88.40% pe validare) și dificultăți în a distinge semnele cu forme geometrice similare, pierzând detaliul cifrelor.





**Concluzie performanțe:** Rețeaua neuronală CNN câștigă detașat, fiind cu ~10% mai performantă la toate capitolele și demonstrând o capacitate superioară de generalizare.

##  Interfața Grafică (GUI)

Pentru a oferi un mediu vizual interactiv, a fost dezvoltată o aplicație software dedicată:

* **Tehnologii GUI:** Python, CustomTkinter, OpenCV, PyTorch.


* **Funcționalitate:** Automatizare completă – utilizatorul încarcă imaginea de trafic, iar sistemul hibrid o detectează și o clasifică în doar câteva milisecunde. Interfața permite inclusiv afișarea diferențelor de predicție între modelul CNN și cel SVM, alături de scorurile de încredere (probabilitatea) și graficele de performanță.



##  Direcții Viitoare

* Optimizarea modelului pentru dispozitive hardware auto (embedded).


* Adaptarea sistemului pentru procesarea unui flux video continuu (real-time video stream).
