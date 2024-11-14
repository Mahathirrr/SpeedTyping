**SCRIPT VIDEO: KLASIFIKASI MALWARE MENGGUNAKAN SVM**
[Durasi Total: ~6 menit]

[0:00 - 0:45] **PENDAHULUAN**
"Halo semua! Dalam video ini, kita akan membahas implementasi Support Vector Machine untuk klasifikasi malware menggunakan dataset TUNADROMD.

Mengapa klasifikasi malware penting? 
Dengan meningkatnya penggunaan smartphone Android, ancaman malware juga semakin meningkat. Kita perlu sistem yang dapat membedakan aplikasi berbahaya (malware) dari aplikasi aman (goodware) secara akurat.

Dalam video ini, kita akan:
1. Menganalisis dataset TUNADROMD
2. Membandingkan tiga jenis kernel SVM
3. Menentukan model terbaik untuk deteksi malware

Dataset TUNADROMD berisi 4465 sampel dengan 241 fitur yang merepresentasikan karakteristik aplikasi Android, seperti permissions dan API calls."

[0:45 - 1:45] **EKSPLORASI DATA**
"Mari kita mulai dengan eksplorasi data. Pertama, import library yang diperlukan:

```python
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
```

Load dan periksa dataset kita:
```python
df = pd.read_csv('TUANDROMD.csv')
print(df.info())
print(df.describe())
```

Dataset kita memiliki karakteristik menarik:
1. Struktur Data:
   - 4465 sampel aplikasi Android
   - 241 fitur binary (0 atau 1)
   - 1 kolom label (malware/goodware)

2. Distribusi Kelas:
   - 79.87% malware
   - 20.13% goodware
   Kita memiliki kasus class imbalance yang perlu ditangani.

3. Tipe Fitur:
   - Android permissions (ACCESS_NETWORK_STATE, CAMERA, dll)
   - API calls (TelephonyManager, SmsManager, dll)
   - System operations"

[1:45 - 3:00] **PREPROCESSING**
"Preprocessing data sangat krusial untuk performa SVM. Ada tiga tahap penting:

1. Penanganan missing values menggunakan mode imputation:
```python
def handle_missing_values(df):
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df
```
Kita memilih mode imputation karena:
- Data bersifat binary (0/1)
- Mempertahankan distribusi nilai asli
- Cocok untuk categorical data

2. Feature selection - 50 fitur terpenting:
```python
def select_features(X, y, k=50):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    return X_selected, selected_features

X_selected, selected_features = select_features(X, y)
```
Feature selection penting karena:
- Mengurangi dimensi data
- Menghilangkan fitur tidak relevan
- Mempercepat training
- Mencegah overfitting

3. Standardisasi data:
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)
```
Standardisasi crucial untuk SVM karena:
- Menyamakan skala semua fitur
- Meningkatkan konvergensi
- Mencegah dominasi fitur tertentu"

[3:00 - 4:15] **IMPLEMENTASI MODEL**
"Sekarang kita implementasikan SVM dengan tiga kernel berbeda. Mari bahas karakteristik setiap kernel:

1. Linear Kernel:
   - Sederhana dan interpretable
   - Cocok untuk data linearly separable
   - Parameter utama: C (trade-off margin)

2. Polynomial Kernel:
   - Menangkap pola non-linear moderat
   - Fleksibel dengan parameter degree
   - Cocok untuk interaksi antar fitur

3. RBF Kernel:
   - Paling fleksibel untuk non-linearitas
   - Parameter gamma mengontrol kompleksitas
   - Best choice untuk pattern recognition

Konfigurasi untuk GridSearchCV:
```python
kernel_configs = {
    'linear': {
        'kernel': ['linear'],
        'C': [0.1, 1, 10]
    },
    'poly': {
        'kernel': ['poly'],
        'C': [0.1, 1, 10],
        'degree': [2, 3],
        'gamma': ['scale', 'auto']
    },
    'rbf': {
        'kernel': ['rbf'],
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto']
    }
}
```

Split data dengan stratified sampling:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
```"

[4:15 - 5:00] **HASIL DAN EVALUASI**
"Setelah training, kita dapatkan hasil menarik:

1. Linear Kernel:
   - Accuracy: 99.02%
   - Best Parameters: C=1
   - F1-score: 0.98

2. Polynomial Kernel:
   - Accuracy: 99.65%
   - Best Parameters: C=10, degree=3
   - F1-score: 0.99

3. RBF Kernel:
   - Accuracy: 99.79%
   - Best Parameters: C=10, gamma='scale'
   - F1-score: 1.00

RBF kernel unggul dalam semua metrik karena:
- Mampu menangkap pola kompleks
- Adaptif terhadap noise
- Balance antara flexibility dan generalization"

[5:00 - 5:30] **ANALISIS MENDALAM**
"Beberapa insight penting:
1. Semua kernel mencapai akurasi >99%
2. Feature selection sangat efektif
3. Class imbalance berhasil ditangani
4. RBF kernel konsisten terbaik"

[5:30 - 6:00] **KESIMPULAN**
"Kesimpulannya:
1. SVM dengan RBF kernel optimal untuk klasifikasi malware
2. Preprocessing dan feature selection crucial
3. Parameter optimal: C=10, gamma='scale'

Rekomendasi implementasi:
- Gunakan RBF kernel
- Pertahankan 50 fitur terpilih
- Monitor performa pada data baru

Terima kasih telah menyaksikan! Jangan lupa like dan subscribe untuk konten machine learning lainnya!"

[END]

Script ini memberikan:
1. Penjelasan teknis yang jelas
2. Alasan di balik setiap keputusan
3. Flow yang logis dan mudah diikuti
4. Balance antara teori dan praktik
