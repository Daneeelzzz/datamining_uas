import pandas as pd
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ==========================================
# 1. LOAD & FILTER DATASET (Dataset sh1zuka)
# ==========================================
print("[1] Memuat dan menyaring dataset sh1zuka...")

# Ganti 'indonesia_news_2025.csv' dengan nama file asli yang Anda unduh
try:
    df_raw = pd.read_csv('final_merge_dataset.csv') 
except FileNotFoundError:
    print("[ERROR] File CSV tidak ditemukan. Pastikan nama dan lokasi file benar.")
    exit()

# Ambil kolom yang dibutuhkan (Content -> teks, tag1 -> label)
df = df_raw[['Content', 'tag1']].copy()
df.rename(columns={'Content': 'teks', 'tag1': 'label'}, inplace=True)

# Hapus baris kosong (Missing Values)
df = df.dropna()

# FILTER: Ambil hanya 5 kategori/tag dengan jumlah berita terbanyak
top_5_kategori = df['label'].value_counts().nlargest(5).index
df = df[df['label'].isin(top_5_kategori)]

print(f"Distribusi 5 Kategori Teratas:\n{df['label'].value_counts()}\n")

# [OPSIONAL] Potong dataset ke 1.500 baris acak untuk mempercepat proses Sastrawi saat uji coba
# Jika Anda ingin melatih seluruh data, beri tanda komentar (#) pada dua baris di bawah ini
if len(df) > 1500:
    df = df.sample(1500, random_state=42)
    print(f"[INFO] Dataset dipotong menjadi {len(df)} baris untuk uji coba komputasi.\n")

# ==========================================
# 2. PREPROCESSING TEKS (NLP)
# ==========================================
print("[2] Memulai preprocessing teks (Sastrawi membutuhkan waktu)...")
factory_stemmer = StemmerFactory()
stemmer = factory_stemmer.create_stemmer()

factory_stopword = StopWordRemoverFactory()
stopword_remover = factory_stopword.create_stop_word_remover()

def cleaning_teks(teks):
    teks = str(teks).lower()                      # Case folding
    teks = re.sub(r'[^a-z\s]', '', teks)          # Hapus angka dan tanda baca
    teks = re.sub(r'\s+', ' ', teks).strip()      # Hapus spasi berlebih
    teks = stopword_remover.remove(teks)          # Stopword removal
    teks = stemmer.stem(teks)                     # Stemming
    return teks

df['teks_bersih'] = df['teks'].apply(cleaning_teks)

# ==========================================
# 3. PEMBAGIAN DATA (SPLIT TRAIN & TEST)
# ==========================================
print("[3] Membagi data latih dan data uji...")
X = df['teks_bersih']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 4. EKSTRAKSI FITUR (TF-IDF)
# ==========================================
print("[4] Melakukan ekstraksi fitur TF-IDF...")
vectorizer = TfidfVectorizer(max_features=3000) 

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"    -> Dimensi Matriks Data Latih: {X_train_vec.shape}")

# ==========================================
# 5. PEMODELAN RANDOM FOREST
# ==========================================
print("[5] Melatih model Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_vec, y_train)

# ==========================================
# 6. EVALUASI DAN HASIL
# ==========================================
print("[6] Menguji model...\n")
y_pred = rf_model.predict(X_test_vec)

print("="*50)
print(f"AKURASI MODEL AKHIR : {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("="*50)
print("\n=== Laporan Klasifikasi (Confusion Matrix Metrics) ===")
print(classification_report(y_test, y_pred))

# ==========================================
# 7. EXPORT MODEL 
# ==========================================
print("\n[7] Menyimpan model untuk API Backend...")
joblib.dump(rf_model, 'model_klasifikasi_berita.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("[SUCCESS] Pipeline selesai. File .pkl berhasil dibuat!")