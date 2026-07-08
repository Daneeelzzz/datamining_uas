import pandas as pd
import numpy as np
import re
import time
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib

# ==========================================
# 1. LOAD & FILTER DATASET
# ==========================================
print("=" * 60)
print("  KLASIFIKASI BERITA INDONESIA - RF vs SVM  ")
print("=" * 60)
print("\n[1] Memuat dan menyaring dataset...")

CSV_FILE = 'final_merge_dataset.csv'

try:
    df_raw = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    print(f"\n[ERROR] File '{CSV_FILE}' tidak ditemukan!")
    print("  -> Download dataset dari:")
    print("     https://www.kaggle.com/datasets/sh1zuka/indonesia-news-dataset-2024")
    print(f"  -> Taruh file CSV di folder yang sama dengan main.py ini")
    exit()

print(f"  -> Dataset berhasil dimuat: {df_raw.shape[0]} baris, {df_raw.shape[1]} kolom")

# Validasi kolom
required_cols = ['Content', 'tag1']
missing_cols = [c for c in required_cols if c not in df_raw.columns]
if missing_cols:
    print(f"\n[ERROR] Kolom tidak ditemukan: {missing_cols}")
    print(f"  -> Kolom yang ada: {list(df_raw.columns)}")
    exit()

# Ambil kolom yang dibutuhkan
df = df_raw[['Content', 'tag1']].copy()
df.rename(columns={'Content': 'teks', 'tag1': 'label'}, inplace=True)

# Hapus baris kosong
sebelum = len(df)
df = df.dropna()
print(f"  -> Baris valid setelah dropna: {len(df)} (dihapus {sebelum - len(df)} baris)")

# Normalisasi label: lowercase + strip spasi
df['label'] = df['label'].str.lower().str.strip()

# [UPGRADE] Gabungkan 'prabowo' dan 'prabowo subianto' -> 'prabowo/prabowo subianto'
# Karena konten berita keduanya sangat mirip (semantik sama)
df['label'] = df['label'].replace({
    'prabowo': 'prabowo/prabowo subianto',
    'prabowo subianto': 'prabowo/prabowo subianto'
})
print(f"  -> Label 'prabowo' + 'prabowo subianto' digabung menjadi 1 kelas")

# Filter top-5 kategori terbanyak (setelah penggabungan & normalisasi)
top_5_kategori = df['label'].value_counts().nlargest(5).index
df = df[df['label'].isin(top_5_kategori)].reset_index(drop=True)

print(f"\n  Distribusi 5 Kategori (setelah normalisasi & merge):")
vc = df['label'].value_counts()
for label, count in vc.items():
    bar = '#' * (count * 30 // vc.max())
    print(f"  {label:<30} {count:>5}  {bar}")

# [UPGRADE] Naikkan sample ke 3000 baris untuk akurasi lebih baik
MAX_ROWS = 3000
if len(df) > MAX_ROWS:
    df = df.sample(MAX_ROWS, random_state=42).reset_index(drop=True)
    print(f"\n  [INFO] Dataset di-sample menjadi {MAX_ROWS} baris.")
    print(f"         (Set MAX_ROWS = None di main.py untuk latih seluruh data)")

print(f"\n  Total data untuk training: {len(df)} baris")

# ==========================================
# 2. PREPROCESSING TEKS (NLP)
# ==========================================
print("\n[2] Inisialisasi tools Sastrawi...")
factory_stemmer = StemmerFactory()
stemmer = factory_stemmer.create_stemmer()
factory_stopword = StopWordRemoverFactory()
stopword_remover = factory_stopword.create_stop_word_remover()
print("  -> Stemmer & Stopword Remover: OK")

def cleaning_teks(teks):
    teks = str(teks).lower()
    teks = re.sub(r'http\S+|www\S+', '', teks)   # Hapus URL
    teks = re.sub(r'[^a-z\s]', '', teks)          # Hapus angka & tanda baca
    teks = re.sub(r'\s+', ' ', teks).strip()      # Hapus spasi berlebih
    teks = stopword_remover.remove(teks)           # Stopword removal
    teks = stemmer.stem(teks)                      # Stemming
    return teks

print(f"\n  Memproses {len(df)} teks (Sastrawi butuh waktu ~20-30 detik)...")
start_time = time.time()

hasil = []
for i, teks in enumerate(df['teks'], 1):
    hasil.append(cleaning_teks(teks))
    if i % 200 == 0 or i == len(df):
        elapsed = time.time() - start_time
        pct = i / len(df) * 100
        eta = (elapsed / i) * (len(df) - i)
        print(f"  [{pct:5.1f}%] {i}/{len(df)} | waktu: {elapsed:.0f}s | ETA: {eta:.0f}s")

df['teks_bersih'] = hasil
print(f"\n  -> Preprocessing selesai dalam {time.time() - start_time:.1f} detik")

# Hapus teks yang jadi kosong setelah preprocessing
kosong = df['teks_bersih'].str.strip().eq('').sum()
if kosong > 0:
    print(f"  [WARN] {kosong} teks kosong setelah preprocessing, dihapus.")
    df = df[df['teks_bersih'].str.strip() != ''].reset_index(drop=True)

# ==========================================
# 3. SPLIT TRAIN & TEST
# ==========================================
print("\n[3] Membagi data latih dan data uji (80:20)...")
X = df['teks_bersih']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  -> Data Latih : {len(X_train)} baris")
print(f"  -> Data Uji   : {len(X_test)} baris")

# ==========================================
# 4. EKSTRAKSI FITUR (TF-IDF)
# ==========================================
print("\n[4] Ekstraksi fitur TF-IDF (Unigram + Bigram)...")
vectorizer = TfidfVectorizer(
    max_features=5000,    # Lebih banyak fitur karena data lebih banyak
    ngram_range=(1, 2),   # Unigram + Bigram
    min_df=2,
    sublinear_tf=True     # Normalisasi log untuk teks panjang
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)
print(f"  -> Dimensi Matriks Latih : {X_train_vec.shape}")
print(f"  -> Dimensi Matriks Uji   : {X_test_vec.shape}")

# ==========================================
# 5A. TRAINING RANDOM FOREST
# ==========================================
print("\n[5A] Melatih Random Forest Classifier...")
start_rf = time.time()
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
rf_model.fit(X_train_vec, y_train)
waktu_rf = time.time() - start_rf
print(f"  -> Training RF selesai dalam {waktu_rf:.1f} detik")

print("  -> Cross-Validation 5-fold RF...")
cv_rf = cross_val_score(rf_model, X_train_vec, y_train, cv=5, scoring='accuracy', n_jobs=-1)
print(f"  -> CV Mean : {cv_rf.mean():.4f} +/- {cv_rf.std():.4f}")

# ==========================================
# 5B. TRAINING SVM (LinearSVC)
# ==========================================
print("\n[5B] Melatih SVM (LinearSVC) sebagai pembanding...")
start_svm = time.time()
svm_model = LinearSVC(
    max_iter=2000,
    random_state=42,
    class_weight='balanced'
)
svm_model.fit(X_train_vec, y_train)
waktu_svm = time.time() - start_svm
print(f"  -> Training SVM selesai dalam {waktu_svm:.1f} detik")

print("  -> Cross-Validation 5-fold SVM...")
cv_svm = cross_val_score(svm_model, X_train_vec, y_train, cv=5, scoring='accuracy', n_jobs=-1)
print(f"  -> CV Mean : {cv_svm.mean():.4f} +/- {cv_svm.std():.4f}")

# ==========================================
# 6. EVALUASI DAN PERBANDINGAN
# ==========================================
print("\n[6] Evaluasi dan perbandingan model...\n")

y_pred_rf  = rf_model.predict(X_test_vec)
y_pred_svm = svm_model.predict(X_test_vec)

acc_rf  = accuracy_score(y_test, y_pred_rf)
acc_svm = accuracy_score(y_test, y_pred_svm)

print("=" * 60)
print(f"  AKURASI RANDOM FOREST : {acc_rf  * 100:.2f}%  (CV: {cv_rf.mean()*100:.2f}%)")
print(f"  AKURASI SVM           : {acc_svm * 100:.2f}%  (CV: {cv_svm.mean()*100:.2f}%)")
winner = "Random Forest" if acc_rf >= acc_svm else "SVM"
print(f"  PEMENANG              : >> {winner} <<")
print("=" * 60)

print("\n=== Laporan Klasifikasi - Random Forest ===")
print(classification_report(y_test, y_pred_rf))

print("\n=== Laporan Klasifikasi - SVM ===")
print(classification_report(y_test, y_pred_svm))

# ==========================================
# VISUALISASI LENGKAP (4 panel)
# ==========================================
print("\n  Membuat visualisasi...")
classes = rf_model.classes_

fig = plt.figure(figsize=(20, 14))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# --- Panel 1: Confusion Matrix RF ---
ax1 = fig.add_subplot(gs[0, 0])
cm_rf = confusion_matrix(y_test, y_pred_rf, labels=classes)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=classes)
disp1.plot(ax=ax1, cmap='Blues', colorbar=False)
ax1.set_title(f'Confusion Matrix - Random Forest\nAkurasi: {acc_rf*100:.2f}%', fontsize=12, fontweight='bold')
plt.setp(ax1.get_xticklabels(), rotation=25, ha='right', fontsize=8)
plt.setp(ax1.get_yticklabels(), fontsize=8)

# --- Panel 2: Confusion Matrix SVM ---
ax2 = fig.add_subplot(gs[0, 1])
cm_svm = confusion_matrix(y_test, y_pred_svm, labels=classes)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_svm, display_labels=classes)
disp2.plot(ax=ax2, cmap='Oranges', colorbar=False)
ax2.set_title(f'Confusion Matrix - SVM (LinearSVC)\nAkurasi: {acc_svm*100:.2f}%', fontsize=12, fontweight='bold')
plt.setp(ax2.get_xticklabels(), rotation=25, ha='right', fontsize=8)
plt.setp(ax2.get_yticklabels(), fontsize=8)

# --- Panel 3: Feature Importance RF ---
ax3 = fig.add_subplot(gs[1, 0])
feature_names = vectorizer.get_feature_names_out()
importances   = rf_model.feature_importances_
top_idx       = np.argsort(importances)[::-1][:20]
top_features  = [feature_names[i] for i in top_idx]
top_scores    = [importances[i]   for i in top_idx]
ax3.barh(top_features[::-1], top_scores[::-1], color='steelblue', edgecolor='white')
ax3.set_title('Top-20 Kata Paling Berpengaruh\n(Random Forest Feature Importance)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Importance Score')
ax3.tick_params(axis='y', labelsize=8)
ax3.grid(axis='x', alpha=0.3)

# --- Panel 4: Perbandingan Akurasi RF vs SVM ---
ax4 = fig.add_subplot(gs[1, 1])
models    = ['Random Forest', 'SVM (LinearSVC)']
test_acc  = [acc_rf * 100, acc_svm * 100]
cv_acc    = [cv_rf.mean() * 100, cv_svm.mean() * 100]
cv_std    = [cv_rf.std() * 100, cv_svm.std() * 100]

x      = np.arange(len(models))
width  = 0.35
bars1  = ax4.bar(x - width/2, test_acc, width, label='Test Accuracy', color=['#4472C4', '#ED7D31'], alpha=0.85)
bars2  = ax4.bar(x + width/2, cv_acc,   width, label='CV Accuracy (5-fold)',
                  color=['#4472C4', '#ED7D31'], alpha=0.5,
                  yerr=cv_std, capsize=5)

ax4.set_title('Perbandingan Akurasi Model\nRandom Forest vs SVM', fontsize=12, fontweight='bold')
ax4.set_ylabel('Akurasi (%)')
ax4.set_xticks(x)
ax4.set_xticklabels(models, fontsize=10)
ax4.set_ylim(70, 100)
ax4.legend(fontsize=9)
ax4.grid(axis='y', alpha=0.3)
for bar in bars1:
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

fig.suptitle('Klasifikasi Berita Indonesia - Evaluasi Lengkap\n(Random Forest vs SVM | TF-IDF + Sastrawi)',
             fontsize=14, fontweight='bold', y=1.01)

plt.savefig('hasil_evaluasi.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> Grafik evaluasi disimpan: hasil_evaluasi.png")

# ==========================================
# 7. EXPORT MODEL TERBAIK
# ==========================================
print("\n[7] Menyimpan model terbaik...")

best_model      = rf_model  if acc_rf >= acc_svm else svm_model
best_model_name = "Random Forest" if acc_rf >= acc_svm else "SVM"

joblib.dump(best_model, 'model_klasifikasi_berita.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("\n" + "=" * 60)
print("  [SUCCESS] Pipeline selesai!")
print(f"  Model terbaik disimpan : {best_model_name}")
print(f"  Akurasi test           : {max(acc_rf, acc_svm)*100:.2f}%")
print("  File yang dihasilkan:")
print("   - model_klasifikasi_berita.pkl  (model terbaik)")
print("   - tfidf_vectorizer.pkl")
print("   - hasil_evaluasi.png           (4 panel visualisasi)")
print("=" * 60)