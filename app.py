from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Inisialisasi API
app = FastAPI(title="Berita Classifier API")

# Konfigurasi CORS (PENTING: Agar HTML lokal bisa mengakses API ini)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Di produksi, ganti dengan domain front-end spesifik
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model dan Vectorizer
try:
    model = joblib.load('model_klasifikasi_berita.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
except FileNotFoundError:
    raise RuntimeError("File .pkl tidak ditemukan. Jalankan main.py (training) dulu.")

# Inisialisasi Sastrawi (Preprocessing harus SAMA PERSIS dengan saat training)
stemmer = StemmerFactory().create_stemmer()
stopword_remover = StopWordRemoverFactory().create_stop_word_remover()

def cleaning_teks(teks):
    teks = str(teks).lower()
    teks = re.sub(r'http\S+|www\S+', '', teks)
    teks = re.sub(r'[^a-z\s]', '', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    teks = stopword_remover.remove(teks)
    teks = stemmer.stem(teks)
    return teks

# Skema Input (Payload)
class BeritaInput(BaseModel):
    teks: str

# Endpoint Prediksi
@app.post("/predict")
def predict_berita(data: BeritaInput):
    if not data.teks.strip():
        raise HTTPException(status_code=400, detail="Teks berita tidak boleh kosong")
    
    # 1. Preprocessing teks
    teks_bersih = cleaning_teks(data.teks)
    
    if not teks_bersih:
        return {"kategori": "Tidak diketahui (Teks tidak valid setelah dibersihkan)"}
    
    # 2. Ekstraksi Fitur (Transform, BUKAN fit_transform)
    vektor = vectorizer.transform([teks_bersih])
    
    # 3. Prediksi
    prediksi = model.predict(vektor)
    
    return {
        "teks_asli": data.teks,
        "kategori": prediksi[0].upper()
    }