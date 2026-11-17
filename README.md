# Panduan Sistem GERD Image Classifier
Sistem Klasifikasi Citra Endoskopi Berbasis Deep Learning & Image-Type Gate Guard

---

## 1. Ikhtisar Sistem & Arsitektur

**GERD Image Classifier** adalah sistem pendukung keputusan klinis (*Clinical Decision Support System* / CDSS) berbasis kecerdasan buatan (*Deep Learning*) yang dirancang untuk menganalisis citra endoskopi saluran cerna atas (*Esophagogastroduodenoscopy* / EGD). Sistem ini membantu mendeteksi tanda-tanda erosi mukosa esofagus distal dan lesi pada Z-line yang berkaitan dengan GERD (*Gastroesophageal Reflux Disease*).

Sistem mengklasifikasikan citra ke dalam dua kategori:
- **`GERD`**: Esofagitis Erosif (terdapat mucosal breaks / erosi pada esofagus distal).
- **`GERD_Normal`**: Mukosa esofagus normal atau Non-Erosive Reflux Disease (NERD).

---

## 2. Stack Teknologi yang Digunakan

### Backend (`/backend`)
| Komponen | Teknologi | Peran |
|---|---|---|
| **Runtime** | Python 3.11 | Lingkungan eksekusi backend |
| **Web Framework** | FastAPI | Framework REST API asinkron berkinerja tinggi |
| **ASGI Server** | Uvicorn | Server HTTP/ASGI produksi dan pengembangan |
| **Deep Learning** | TensorFlow / Keras 2.21 | Pemuatan model `.keras` dan eksekusi inferensi |
| **Model Klasifikasi** | EfficientNetB0 (`model_gerd_combined.keras`) | Model CNN transfer learning hasil pelatihan gabungan |
| **Image-Type Gate** | EfficientNetB0 (`image_type_classifier.keras`) | Filter keamanan penyaring citra non-endoskopi |
| **Image Processing** | Pillow (PIL) | Validasi berkas, pembacaan EXIF, dan resize citra |
| **Komputasi Matriks** | NumPy | Manipulasi tensor dan operasi array |
| **Upload Parser** | python-multipart | Penanganan pengunggahan file multipart |

### Frontend (`/frontend`)
| Komponen | Teknologi | Peran |
|---|---|---|
| **Framework Web** | Next.js 15 (App Router) | Framework React modern untuk produksi |
| **UI Library** | React 19 | Library antarmuka deklaratif |
| **Bahasa** | TypeScript 5.6 | Menjamin keamanan tipe (*type safety*) kode |
| **Desain & Tema** | CSS Variables Modern (`globals.css`) | Tema dinamis Terang (*Light*) dan Gelap (*Dark Mode*) |
| **Penyimpanan Lokal** | Browser LocalStorage (`lib/store.ts`) | Penyimpanan riwayat pemeriksaan & pengaturan profil |
| **Proxy Gateway** | Next.js Rewrites (`next.config.mjs`) | Meneruskan `/api/*` ke backend tanpa kendala CORS |

---

## 3. Prasyarat Sistem (*Prerequisites*)

Sebelum memulai instalasi, pastikan perangkat Anda telah terpasang:
1. **Python 3.11 (64-bit)** ([Unduh Python](https://www.python.org/downloads/))
2. **Node.js (v18.18+ atau v20+) & npm** ([Unduh Node.js](https://nodejs.org/))
3. **Git** (opsional, untuk manajemen repositori)
4. **Sistem Operasi**: Windows 10/11, macOS, atau Linux

---

## 4. Panduan Instalasi & Penyiapan Pertama Kali

### Langkah A: Penyiapan Backend
1. Buka terminal (PowerShell / Command Prompt) dan masuk ke folder `backend`:
   ```powershell
   cd "c:\Users\User\Documents\Project\GERD Classification\backend"
   ```

2. Buat dan aktifkan virtual environment Python:
   ```powershell
   # Membuat virtual environment
   python -m venv .venv

   # Mengaktifkan di Windows (PowerShell):
   .\.venv\Scripts\Activate.ps1
   # Atau jika di Command Prompt (cmd):
   .\.venv\Scripts\activate.bat
   ```

3. Pasang seluruh pustaka dependensi backend:
   ```powershell
   pip install -r requirements.txt
   ```

4. Pastikan berkas model berikut tersedia pada folder `backend/model/`:
   - `model/combined/model_gerd_combined.keras` *(Model klasifikasi aktif)*
   - `model/gate/image_type_classifier.keras` *(Model gate filter citra)*
   - `model/gate/image_type_config.json` *(Konfigurasi ambang non-endoskopi)*

---

### Langkah B: Penyiapan Frontend
1. Buka terminal baru dan masuk ke folder `frontend`:
   ```powershell
   cd "c:\Users\User\Documents\Project\GERD Classification\frontend"
   ```

2. Pasang dependensi Node.js:
   ```powershell
   npm install
   ```

---

## 5. Cara Menjalankan Aplikasi (*Running the Application*)

Aplikasi dijalankan pada dua terminal terpisah:

### Terminal 1: Jalankan Backend API
```powershell
cd backend
.\.venv\Scripts\python.exe -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
*Tanda backend berhasil aktif:*
```text
Model GERD: ...\backend\model\combined\model_gerd_combined.keras
Image-type gate: ...\backend\model\gate\image_type_classifier.keras
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### Terminal 2: Jalankan Frontend Web
```powershell
cd "c:\Users\User\Documents\Project\GERD Classification\frontend"
npm run dev
```
*Tanda frontend berhasil aktif:*
```text
   ▲ Next.js 15.1.0
   - Local:        http://localhost:3000
```

```text
   powershell -ExecutionPolicy Bypass -File .\run-all.ps1
```

### Akses Melalui Peramban (*Browser*):
- 🌐 **Antarmuka Aplikasi Web**: [http://localhost:3000](http://localhost:3000)
- 📄 **Dokumentasi Interaktif API (Swagger UI)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- 🩺 **Pemeriksaan Kesiapan Model (*Health Check*)**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 6. Panduan Penggunaan Modul Aplikasi

### 1. Halaman Beranda (`/`)
- Memuat ringkasan sistem, informasi model EfficientNetB0, alur kerja pemeriksaan, serta tombol cepat **"Mulai Analisis Citra"**.

### 2. Modul Analisis (`/analisis`)
- **Unggah Citra Endoskopi:**
  - Klik area dropzone atau seret (*drag-and-drop*) berkas citra.
  - Didukung navigasi papan ketik (*Enter* / *Space*).
  - Format yang didukung: **JPG, JPEG, PNG, BMP** (maksimal 10 MB).
  - **Sampel Uji Cepat:** Tersedia tombol pintasan (*Sampel Z-Line*, *Sampel GERD*, *Sampel Normal*) untuk pengujian instan tanpa harus menyiapkan file lokal.
  - Pratinjau citra menampilkan nama berkas, ukuran format (KB / MB), serta tombol **Ganti citra** dan **Hapus**.
- **Informasi Pasien:**
  - Masukkan ID Pasien Anonim (misal: `PS-8821`), Usia, Jenis Kelamin, dan Keluhan Klinis (opsional).
- **Hasil Analisis (`ResultDialog`):**
  - Klik tombol **"Mulai analisis"**.
  - Jendela modal akan menampilkan label prediksi (`GERD` atau `Normal`), persentase keyakinan (*confidence meter*), probabilitas detail per kelas, status kepatuhan ambang batas profil, tombol cetak rekap medis, dan tombol **"Simpan ke riwayat"**.
- **Peringatan Guard:** Jika gambar bukan citra endoskopi (dokumen, foto luar, x-ray), sistem akan menampilkan kartu peringatan merah informatif: *Citra ditolak oleh sistem keamanan (Guard)*.

### 3. Riwayat Analisis (`/riwayat`)
- Menyimpan hasil pemeriksaan di penyimpanan lokal peramban (*localStorage*).
- Dilengkapi pencarian ID pasien, filter cepat (*Semua*, *GERD*, *Normal*), pagination, dan tombol **Ekspor CSV** untuk pelaporan data.

### 4. Pengaturan Profil (`/profil`)
- Mengatur identitas pengguna/dokter serta instansi.
- Mengatur **Ambang Batas Keyakinan (*Threshold*)** kustom ($50\% - 98\%$) dengan opsi preset:
  - `50%` (Sensitivitas tinggi)
  - `85%` (Rekomendasi standar)
  - `98%` (Spesifisitas maksimal)

---

## 7. Pengujian Model di RapidMiner Studio

Untuk menguji performa model (Confusion Matrix, Kurva ROC, AUC, F1-Score) di **RapidMiner Studio**:

1. Jalankan script batch evaluation bawaan:
   ```powershell
   cd backend
   python export_for_rapidminer.py --folder "path/ke/folder_citra_uji" --out "hasil_rapidminer.csv"
   ```
2. Buka **RapidMiner Studio** $\rightarrow$ Buat proses baru:
   - Tarik operator **`Read CSV`** (arahkan ke `hasil_rapidminer.csv`).
   - Tarik operator **`Set Role`** (pilih atribut `actual_label` $\rightarrow$ role `label`).
   - Tarik operator **`Set Role`** (pilih atribut `prediction` $\rightarrow$ role `prediction`).
   - Tarik operator **`Performance (Binominal Classification)`** $\rightarrow$ centang *accuracy*, *precision*, *recall*, *AUC*, *f_measure*.
3. Klik tombol **Run (▶️)** untuk melihat Confusion Matrix visual dan Kurva ROC.

---

## 8. Endpoint API Backend (FastAPI)

| Method | Endpoint | Deskripsi |
|---|---|---|
| `GET` | `/` | Informasi status API dan versi aktif. |
| `GET` | `/health` | Memeriksa status muat model GERD dan Image-Type Gate. |
| `POST` | `/predict` | Menerima berkas `multipart/form-data` (`file`) dan mengembalikan hasil klasifikasi. |

### Contoh Respons Sukses (`200 OK`):
```json
{
  "prediction": "GERD",
  "status": "GERD",
  "confidence": 0.941258,
  "probabilities": {
    "GERD_Normal": 0.058742,
    "GERD": 0.941258
  },
  "skor_gerd": 0.941258,
  "threshold": 0.5,
  "guard": true,
  "gate": {
    "status": "LOLOS_GATE",
    "skor_non_endoscopy": 0.000124,
    "ambang_non_endoscopy": 0.01
  },
  "reason": "Lolos filter tipe gambar.",
  "model": "EfficientNetB0_GERD_Combined",
  "disclaimer": "Hasil eksperimental, bukan diagnosis medis. Perlu interpretasi tenaga kesehatan.",
  "filename": "pasien_endoskopi.jpg"
}
```

### Contoh Respons Citra Ditolak (`422 Unprocessable Entity`):
```json
{
  "detail": "Gambar terdeteksi sebagai non-endoskopi oleh image-type gate."
}
```

---

## 9. Panduan Mengatasi Kendala (*Troubleshooting*)

1. **PowerShell Script Execution Error (`PSSecurityException`):**
   Jika aktivasi venv diblokir oleh sistem Windows PowerShell, jalankan perintah ini satu kali:
   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
   ```
2. **Port 8000 atau Port 3000 Terpakai (*Port already in use*):**
   ```powershell
   # Periksa PID yang memakai port
   netstat -ano | findstr :8000
   # Hentikan proses
   taskkill /PID <PID_NOMOR> /F
   ```
3. **Backend Tidak Terjangkau dari Frontend:**
   - Pastikan backend aktif di `http://127.0.0.1:8000`.
   - Next.js telah dikonfigurasi melalui `next.config.mjs` untuk mem-proxy panggilan `/api/*` secara otomatis ke `http://127.0.0.1:8000/*`.
4. **Format Berkas Tidak Didukung:**
   - Gunakan format **JPG**, **JPEG**, **PNG**, atau **BMP**. Berkas format lain seperti PDF, WEBP, atau DOCX akan ditolak secara otomatis untuk menjaga integritas model.

---

## 10. Batasan Tanggung Jawab Klinis (*Medical Disclaimer*)

> ⚠️ **PERINGATAN MEDIS:**
> Sistem ini dikembangkan sebagai alat bantu komputasi edukasi dan riset pendukung keputusan klinis (*Clinical Decision Support*). Sistem ini **hanya mengklasifikasikan pola citra medis, bukan menetapkan diagnosis medis final secara definitif**. Segala keputusan diagnostik dan intervensi terapi tetap menjadi wewenang dan tanggung jawab penuh dokter spesialis gastroenterologi / tenaga medis profesional.
