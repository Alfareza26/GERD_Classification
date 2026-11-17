"""GERD Classification & Explainability API (GERDGuard).

Arsitektur:
1. Image Type Gate (EfficientNetB0) -> RGB 224x224 (threshold verifikasi).
2. GERD Classifier (EfficientNetB4) -> RGB 380x380 (input gambar asli).
3. Grad-CAM Explainability -> heatmap & overlay pada layer top_conv.
"""

from __future__ import annotations

import base64
import io
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from gradcam import GradCAMError, generate_gradcam
from guard import GERDGuard, InvalidImageError

# ---------------------------------------------------------------------------
# Konfigurasi & Path
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "combined" / "model_gerd_efficientnetb4.keras"
GATE_MODEL_PATH = BASE_DIR / "model" / "gate" / "image_type_classifier.keras"
GATE_CONFIG_PATH = BASE_DIR / "model" / "gate" / "image_type_config.json"

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_MIME = {"image/jpeg", "image/png", "image/bmp"}
MODEL_NAME = "EfficientNetB4_GERD_Model"
CLASSIFIER_THRESHOLD = 0.50

# State aplikasi (dimuat sekali saat startup)
model: tf.keras.Model | None = None
guard: GERDGuard | None = None
lock = threading.Lock()


# ---------------------------------------------------------------------------
# Lifespan: Pemuatan Model Sekali Saat Startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, guard
    for path in (MODEL_PATH, GATE_MODEL_PATH, GATE_CONFIG_PATH):
        if not path.is_file():
            raise RuntimeError(f"Berkas wajib tidak ditemukan: {path}")

    # Muat model EfficientNetB4
    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    if tuple(model.input_shape[1:]) != (380, 380, 3):
        raise RuntimeError(
            f"Input shape EfficientNetB4 tidak cocok: {model.input_shape} (expected (None, 380, 380, 3))"
        )

    # Muat Image Type Gate & konfigurasinya
    guard = GERDGuard(GATE_MODEL_PATH, GATE_CONFIG_PATH)

    print("========================================")
    print("GERDGuard API siap.")
    print("Model GERD      :", MODEL_PATH)
    print("Image-type gate :", GATE_MODEL_PATH)
    print(f"Gate threshold  : {guard.threshold}")
    print(f"GERD threshold  : {CLASSIFIER_THRESHOLD}")
    print("========================================")

    yield

    guard = None
    model = None
    tf.keras.backend.clear_session()


# ---------------------------------------------------------------------------
# Inisialisasi FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="GERD API",
    description=(
        "API klasifikasi endoskopi GERD dengan Image Type Gate dan Grad-CAM visualizer. "
        "Eksperimental, bukan alat diagnosis mandiri."
    ),
    version="4.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Utilitas Encode Gambar ke Base64 (In-Memory, Tidak Disimpan ke Disk)
# ---------------------------------------------------------------------------

def _bgr_to_base64(bgr_array: np.ndarray, quality: int = 90) -> str:
    """Encode OpenCV BGR image ke data URI base64 JPEG in-memory."""
    success, buffer = cv2.imencode(".jpg", bgr_array, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise RuntimeError("Gagal meng-encode array gambar ke format JPEG.")
    encoded = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def _bytes_to_base64(data: bytes, mime_type: str = "image/jpeg") -> str:
    """Encode raw image bytes ke data URI base64."""
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


# ---------------------------------------------------------------------------
# Pipeline Prediksi & Grad-CAM
# ---------------------------------------------------------------------------

def run_prediction_pipeline(data: bytes, mime_type: str) -> dict:
    """Pipeline penuh prediksi endoskopi.

    Langkah-langkah:
    1. Validasi gambar & cek Image Type Gate (224x224).
    2. Jika DITOLAK -> hentikan proses, jangan jalankan EfficientNetB4 atau Grad-CAM.
    3. Jika LOLOS -> Preprocessing khusus EfficientNetB4 (380x380) dari gambar asli.
    4. Inference EfficientNetB4 (training=False, augmentasi nonaktif).
    5. Jalankan Grad-CAM berdasarkan kelas prediksi (target kelas GERD atau Normal).
    6. Susun respons lengkap dengan heatmap & overlay base64.
    """
    if model is None or guard is None:
        raise RuntimeError("Model atau filter belum siap.")

    # 1. Jalankan Image Type Gate
    gate_result, pil_image = guard.check_bytes(data)

    # Jika gambar rusak / decoding gagal, lempar InvalidImageError agar menghasilkan HTTP 400
    if gate_result["status"] == "DITOLAK" and pil_image is None:
        alasan = gate_result.get("alasan", "")
        if "rusak" in alasan.lower() or "bukan gambar valid" in alasan.lower() or "batas" in alasan.lower():
            raise InvalidImageError(alasan)

    # 2. Jika gambar ditolak oleh Gate (bukan endoskopi) -> stop di sini
    if gate_result["status"] == "DITOLAK":
        return {
            "status": "DITOLAK",
            "prediction": "DITOLAK",
            "predicted_class": None,
            "label": None,
            "probability_gerd": None,
            "probability_normal": None,
            "confidence": None,
            "gate_status": "DITOLAK",
            "gate_score": gate_result.get("skor_non_endoscopy"),
            "model_name": MODEL_NAME,
            "reason": gate_result.get("alasan", "Gambar terdeteksi bukan endoskopi."),
            "original_image": None,
            "heatmap_image": None,
            "overlay_image": None,
        }

    # 3. Preprocessing gambar asli ke RGB 380x380 (skala 0..255 float32)
    arr_b4 = guard.prepare_classifier_image(pil_image)
    batch = tf.convert_to_tensor(arr_b4[None, ...], dtype=tf.float32)

    # 4. Prediksi EfficientNetB4 (dengan lock thread-safety)
    with lock:
        pred_tensor = model(batch, training=False)
        p_gerd = float(np.asarray(pred_tensor).reshape(-1)[0])
    p_gerd = float(np.clip(p_gerd, 0.0, 1.0))
    p_normal = float(1.0 - p_gerd)

    predicted_class = "GERD" if p_gerd >= CLASSIFIER_THRESHOLD else "GERD_Normal"
    confidence = p_gerd if predicted_class == "GERD" else p_normal

    # 5. Jalankan Grad-CAM berdasarkan kelas prediksi
    target_class_gerd = (predicted_class == "GERD")
    try:
        with lock:
            cam_result = generate_gradcam(
                model=model,
                img_array=arr_b4,
                original_image=pil_image,
                target_class_gerd=target_class_gerd,
            )
    except Exception as exc:
        raise GradCAMError(f"Kegagalan komputasi Grad-CAM: {exc}") from exc

    # 6. Encode gambar ke base64 (in-memory)
    original_b64 = _bytes_to_base64(data, mime_type)
    heatmap_b64 = _bgr_to_base64(cam_result["heatmap"])
    overlay_b64 = _bgr_to_base64(cam_result["overlay"])

    return {
        "status": predicted_class,
        "prediction": predicted_class,
        "predicted_class": predicted_class,
        "label": predicted_class,
        "probability_gerd": round(p_gerd, 6),
        "probability_normal": round(p_normal, 6),
        "confidence": round(confidence, 6),
        "gate_status": gate_result.get("status", "LOLOS_GATE"),
        "gate_score": gate_result.get("skor_non_endoscopy"),
        "model_name": MODEL_NAME,
        "original_image": original_b64,
        "heatmap_image": heatmap_b64,
        "overlay_image": overlay_b64,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "GERD Classification & Explainability API aktif",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    ready = model is not None and guard is not None
    return {
        "status": "ok" if ready else "not_ready",
        "model_loaded": model is not None,
        "image_type_gate_loaded": guard is not None,
        "model_name": MODEL_NAME,
        "model_path": str(MODEL_PATH.relative_to(BASE_DIR)),
        "gate_model_path": str(GATE_MODEL_PATH.relative_to(BASE_DIR)),
        "gate_threshold": guard.threshold if guard is not None else None,
        "classifier_threshold": CLASSIFIER_THRESHOLD,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Validasi kesiapan model
    if model is None or guard is None:
        raise HTTPException(503, "Model atau filter belum siap.")

    # 2. Validasi MIME type
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in ALLOWED_MIME:
        await file.close()
        raise HTTPException(
            status_code=415,
            detail="Format gambar tidak didukung. Gunakan JPG, PNG, atau BMP.",
        )

    # 3. Baca bytes dengan batasan ukuran
    try:
        data = await file.read(MAX_FILE_SIZE + 1)
    finally:
        await file.close()

    if not data:
        raise HTTPException(status_code=400, detail="File gambar kosong.")
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Ukuran gambar maksimal 10 MB.")

    mime_for_encode = content_type if content_type in ALLOWED_MIME else "image/jpeg"

    # 4. Jalankan pipeline di thread pool
    try:
        result = await run_in_threadpool(run_prediction_pipeline, data, mime_for_encode)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except GradCAMError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        print("Prediction error:", repr(exc))
        raise HTTPException(
            status_code=500,
            detail=f"Terjadi kesalahan saat pemrosesan citra: {exc}",
        ) from exc

    return result
