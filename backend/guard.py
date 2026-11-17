import io
import json

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

ALLOWED_FORMATS = ("JPEG", "PNG", "BMP")
MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_PIXELS = 50_000_000
MIN_SIDE = 32
GATE_SIZE = (224, 224)            # (tinggi, lebar) input Image Type Gate (EfficientNetB0)
CLASSIFIER_SIZE = (380, 380)     # (tinggi, lebar) input EfficientNetB4


class InvalidImageError(ValueError):
    """Gambar gagal validasi. Pesan sudah jelas dan aman ditampilkan ke pengguna."""


# ---------------------------------------------------------------------------
# Utilitas internal
# ---------------------------------------------------------------------------

def _load_rgb(image_bytes: bytes) -> Image.Image:
    """Validasi byte gambar asli lalu decode ke PIL RGB (belum di-resize).

    Raise InvalidImageError bila gambar tidak valid atau rusak.
    """
    if not image_bytes:
        raise InvalidImageError("File gambar kosong.")
    if len(image_bytes) > MAX_FILE_SIZE:
        raise InvalidImageError("Ukuran gambar maksimal 10 MB.")
    try:
        with Image.open(io.BytesIO(image_bytes)) as im:
            if im.format not in ALLOWED_FORMATS:
                raise InvalidImageError(
                    "Format gambar tidak didukung. Gunakan JPG, PNG, atau BMP."
                )
            width, height = im.size
            if width * height > MAX_PIXELS or min(width, height) < MIN_SIDE:
                raise InvalidImageError(
                    f"Resolusi {width}x{height} tidak memenuhi batas "
                    f"(sisi minimal {MIN_SIDE} px, maksimal {MAX_PIXELS:,} piksel)."
                )
            # convert() memuat seluruh piksel; file terpotong/rusak gagal di sini
            return im.convert("RGB")
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise InvalidImageError("File bukan gambar valid atau rusak.") from exc


def _resize_to_array(im: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """Resize PIL Image ke (tinggi, lebar) lalu kembalikan float32 skala 0..255.

    Tidak melakukan pembagian /255 — model sudah punya Rescaling/Normalization
    internal di layer pertamanya.
    """
    resized = im.resize((size[1], size[0]), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


# ---------------------------------------------------------------------------
# GERDGuard
# ---------------------------------------------------------------------------

class GERDGuard:
    """Filter awal berbasis image-type classifier, terpisah dari model GERD.

    Dua jalur preprocessing **independen**, keduanya dari byte gambar asli:

    1. ``check_bytes``  →  RGB 224×224  untuk Image Type Gate (EfficientNetB0).
    2. ``prepare_classifier_image``  →  RGB 380×380  untuk EfficientNetB4.

    Kedua model sudah punya Rescaling/Normalization di dalamnya, sehingga
    input cukup float32 RGB 0..255.  **Jangan** /255 atau preprocess_input
    lagi di luar.
    """

    def __init__(self, gate_model_path: str, config_path: str):
        # ---- konfigurasi threshold yang sudah diverifikasi ----
        with open(config_path, encoding="utf-8") as f:
            self.cfg = json.load(f)
        size = self.cfg.get("input_size", [224, 224])
        self.img_size = (int(size[0]), int(size[1]))      # selalu (224, 224)
        self.threshold = float(self.cfg.get("non_endoscopy_threshold", 0.5))

        # ---- muat gate model ----
        self.gate = tf.keras.models.load_model(str(gate_model_path), compile=False)
        if tuple(self.gate.input_shape[1:]) != (*self.img_size, 3):
            raise RuntimeError(
                f"Input image-type gate tidak cocok: {self.gate.input_shape}"
            )
        if self.gate.output_shape[-1] != 1:
            raise RuntimeError("Gate harus memiliki satu output sigmoid.")

        # Kompilasi graph inferensi gate dengan @tf.function untuk eksekusi CPU cepat
        @tf.function(reduce_retracing=True)
        def _fast_gate_call(x: tf.Tensor) -> tf.Tensor:
            return self.gate(x, training=False)

        self._fast_gate_call = _fast_gate_call

    def warmup(self):
        """Warmup JIT graph trace saat startup."""
        dummy = tf.zeros((1, *self.img_size, 3), dtype=tf.float32)
        self._fast_gate_call(dummy)

    # ----- 1. Image Type Gate  (RGB 224×224) -----

    def check_bytes(self, image_bytes: bytes) -> tuple[dict, Image.Image | None]:
        """Jalankan Image Type Gate pada gambar.

        Return
        ------
        (gate_result, pil_image | None)
            - ``pil_image`` adalah gambar asli (RGB, ukuran asli, **bukan**
              hasil resize 224×224) yang bisa digunakan untuk preprocessing
              EfficientNetB4 selanjutnya.
            - Jika DITOLAK, ``pil_image`` = None.
        """
        try:
            pil_image = _load_rgb(image_bytes)
        except InvalidImageError as exc:
            return {"status": "DITOLAK", "alasan": str(exc)}, None

        # Resize ke 224×224 **hanya** untuk gate — array ini tidak boleh
        # dipakai sebagai input EfficientNetB4.
        gate_arr = _resize_to_array(pil_image, self.img_size)  # (224, 224, 3)

        gate_tensor = tf.convert_to_tensor(gate_arr[None, ...], dtype=tf.float32)
        score = float(np.asarray(self._fast_gate_call(gate_tensor)).reshape(-1)[0])
        score = float(np.clip(score, 0.0, 1.0))

        # Konfigurasi gate: kelas 0=endoscopy, kelas 1=non_endoscopy.
        if score >= self.threshold:
            return {
                "status": "DITOLAK",
                "alasan": "Gambar terdeteksi sebagai non-endoskopi oleh image-type gate.",
                "skor_non_endoscopy": round(score, 6),
                "ambang_non_endoscopy": self.threshold,
            }, None

        # LOLOS — kembalikan gambar asli (bukan gate_arr)
        return {
            "status": "LOLOS_GATE",
            "skor_non_endoscopy": round(score, 6),
            "ambang_non_endoscopy": self.threshold,
        }, pil_image

    # ----- 2. EfficientNetB4 Preprocessing  (RGB 380×380) -----

    def prepare_classifier_image(
        self,
        source: bytes | Image.Image,
    ) -> np.ndarray:
        """Siapkan input untuk EfficientNetB4 dari gambar asli.

        Parameters
        ----------
        source : bytes | PIL.Image.Image
            - ``bytes``  → decode ulang dari byte asli (jalur mandiri).
            - ``Image``  → PIL RGB yang sudah lolos ``check_bytes`` (hemat
              decode ulang, tetap ukuran asli).

        Returns
        -------
        np.ndarray
            float32 (380, 380, 3), RGB, skala 0..255.
            Tanpa pembagian /255 — EfficientNetB4 menggunakan preprocessing
            internal (Rescaling layer).

        Raises
        ------
        InvalidImageError
            Bila ``source`` berupa bytes dan gambar tidak valid.
        TypeError
            Bila ``source`` bukan bytes maupun PIL Image.
        """
        if isinstance(source, bytes):
            pil_image = _load_rgb(source)
        elif isinstance(source, Image.Image):
            # Pastikan sudah RGB; convert() aman dipanggil berulang
            pil_image = source if source.mode == "RGB" else source.convert("RGB")
        else:
            raise TypeError(
                f"source harus bytes atau PIL.Image.Image, bukan {type(source).__name__}"
            )

        return _resize_to_array(pil_image, CLASSIFIER_SIZE)  # (380, 380, 3)
