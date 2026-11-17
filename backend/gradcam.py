"""Grad-CAM untuk model GERD EfficientNetB4.

Menghasilkan heatmap dan overlay visualisasi dari layer konvolusi terakhir
(``top_conv``) di sub-model ``efficientnetb4``.

Arsitektur model (verified):
    image (Input 380×380×3)
    → data_augmentation (Sequential: RandomFlip, RandomRotation, RandomZoom, RandomContrast)
    → efficientnetb4 (Functional, 476 layers, output 1792-d)
    → dropout (mixed_float16)
    → gerd_probability (Dense sigmoid, 1 neuron)

Layer target: ``efficientnetb4/top_conv`` → output shape (batch, 12, 12, 1792)
Alur downstream: ``top_bn`` → ``top_activation`` → ``avg_pool`` → ``dropout`` → ``gerd_probability``

Catatan optimasi performa:
- Sub-model feature extractor dan layer references dicache sekali (GradCAMExplainer).
- Operasi forward pass dan komputasi gradien dikompilasi dengan ``@tf.function``
  sehingga inferensi CPU berjalan ~30-40x lebih cepat (~0.8s vs ~35s).
- Komputasi skor prediksi dan Grad-CAM digabungkan dalam 1 pass (tanpa re-run forward pass).
"""

from __future__ import annotations

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------

_EFFNET_SUBMODEL = "efficientnetb4"
_TARGET_LAYER = "top_conv"          # Conv2D terakhir — (batch, 12, 12, 1792)

# Colormap untuk heatmap
_COLORMAP = cv2.COLORMAP_JET
_OVERLAY_ALPHA = 0.45               # Bobot heatmap pada overlay


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class GradCAMError(RuntimeError):
    """Kesalahan saat komputasi Grad-CAM."""


# ---------------------------------------------------------------------------
# Internal helpers & Layer Search
# ---------------------------------------------------------------------------

def _find_target_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    """Cari layer ``top_conv`` di dalam sub-model ``efficientnetb4``."""
    try:
        effnet = model.get_layer(_EFFNET_SUBMODEL)
    except ValueError as exc:
        raise GradCAMError(
            f"Sub-model '{_EFFNET_SUBMODEL}' tidak ditemukan dalam model."
        ) from exc

    try:
        layer = effnet.get_layer(_TARGET_LAYER)
    except ValueError as exc:
        raise GradCAMError(
            f"Layer '{_TARGET_LAYER}' tidak ditemukan dalam sub-model '{_EFFNET_SUBMODEL}'."
        ) from exc

    if "Conv" not in layer.__class__.__name__:
        raise GradCAMError(
            f"Layer '{_TARGET_LAYER}' bukan Conv layer: {layer.__class__.__name__}"
        )
    return layer


# ---------------------------------------------------------------------------
# GradCAMExplainer: Pre-built & Compiled Fast Pipeline
# ---------------------------------------------------------------------------

class GradCAMExplainer:
    """Explainer Grad-CAM teroptimasi dengan model cache dan @tf.function."""

    def __init__(self, model: tf.keras.Model):
        self.model = model
        try:
            effnet = model.get_layer(_EFFNET_SUBMODEL)
            self.top_conv = effnet.get_layer(_TARGET_LAYER)
            self.top_bn = effnet.get_layer("top_bn")
            self.top_act = effnet.get_layer("top_activation")
            self.avg_pool = effnet.get_layer("avg_pool")
            self.dropout = model.get_layer("dropout")
            self.dense = model.get_layer("gerd_probability")
            self.aug = model.get_layer("data_augmentation")
        except Exception as exc:
            raise GradCAMError(f"Struktur layer model tidak sesuai: {exc}") from exc

        # Feature extractor dibentuk SEKALI saja saat inisialisasi
        self.feat_model = tf.keras.Model(inputs=effnet.input, outputs=self.top_conv.output)

        # Kompilasi graph gradien dengan @tf.function
        @tf.function(reduce_retracing=True)
        def _fast_gradcam_step(img_tensor: tf.Tensor, is_gerd: tf.Tensor):
            with tf.GradientTape() as tape:
                aug_x = self.aug(img_tensor, training=False)
                conv_out = self.feat_model(aug_x, training=False)
                tape.watch(conv_out)

                h = self.top_bn(conv_out, training=False)
                h = self.top_act(h)
                h = self.avg_pool(h)
                h = self.dropout(h, training=False)
                preds = self.dense(h, training=False)

                p_gerd = tf.cast(preds[0, 0], tf.float32)
                target_score = tf.where(is_gerd, p_gerd, 1.0 - p_gerd)

            grads = tape.gradient(target_score, conv_out)
            grads = tf.cast(grads, tf.float32)
            conv_out_f32 = tf.cast(conv_out, tf.float32)

            weights = tf.reduce_mean(grads, axis=(1, 2))
            cam = tf.reduce_sum(conv_out_f32[0] * weights[0], axis=-1)
            cam = tf.nn.relu(cam)

            cam_max = tf.reduce_max(cam)
            cam_norm = tf.where(cam_max > 0.0, cam / cam_max, cam)

            return p_gerd, cam_norm, target_score

        self._fast_gradcam_step = _fast_gradcam_step

    def warmup(self):
        """Warmup JIT trace agar request pertama pengguna langsung cepat."""
        dummy = tf.zeros((1, 380, 380, 3), dtype=tf.float32)
        self._fast_gradcam_step(dummy, tf.constant(True))
        self._fast_gradcam_step(dummy, tf.constant(False))


_EXPLAINER_CACHE: dict[int, GradCAMExplainer] = {}


def get_or_create_explainer(model: tf.keras.Model) -> GradCAMExplainer:
    """Ambil instance GradCAMExplainer dari cache atau buat baru jika belum ada."""
    mid = id(model)
    if mid not in _EXPLAINER_CACHE:
        explainer = GradCAMExplainer(model)
        explainer.warmup()
        _EXPLAINER_CACHE[mid] = explainer
    return _EXPLAINER_CACHE[mid]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_gradcam(
    model: tf.keras.Model,
    img_array: np.ndarray,
    original_image: np.ndarray | Image.Image,
    target_class_gerd: bool = True,
    explainer: GradCAMExplainer | None = None,
) -> dict:
    """Hasilkan Grad-CAM heatmap dan overlay secara teroptimasi (~0.8s).

    Parameters
    ----------
    model : tf.keras.Model
        Model GERD EfficientNetB4 yang sudah dimuat.
    img_array : np.ndarray
        float32 (380, 380, 3), skala 0..255 — input preprocessed untuk model.
    original_image : np.ndarray | PIL.Image.Image
        Gambar asli (ukuran asli) untuk overlay.
    target_class_gerd : bool
        True  → visualisasi terhadap prediksi GERD (kelas 1).
        False → visualisasi terhadap prediksi GERD_Normal (kelas 0).
    explainer : GradCAMExplainer, optional
        Instance explainer yang sudah di-warmup (jika disediakan).

    Returns
    -------
    dict dengan kunci:
        - ``heatmap``   : np.ndarray uint8 (H_orig, W_orig, 3) BGR colormap.
        - ``overlay``   : np.ndarray uint8 (H_orig, W_orig, 3) BGR overlay.
        - ``cam_raw``   : np.ndarray float32 (12, 12), rentang [0, 1].
        - ``score``     : float, probabilitas kelas target.
        - ``p_gerd``    : float, P(GERD) langsung dari model.
        - ``target``    : str, "GERD" atau "GERD_Normal".
        - ``prediction``: str, label prediksi berdasarkan threshold 0.5.
    """
    if img_array is None or not isinstance(img_array, np.ndarray):
        raise ValueError("img_array harus numpy array, bukan None.")
    if img_array.shape != (380, 380, 3):
        raise ValueError(f"img_array harus (380, 380, 3), didapat {img_array.shape}")
    if img_array.dtype != np.float32:
        raise ValueError(f"img_array harus float32, didapat {img_array.dtype}")

    if isinstance(original_image, Image.Image):
        orig_rgb = np.array(
            original_image.convert("RGB") if original_image.mode != "RGB" else original_image,
            dtype=np.uint8,
        )
    elif isinstance(original_image, np.ndarray):
        orig_rgb = original_image.astype(np.uint8) if original_image.dtype != np.uint8 else original_image
    else:
        raise ValueError(
            f"original_image harus PIL Image atau numpy array, bukan {type(original_image).__name__}"
        )

    if orig_rgb.ndim != 3 or orig_rgb.shape[2] != 3:
        raise ValueError(f"original_image harus HWC RGB, didapat shape {orig_rgb.shape}")

    orig_h, orig_w = orig_rgb.shape[:2]

    # Ambil explainer teroptimasi
    exp = explainer if explainer is not None else get_or_create_explainer(model)

    img_tensor = tf.convert_to_tensor(img_array[None, ...], dtype=tf.float32)
    is_gerd_tf = tf.constant(bool(target_class_gerd))

    # Eksekusi step terkompilasi
    p_gerd_tf, cam_tf, score_tf = exp._fast_gradcam_step(img_tensor, is_gerd_tf)

    p_gerd = float(np.clip(p_gerd_tf.numpy(), 0.0, 1.0))
    target_score = float(score_tf.numpy())
    cam_raw = cam_tf.numpy().astype(np.float32)

    # Resize heatmap ke resolusi asli
    cam_resized = cv2.resize(cam_raw, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # Buat heatmap BGR
    cam_uint8 = np.uint8(255 * cam_resized)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, _COLORMAP)

    # Buat overlay BGR
    orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.addWeighted(orig_bgr, 1.0 - _OVERLAY_ALPHA, heatmap_bgr, _OVERLAY_ALPHA, 0)

    # Validasi output
    if heatmap_bgr.shape[:2] != (orig_h, orig_w):
        raise GradCAMError(
            f"Dimensi heatmap ({heatmap_bgr.shape[:2]}) tidak cocok dengan gambar asli ({orig_h}, {orig_w})."
        )
    if overlay_bgr.shape != orig_bgr.shape:
        raise GradCAMError(
            f"Dimensi overlay ({overlay_bgr.shape}) tidak cocok dengan gambar asli ({orig_bgr.shape})."
        )
    if heatmap_bgr.size == 0 or overlay_bgr.size == 0:
        raise GradCAMError("Heatmap atau overlay kosong.")

    prediction_label = "GERD" if p_gerd >= 0.5 else "GERD_Normal"
    target_label = "GERD" if target_class_gerd else "GERD_Normal"

    return {
        "heatmap": heatmap_bgr,
        "overlay": overlay_bgr,
        "cam_raw": cam_raw,
        "score": target_score,
        "p_gerd": p_gerd,
        "target": target_label,
        "prediction": prediction_label,
    }
