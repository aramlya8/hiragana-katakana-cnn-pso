import cv2  # type: ignore
import numpy as np  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore

from preprocessing.grayscale import grayscale
from preprocessing.noise_removal import mean_filter_3x3
from preprocessing.normalization import normalize
from preprocessing.resize import resize

# GUI pilih file
import tkinter as tk
from tkinter import filedialog

# =========================
# LOAD MODEL
# =========================
model = load_model("model/cnn_pso.keras")

# =========================
# PILIH GAMBAR (GUI)
# =========================
root = tk.Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Pilih Gambar Uji",
    filetypes=[("Image files", "*.png *.jpg *.jpeg")]
)

if not image_path:
    raise ValueError("Tidak ada gambar yang dipilih")

# =========================
# LOAD GAMBAR ASLI
# =========================
img_bgr = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# =========================
# PREPROCESSING TAHAP DEMI TAHAP
# =========================
img_gray = grayscale(img_rgb)                 # grayscale
img_denoise = mean_filter_3x3(img_gray)       # noise removal
img_norm = normalize(img_denoise)             # normalisasi (0–1)
img_28 = resize(img_norm, 28, 28)              # resize

# =========================
# INPUT CNN
# =========================
img_cnn = img_28.reshape(28, 28, 1)
img_cnn = np.expand_dims(img_cnn, axis=0)

# =========================
# PREDIKSI
# =========================
y_hat = model.predict(img_cnn)[0][0]

prob_katakana = float(y_hat)
prob_hiragana = float(1 - y_hat)

persen_katakana = prob_katakana * 100
persen_hiragana = prob_hiragana * 100

kelas = "Hiragana" if y_hat < 0.5 else "Katakana"

# =========================
# TAMPILKAN HASIL DI TERMINAL
# =========================
if y_hat < 0.5:
    label = 0
    kelas = "Hiragana"
else:
    label = 1
    kelas = "Katakana"

print("\n=== HASIL KLASIFIKASI ===")
print("Prediksi kelas :", kelas)
print("Label     :", label) 
print("Hiragana  :", round(persen_hiragana, 2), "%")
print("Katakana  :", round(persen_katakana, 2), "%")

print("\n=== NILAI PREPROCESSING (CONTOH) ===")

print("\n[Grayscale] (3x3 pertama)")
print(img_gray[:3, :3])

print("\n[Noise Removal] (3x3 pertama)")
print(img_denoise[:3, :3])

print("\n[Normalisasi] (3x3 pertama)")
print(img_norm[:3, :3])

print("\n[Final 28x28] Statistik")
print("Min :", img_28.min())
print("Max :", img_28.max())
print("Mean:", img_28.mean())

# =========================
# VISUALISASI OPENCV
# =========================
def show_stage(title, img, scale=10):
    img_uint8 = (img * 255).astype(np.uint8)
    img_zoom = cv2.resize(
        img_uint8,
        (img_uint8.shape[1]*scale, img_uint8.shape[0]*scale),
        interpolation=cv2.INTER_NEAREST
    )
    cv2.imshow(title, img_zoom)

cv2.imshow("Citra Asli", img_bgr)
show_stage("Grayscale", img_gray)
show_stage("Noise Removal", img_denoise)
show_stage("Normalisasi", img_norm)
show_stage("Input CNN 28x28", img_28)

cv2.waitKey(0)
cv2.destroyAllWindows()
