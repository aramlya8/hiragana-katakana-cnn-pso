import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

from preprocessing.pipeline import preprocess_pipeline
from evaluation.akurasi import evaluate_model # type: ignore

# =========================
# LOAD DATA TESTING
# =========================
def load_test_dataset(folder, label):
    X, y = [], []

    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img_bgr = cv2.imread(path)

        if img_bgr is None:
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # PREPROCESSING (SAMA DENGAN TRAINING)
        img = preprocess_pipeline(img_rgb)      # (28,28)
        img = img.reshape(28,28,1)

        X.append(img)
        y.append(label)

    return np.array(X), np.array(y)

X1, y1 = load_test_dataset("data/testing/hiragana", 0)
X2, y2 = load_test_dataset("data/testing/katakana", 1)

X_test = np.concatenate((X1, X2))
y_test = np.concatenate((y1, y2))

# =========================s
# LOAD MODEL HASIL PSO
# =========================
model = load_model("model/cnn_pso.keras")


# =========================
# EVALUASI
# =========================
evaluate_model(model, X_test, y_test)
