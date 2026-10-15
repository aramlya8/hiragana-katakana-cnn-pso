import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical # pyright: ignore[reportMissingImports]
from tensorflow.keras.optimizers import Adam # pyright: ignore[reportMissingImports]

from preprocessing.pipeline import preprocess_pipeline
from cnn.model import build_model

# =========================
# LOAD DATA MANUAL
# =========================
def load_dataset(base_dir, label):
    X, y = [], []

    for file in os.listdir(base_dir):
        path = os.path.join(base_dir, file)
        img_bgr = cv2.imread(path)

        if img_bgr is None:
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # PREPROCESSING PIPELINE
        img = preprocess_pipeline(img_rgb)     # (28,28)

        # RESHAPE UNTUK CNN
        img = img.reshape(28, 28, 1)

        X.append(img)
        y.append(label)

    return np.array(X), np.array(y)

# =========================
# DATA TRAINING
# =========================
X_train_1, y_train_1 = load_dataset("data/train/hiragana", 0)
X_train_2, y_train_2 = load_dataset("data/train/katakana", 1)

X_train = np.concatenate((X_train_1, X_train_2))
y_train = np.concatenate((y_train_1, y_train_2))

# =========================
# BUILD & COMPILE CNN
# =========================
model = build_model()
model.compile(
    optimizer=Adam(),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# =========================
# TRAINING
# =========================
model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32
)

model.save("model/cnn_model.h5")
