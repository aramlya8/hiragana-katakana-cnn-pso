from preprocessing.pipeline import preprocess_pipeline
import tensorflow as tf  # pyright: ignore[reportMissingImports]
import numpy as np       # pyright: ignore[reportMissingImports]

from cnn.model import build_model  # arsitektur CNN
from optimization.pso import PSO   

import cv2, os  # pyright: ignore[reportMissingImports]

# =========================
# LOAD DATA TRAINING
# =========================
def load_dataset(folder, label):
    X, y = [], []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = preprocess_pipeline(img_rgb)   # (28,28)
        img = img.reshape(28,28,1)            # input CNN

        X.append(img)
        y.append(label)

    return np.array(X), np.array(y)

X1, y1 = load_dataset("data/training/hiragana", 0)
X2, y2 = load_dataset("data/training/katakana", 1)

X_train = np.concatenate((X1, X2))
y_train = np.concatenate((y1, y2))

# =========================
# BUILD & TRAIN CNN (BACKPROP)
# =========================
model = build_model()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32
)

# =========================
# OPTIMASI PSO
# =========================
n_particles = 5
particles = [model.get_weights() for _ in range(n_particles)]

pso = PSO(n_particles=n_particles)
pso.initialize(particles)

fitness_scores = []

for i, w in enumerate(particles):
    model.set_weights(w)

    # Hitung loss sebagai fitness
    loss, _ = model.evaluate(X_train, y_train, verbose=0)
    fitness_scores.append(loss)

particles = pso.update(particles, fitness_scores)

# Set bobot terbaik (gbest)
model.set_weights(pso.gbest)
print("OPTIMASI PARTICLE SWARM OPTIMIZATION (PSO) SELESAI")

# =========================
# SAVE MODEL
# =========================
model.save("model/cnn_pso.keras")
