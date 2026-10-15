import numpy as np

def normalize(image):
    """
    Normalisasi piksel:
    I_norm = I / 255
    """
    if image.ndim != 2:
        raise ValueError("Normalisasi membutuhkan citra grayscale 2D")

    height, width = image.shape
    normalized = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            normalized[y, x] = image[y, x] / 255.0

    return normalized
