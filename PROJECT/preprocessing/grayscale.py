import numpy as np

def grayscale(image_rgb):
    """
    Konversi RGB ke grayscale:
    I = (R + G + B) / 3
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Input grayscale harus citra RGB (H, W, 3)")

    height, width, _ = image_rgb.shape
    gray = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            R, G, B = image_rgb[y, x]
            gray[y, x] = (float(R) + float(G) + float(B)) / 3.0

    return gray
