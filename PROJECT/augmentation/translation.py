import cv2
import numpy as np

def translate(image, tx, ty):
    """
    tx > 0 : geser kanan
    ty < 0 : geser bawah
    """
    h, w = image.shape

    translation_matrix = np.float32([
        [1, 0, tx],
        [0, 1, ty]
    ])

    translated_image = cv2.warpAffine(image, translation_matrix, (w, h))
    return translated_image
