import cv2
import numpy as np

def rotate(image, angle):
    """
    Rotasi citra berdasarkan sudut tertentu
    θ > 0  : berlawanan arah jarum jam
    θ < 0  : searah jarum jam
    """
    h, w = image.shape
    center = (w // 2, h // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

    return rotated_image
