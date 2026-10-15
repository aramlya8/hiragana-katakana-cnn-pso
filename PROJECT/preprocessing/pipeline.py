import cv2
import numpy as np

from preprocessing.grayscale import grayscale
from preprocessing.resize import resize
from preprocessing.normalization import normalize
from preprocessing.noise_removal import mean_filter_3x3

def preprocess_pipeline(image_rgb):
    img = grayscale(image_rgb)
    img = resize(img, 28, 28)
    img = normalize(img)
    img = mean_filter_3x3(img)
    return img