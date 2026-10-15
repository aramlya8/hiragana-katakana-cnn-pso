import numpy as np

def mean_filter_3x3(image):
    """
    Noise removal dengan mean filter 3x3:
    I′(x,y)=91​∑I(x+i,y+j)
    """
    if image.ndim != 2:
        raise ValueError("Mean filter membutuhkan citra 2D")

    height, width = image.shape
    filtered = np.zeros((height, width))

    padded = np.pad(image, pad_width=1, mode='edge')

    for y in range(height):
        for x in range(width):
            kernel = padded[y:y+3, x:x+3]
            filtered[y, x] = np.mean(kernel)

    return filtered
