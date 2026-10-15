import cv2 # pyright: ignore[reportMissingImports]

def zoom(image, scale):
    """
    sx, sy > 1 : zoom in
    sx, sy < 1 : zoom out
    """
    h, w = image.shape
    zoomed = cv2.resize(image, None, fx=scale, fy=scale)

    zh, zw = zoomed.shape

    start_x = max((zw - w) // 2, 0)
    start_y = max((zh - h) // 2, 0)

    zoomed_cropped = zoomed[start_y:start_y+h, start_x:start_x+w]

    return zoomed_cropped
