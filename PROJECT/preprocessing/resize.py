import numpy as np

def resize(image, new_height=28, new_width=28):
    """
    Resize citra grayscale 2D menggunakan nearest neighbor:
    I'(x', y') = I(x * sx, y * sy)
    """
    image = np.array(image)

    if image.ndim != 2:
        raise ValueError(
            f"Resize gagal: input harus citra 2D, shape={image.shape}"
        )

    old_height, old_width = image.shape

    sx = old_width / new_width
    sy = old_height / new_height

    resized = np.zeros((new_height, new_width))

    for y_new in range(new_height):
        for x_new in range(new_width):
            x_old = int(x_new * sx)
            y_old = int(y_new * sy)

            x_old = min(x_old, old_width - 1)
            y_old = min(y_old, old_height - 1)

            resized[y_new, x_new] = image[y_old, x_old]

    return resized
