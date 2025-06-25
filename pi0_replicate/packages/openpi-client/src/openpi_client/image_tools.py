import numpy as np
from PIL import Image


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Converts an image to uint8 if it is a float image.

    This is important for reducing the size of the image when sending it over the network.
    """
    if np.issubdtype(img.dtype, np.floating):
        img = np.clip(255 * img, 0, 255).astype(np.uint8)
    return img


def _to_pil_uint8(arr: np.ndarray) -> tuple[Image.Image, tuple[float, float]]:
    """Convert an HWC **or** CHW NumPy array to a PIL image.

    Returns the image and `(scale, bias)` such that
        restored = image_np.astype(float32) * scale + bias
    will approximately reconstruct the original floating-point values.  For
    integer inputs `(scale,bias) == (1.0, 0.0)`.
    """

    arr = np.asarray(arr)

    # Move channel-first → channel-last if necessary.
    if arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[-1] > 4:
        arr = np.moveaxis(arr, 0, -1)

    # Replicate single-channel → RGB so downstream callers always see 3-channel.
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    # Float → uint8 conversion for PIL.
    if np.issubdtype(arr.dtype, np.floating):
        # Heuristically map approximately [-1,1] or [0,1] to [0,255].  We fall
        # back to simple clipping otherwise.
        arr_min, arr_max = arr.min(), arr.max()
        if arr_min >= -1.01 and arr_max <= 1.01:
            arr_uint8 = ((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            scale, bias = 1.0 / 127.5, -1.0
        elif arr_min >= 0.0 and arr_max <= 1.01:
            arr_uint8 = (arr * 255.0).clip(0, 255).astype(np.uint8)
            scale, bias = 1.0 / 255.0, 0.0
        else:
            # Generic fallback: linear map [min,max] → [0,255]
            scale_val = (arr_max - arr_min) if arr_max != arr_min else 1.0
            arr_uint8 = ((arr - arr_min) / scale_val * 255.0).clip(0, 255).astype(np.uint8)
            scale, bias = scale_val / 255.0, arr_min
    else:
        arr_uint8 = arr.astype(np.uint8) if arr.dtype != np.uint8 else arr
        scale, bias = 1.0, 0.0

    return Image.fromarray(arr_uint8), (scale, bias)


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Resize a batch of images to `height×width` with zero-padding, supporting
    both uint8 and float32 inputs and CHW/HWC layouts.
    """

    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape
    flat_imgs = images.reshape(-1, *original_shape[-3:])

    restored_imgs = []
    for im in flat_imgs:
        pil_img, (scale, bias) = _to_pil_uint8(im)
        pil_resized = _resize_with_pad_pil(pil_img, height, width, method=method)
        arr = np.asarray(pil_resized).astype(np.float32) * scale + bias
        restored_imgs.append(arr)

    resized = np.stack(restored_imgs)
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
    width without distortion by padding with zeros.

    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image
