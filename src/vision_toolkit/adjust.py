import numpy as np


def adjust_brightness(image: np.ndarray, amount: float) -> np.ndarray:
    """Add `amount` to every pixel (new = old + amount), clipped to 0-255."""
    out = image.astype(np.float32) + amount
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    """Scale pixels away from mid-gray (new = (old - 128) * factor + 128), clipped to 0-255."""
    out = (image.astype(np.float32) - 128) * factor + 128
    return np.clip(np.round(out), 0, 255).astype(np.uint8)
