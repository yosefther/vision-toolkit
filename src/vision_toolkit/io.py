from pathlib import Path 
from typing import Literal

import cv2 
import numpy as np 
_FLAGS = {
    "color" : cv2.IMREAD_COLOR,
    "grayscale" : cv2.IMREAD_GRAYSCALE,
    "unchanged" : cv2.IMREAD_UNCHANGED

}

def load_image(
        path:str | Path,
        mode: Literal["color" , "grayscale" , "unchanged"] = "color",
        rgb: bool = False
    ) -> np.ndarray:
    """Load an image from disk.

    Args:
        path: Path to the image file.
        mode: "color" (3 channels), "grayscale" (1 channel), or
            "unchanged" (keeps alpha channel and bit depth).
        rgb: If True, convert color images from OpenCV's BGR order to RGB.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file exists but cannot be decoded as an image.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"could not read the image  {str(path)}")

    image = cv2.imread(path,_FLAGS[mode])
    if image is None:
        raise ValueError(f"could not decode image (corrupt or unsupported format): {path}")

    if rgb and image.ndim == 3 :
        code = cv2.COLOR_BGRA2RGBA if image.shape[2] == 4 else cv2.COLOR_BGR2RGB
        image = cv2.cvtColor(image, code)
    return image 

def describe(image):
    height , width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    return {
        "width": width,
        "height": height,
        "channels": channels,
        "dtype": str(image.dtype),
        "min": image.min().item(),
        "max": image.max().item(),
    }