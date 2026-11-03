from pathlib import Path

import cv2
import numpy as np

from vision_toolkit.adjust import adjust_brightness, adjust_contrast
from vision_toolkit.io import describe, load_image

ASSETS = Path(__file__).resolve().parents[2] / "assets"

img = load_image(ASSETS / "eagle.jpg")  # BGR np.ndarray, dtype uint8
print(describe(img))

brighter = adjust_brightness(img, 50)
contrasted = adjust_contrast(img, 1.7)

# original | brighter | contrasted, side by side
comparison = np.hstack([img, brighter, contrasted])

cv2.namedWindow("Result", cv2.WINDOW_NORMAL)  # resizable window, image scales to fit
cv2.resizeWindow("Result", 1500, 300)
cv2.imshow("Result", comparison)
cv2.waitKey(0)
cv2.destroyAllWindows()
