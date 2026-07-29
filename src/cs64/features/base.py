"""Base non-OCR feature extraction for figures."""

from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
from PIL import Image


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode PNG/JPEG bytes into an RGB uint8 array."""
    with Image.open(BytesIO(image_bytes)) as image:
        return np.asarray(image.convert("RGB"))


def compute_pixel_count_mp(image_rgb: np.ndarray) -> float:
    height, width = image_rgb.shape[:2]
    return float(width * height) / 1_000_000.0


def compute_blur_laplacian_var(image_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_whitespace_ratio(image_rgb: np.ndarray, threshold: int = 245) -> float:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    whitespace_mask = gray >= threshold
    return float(np.mean(whitespace_mask))
