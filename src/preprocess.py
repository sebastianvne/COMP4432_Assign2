from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


@dataclass
class PreprocessResult:
    image_bgr: np.ndarray
    original_shape: Tuple[int, int]


def load_image_bgr(image_path: Path | str) -> np.ndarray:
    image_path = Path(image_path)
    buffer = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    return image


def resize_longest_side(image_bgr: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_side:
        return image_bgr

    scale = max_side / float(longest_side)
    resized = cv2.resize(
        image_bgr,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    return resized


def preprocess_image(
    image_path: Path | str,
    max_side: int = 960,
) -> PreprocessResult:
    original = load_image_bgr(image_path)
    resized = resize_longest_side(original, max_side=max_side)
    return PreprocessResult(
        image_bgr=resized,
        original_shape=tuple(original.shape[:2]),
    )
