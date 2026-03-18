from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


@dataclass
class PreprocessResult:
    image_bgr: np.ndarray
    foreground_mask: np.ndarray
    rect: Tuple[int, int, int, int]
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


def build_center_rect(image_shape: Tuple[int, int], margin_ratio: float) -> Tuple[int, int, int, int]:
    height, width = image_shape[:2]
    x_margin = max(1, int(width * margin_ratio))
    y_margin = max(1, int(height * margin_ratio))

    rect_x = min(x_margin, max(width - 2, 0))
    rect_y = min(y_margin, max(height - 2, 0))
    rect_w = max(width - 2 * rect_x, 1)
    rect_h = max(height - 2 * rect_y, 1)
    return rect_x, rect_y, rect_w, rect_h


def apply_grabcut_center_prior(
    image_bgr: np.ndarray,
    margin_ratio: float = 0.1,
    iter_count: int = 5,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
    rect = build_center_rect(image_bgr.shape[:2], margin_ratio)
    mask = np.zeros(image_bgr.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(
            image_bgr,
            mask,
            rect,
            bgd_model,
            fgd_model,
            iter_count,
            cv2.GC_INIT_WITH_RECT,
        )
        foreground_mask = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
            255,
            0,
        ).astype(np.uint8)
    except cv2.error:
        foreground_mask = np.full(image_bgr.shape[:2], 255, dtype=np.uint8)

    if int(foreground_mask.sum()) == 0:
        foreground_mask = np.full(image_bgr.shape[:2], 255, dtype=np.uint8)

    foreground = cv2.bitwise_and(image_bgr, image_bgr, mask=foreground_mask)
    return foreground, foreground_mask, rect


def preprocess_image(
    image_path: Path | str,
    max_side: int = 960,
    grabcut_margin_ratio: float = 0.1,
    grabcut_iter_count: int = 5,
) -> PreprocessResult:
    original = load_image_bgr(image_path)
    resized = resize_longest_side(original, max_side=max_side)
    foreground, foreground_mask, rect = apply_grabcut_center_prior(
        resized,
        margin_ratio=grabcut_margin_ratio,
        iter_count=grabcut_iter_count,
    )
    return PreprocessResult(
        image_bgr=foreground,
        foreground_mask=foreground_mask,
        rect=rect,
        original_shape=tuple(original.shape[:2]),
    )
