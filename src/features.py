from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class FeatureBundle:
    color_hist: np.ndarray
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray
    image_shape: Tuple[int, int]


def extract_hs_histogram(
    image_bgr: np.ndarray,
    mask: np.ndarray | None = None,
    h_bins: int = 30,
    s_bins: int = 32,
) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        mask,
        [h_bins, s_bins],
        [0, 180, 0, 256],
    )
    hist = hist.astype(np.float32).flatten()
    total = float(hist.sum())
    if total > 0.0:
        hist /= total
    return hist


def create_sift(
    nfeatures: int = 0,
    contrast_threshold: float = 0.04,
    edge_threshold: float = 10.0,
    sigma: float = 1.6,
) -> cv2.SIFT:
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("当前 OpenCV 不支持 SIFT，请安装带 SIFT 的版本。")
    return cv2.SIFT_create(
        nfeatures=nfeatures,
        contrastThreshold=contrast_threshold,
        edgeThreshold=edge_threshold,
        sigma=sigma,
    )


def extract_sift_descriptors(
    image_bgr: np.ndarray,
    mask: np.ndarray | None = None,
    nfeatures: int = 0,
    contrast_threshold: float = 0.04,
    edge_threshold: float = 10.0,
    sigma: float = 1.6,
) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    sift = create_sift(
        nfeatures=nfeatures,
        contrast_threshold=contrast_threshold,
        edge_threshold=edge_threshold,
        sigma=sigma,
    )
    keypoints, descriptors = sift.detectAndCompute(gray, mask)

    if keypoints is None:
        keypoints = []
    if descriptors is None:
        descriptors = np.zeros((0, 128), dtype=np.float32)
    else:
        descriptors = descriptors.astype(np.float32, copy=False)

    return list(keypoints), descriptors


def extract_feature_bundle(
    image_bgr: np.ndarray,
    foreground_mask: np.ndarray | None = None,
    h_bins: int = 30,
    s_bins: int = 32,
    sift_nfeatures: int = 0,
    sift_contrast_threshold: float = 0.04,
    sift_edge_threshold: float = 10.0,
    sift_sigma: float = 1.6,
) -> FeatureBundle:
    color_hist = extract_hs_histogram(
        image_bgr,
        mask=foreground_mask,
        h_bins=h_bins,
        s_bins=s_bins,
    )
    keypoints, descriptors = extract_sift_descriptors(
        image_bgr,
        mask=foreground_mask,
        nfeatures=sift_nfeatures,
        contrast_threshold=sift_contrast_threshold,
        edge_threshold=sift_edge_threshold,
        sigma=sift_sigma,
    )
    return FeatureBundle(
        color_hist=color_hist,
        keypoints=keypoints,
        descriptors=descriptors,
        image_shape=tuple(image_bgr.shape[:2]),
    )
