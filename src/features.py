from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import cv2
import numpy as np

try:
    import cudasift
except Exception as exc:  # pragma: no cover - optional dependency
    cudasift = None
    _CUDASIFT_IMPORT_ERROR = exc
else:
    _CUDASIFT_IMPORT_ERROR = None


@dataclass
class FeatureBundle:
    color_hist: np.ndarray
    keypoints: Sequence[cv2.KeyPoint] | np.ndarray
    descriptors: np.ndarray
    image_shape: Tuple[int, int]


def extract_hs_histogram(
    image_bgr: np.ndarray,
    h_bins: int = 30,
    s_bins: int = 32,
) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
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


def resolve_sift_backend(backend: str) -> str:
    normalized = backend.lower()
    if normalized == "auto":
        return "cudasift" if cudasift is not None else "opencv"
    if normalized == "cudasift" and cudasift is None:
        detail = f" 原始错误: {_CUDASIFT_IMPORT_ERROR}" if _CUDASIFT_IMPORT_ERROR is not None else ""
        raise RuntimeError(f"请求使用 cudasift，但当前环境不可用。{detail}")
    if normalized not in {"opencv", "cudasift"}:
        raise ValueError(f"不支持的 SIFT 后端: {backend}")
    return normalized


def extract_sift_descriptors_opencv(
    image_bgr: np.ndarray,
    nfeatures: int = 0,
    contrast_threshold: float = 0.04,
    edge_threshold: float = 10.0,
    sigma: float = 1.6,
) -> Tuple[list[cv2.KeyPoint], np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    sift = create_sift(
        nfeatures=nfeatures,
        contrast_threshold=contrast_threshold,
        edge_threshold=edge_threshold,
        sigma=sigma,
    )
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if keypoints is None:
        keypoints = []
    if descriptors is None:
        descriptors = np.zeros((0, 128), dtype=np.float32)
    else:
        descriptors = descriptors.astype(np.float32, copy=False)

    return list(keypoints), descriptors


def extract_sift_descriptors_cudasift(
    image_bgr: np.ndarray,
    nfeatures: int = 0,
    contrast_threshold: float = 0.04,
    edge_threshold: float = 10.0,
    sigma: float = 1.6,
    cudasift_max_points: int = 32768,
    cudasift_num_octaves: int = 5,
    cudasift_lowest_scale: float = 0.0,
    cudasift_upscale: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    del edge_threshold  # CudaSift 不提供与 OpenCV 完全等价的 edge threshold 参数。

    if cudasift is None:
        detail = f" 原始错误: {_CUDASIFT_IMPORT_ERROR}" if _CUDASIFT_IMPORT_ERROR is not None else ""
        raise RuntimeError(f"cudasift 不可用。{detail}")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    max_points = max(int(cudasift_max_points), int(nfeatures) if nfeatures > 0 else 0, 1)
    sift_data = cudasift.PySiftData(max_points)

    # CudaSift 的阈值量纲与 OpenCV 不同，这里做保守映射，便于复用现有参数。
    thresh = float(max(1.0, contrast_threshold * 100.0))
    cudasift.ExtractKeypoints(
        gray,
        sift_data,
        numOctaves=int(cudasift_num_octaves),
        initBlur=float(max(0.0, sigma)),
        thresh=thresh,
        lowestScale=float(max(0.0, cudasift_lowest_scale)),
        upScale=bool(cudasift_upscale),
    )
    keypoint_df, descriptors = sift_data.to_data_frame()
    if len(keypoint_df) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 128), dtype=np.float32)

    keypoints = keypoint_df[["xpos", "ypos"]].to_numpy(dtype=np.float32, copy=True)
    descriptors = np.asarray(descriptors, dtype=np.float32)

    if nfeatures > 0 and len(keypoints) > nfeatures:
        scores = keypoint_df["sharpness"].to_numpy(dtype=np.float32, copy=False)
        keep = np.argsort(scores)[-nfeatures:]
        keypoints = keypoints[keep]
        descriptors = descriptors[keep]

    return keypoints.reshape(-1, 2), descriptors.reshape(-1, 128)


def extract_sift_descriptors(
    image_bgr: np.ndarray,
    nfeatures: int = 0,
    contrast_threshold: float = 0.04,
    edge_threshold: float = 10.0,
    sigma: float = 1.6,
    backend: str = "auto",
    cudasift_max_points: int = 32768,
    cudasift_num_octaves: int = 5,
    cudasift_lowest_scale: float = 0.0,
    cudasift_upscale: bool = False,
) -> Tuple[Sequence[cv2.KeyPoint] | np.ndarray, np.ndarray]:
    resolved_backend = resolve_sift_backend(backend)
    if resolved_backend == "cudasift":
        return extract_sift_descriptors_cudasift(
            image_bgr=image_bgr,
            nfeatures=nfeatures,
            contrast_threshold=contrast_threshold,
            edge_threshold=edge_threshold,
            sigma=sigma,
            cudasift_max_points=cudasift_max_points,
            cudasift_num_octaves=cudasift_num_octaves,
            cudasift_lowest_scale=cudasift_lowest_scale,
            cudasift_upscale=cudasift_upscale,
        )
    return extract_sift_descriptors_opencv(
        image_bgr=image_bgr,
        nfeatures=nfeatures,
        contrast_threshold=contrast_threshold,
        edge_threshold=edge_threshold,
        sigma=sigma,
    )


def extract_feature_bundle(
    image_bgr: np.ndarray,
    h_bins: int = 30,
    s_bins: int = 32,
    sift_nfeatures: int = 0,
    sift_contrast_threshold: float = 0.04,
    sift_edge_threshold: float = 10.0,
    sift_sigma: float = 1.6,
    sift_backend: str = "auto",
    cudasift_max_points: int = 32768,
    cudasift_num_octaves: int = 5,
    cudasift_lowest_scale: float = 0.0,
    cudasift_upscale: bool = False,
) -> FeatureBundle:
    color_hist = extract_hs_histogram(
        image_bgr,
        h_bins=h_bins,
        s_bins=s_bins,
    )
    keypoints, descriptors = extract_sift_descriptors(
        image_bgr,
        nfeatures=sift_nfeatures,
        contrast_threshold=sift_contrast_threshold,
        edge_threshold=sift_edge_threshold,
        sigma=sift_sigma,
        backend=sift_backend,
        cudasift_max_points=cudasift_max_points,
        cudasift_num_octaves=cudasift_num_octaves,
        cudasift_lowest_scale=cudasift_lowest_scale,
        cudasift_upscale=cudasift_upscale,
    )
    return FeatureBundle(
        color_hist=color_hist,
        keypoints=keypoints,
        descriptors=descriptors,
        image_shape=tuple(image_bgr.shape[:2]),
    )
