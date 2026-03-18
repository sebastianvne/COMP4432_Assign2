from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin
from tqdm.auto import tqdm


def stack_descriptors(descriptor_sets: Iterable[np.ndarray]) -> np.ndarray:
    valid_sets = [
        descriptors.astype(np.float32, copy=False)
        for descriptors in descriptor_sets
        if descriptors is not None and len(descriptors) > 0
    ]
    if not valid_sets:
        raise ValueError("训练集没有可用的 SIFT 描述子，无法训练视觉词典。")
    return np.vstack(valid_sets)


def train_visual_vocabulary(
    descriptor_sets: Sequence[np.ndarray],
    num_words: int = 300,
    random_state: int = 42,
    batch_size: int = 2048,
    max_iter: int = 100,
) -> MiniBatchKMeans:
    all_descriptors = stack_descriptors(
        tqdm(descriptor_sets, desc="堆叠训练描述子", unit="set")
    )
    effective_num_words = max(1, min(num_words, len(all_descriptors)))

    kmeans = MiniBatchKMeans(
        n_clusters=effective_num_words,
        random_state=random_state,
        batch_size=min(batch_size, len(all_descriptors)),
        max_iter=max_iter,
        n_init=10,
        reassignment_ratio=0.01,
        verbose=1,
    )
    kmeans.fit(all_descriptors)
    return kmeans


def compute_center_weights(
    keypoints: Sequence,
    image_shape: Sequence[int],
    min_weight: float = 0.1,
    max_weight: float = 1.5,
) -> np.ndarray:
    if len(keypoints) == 0:
        return np.zeros((0,), dtype=np.float32)

    height, width = image_shape[:2]
    center_x = width / 2.0
    center_y = height / 2.0

    distances = np.array(
        [
            np.hypot(keypoint.pt[0] - center_x, keypoint.pt[1] - center_y)
            for keypoint in keypoints
        ],
        dtype=np.float32,
    )

    corner_distances = np.array(
        [
            np.hypot(center_x, center_y),
            np.hypot(width - center_x, center_y),
            np.hypot(center_x, height - center_y),
            np.hypot(width - center_x, height - center_y),
        ],
        dtype=np.float32,
    )
    max_distance = float(corner_distances.max()) if corner_distances.size else 1.0
    max_distance = max(max_distance, 1e-6)

    normalized = np.clip(distances / max_distance, 0.0, 1.0)
    weights = max_weight - (max_weight - min_weight) * normalized
    return weights.astype(np.float32)


def encode_weighted_bovw(
    descriptors: np.ndarray,
    keypoints: Sequence,
    image_shape: Sequence[int],
    vocabulary: np.ndarray,
    min_weight: float = 0.1,
    max_weight: float = 1.5,
) -> np.ndarray:
    num_words = int(vocabulary.shape[0])
    histogram = np.zeros((num_words,), dtype=np.float32)

    if descriptors is None or len(descriptors) == 0 or len(keypoints) == 0:
        return histogram

    words = pairwise_distances_argmin(descriptors, vocabulary, metric="euclidean")
    weights = compute_center_weights(
        keypoints=keypoints,
        image_shape=image_shape,
        min_weight=min_weight,
        max_weight=max_weight,
    )
    histogram = np.bincount(words, weights=weights, minlength=num_words).astype(np.float32)

    total = float(histogram.sum())
    if total > 0.0:
        histogram /= total
    return histogram


def encode_bovw_batch(
    descriptor_sets: Sequence[np.ndarray],
    keypoint_sets: Sequence[Sequence],
    image_shapes: Sequence[Sequence[int]],
    vocabulary: np.ndarray,
    min_weight: float = 0.1,
    max_weight: float = 1.5,
) -> np.ndarray:
    histograms = []
    iterator = zip(descriptor_sets, keypoint_sets, image_shapes)
    for descriptors, keypoints, image_shape in tqdm(
        iterator,
        total=len(descriptor_sets),
        desc="编码 BoVW",
        unit="img",
    ):
        histograms.append(
            encode_weighted_bovw(
                descriptors=descriptors,
                keypoints=keypoints,
                image_shape=image_shape,
                vocabulary=vocabulary,
                min_weight=min_weight,
                max_weight=max_weight,
            )
        )
    return np.vstack(histograms).astype(np.float32)


def save_artifact(data: object, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(data, path)


def load_artifact(path: Path | str) -> object:
    return joblib.load(path)
