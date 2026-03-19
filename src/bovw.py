from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin
from tqdm.auto import tqdm

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency in some environments
    torch = None


@dataclass(frozen=True)
class TorchBoVWConfig:
    device: str = "cuda"
    descriptor_chunk_size: int = 8192


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
    if isinstance(keypoints, np.ndarray):
        keypoint_points = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
    else:
        keypoint_points = np.array(
            [keypoint.pt for keypoint in keypoints],
            dtype=np.float32,
        ).reshape(-1, 2)

    distances = np.array(
        [
            np.hypot(point[0] - center_x, point[1] - center_y)
            for point in keypoint_points
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


def resolve_torch_device(device: str) -> "torch.device":
    if torch is None:
        raise RuntimeError("当前环境未安装 torch，无法使用 torch 后端。")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求使用 CUDA，但当前环境不可用。")
    return torch.device(device)


def compute_center_weights_torch(
    keypoints: Sequence,
    image_shape: Sequence[int],
    min_weight: float = 0.1,
    max_weight: float = 1.5,
    device: "torch.device | None" = None,
) -> "torch.Tensor":
    if len(keypoints) == 0:
        return torch.zeros((0,), dtype=torch.float32, device=device)

    height, width = image_shape[:2]
    center_x = width / 2.0
    center_y = height / 2.0

    if isinstance(keypoints, np.ndarray):
        keypoint_points = torch.as_tensor(keypoints, dtype=torch.float32, device=device).reshape(-1, 2)
    else:
        keypoint_points = torch.as_tensor(
            [keypoint.pt for keypoint in keypoints],
            dtype=torch.float32,
            device=device,
        ).reshape(-1, 2)

    center = torch.tensor([center_x, center_y], dtype=torch.float32, device=device)
    distances = torch.linalg.norm(keypoint_points - center, dim=1)
    corner_points = torch.tensor(
        [
            [0.0, 0.0],
            [float(width), 0.0],
            [0.0, float(height)],
            [float(width), float(height)],
        ],
        dtype=torch.float32,
        device=device,
    )
    corner_distances = torch.linalg.norm(corner_points - center, dim=1)
    max_distance = torch.clamp(corner_distances.max(), min=1e-6)
    normalized = torch.clamp(distances / max_distance, 0.0, 1.0)
    weights = max_weight - (max_weight - min_weight) * normalized
    return weights.to(dtype=torch.float32)


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


def encode_weighted_bovw_torch(
    descriptors: np.ndarray,
    keypoints: Sequence,
    image_shape: Sequence[int],
    vocabulary: np.ndarray,
    min_weight: float = 0.1,
    max_weight: float = 1.5,
    config: TorchBoVWConfig | None = None,
) -> np.ndarray:
    if torch is None:
        raise RuntimeError("当前环境未安装 torch，无法使用 torch 后端。")

    config = config or TorchBoVWConfig()
    device = resolve_torch_device(config.device)
    num_words = int(vocabulary.shape[0])
    if descriptors is None or len(descriptors) == 0 or len(keypoints) == 0:
        return np.zeros((num_words,), dtype=np.float32)

    descriptor_tensor = torch.as_tensor(descriptors, dtype=torch.float32, device=device)
    vocabulary_tensor = torch.as_tensor(vocabulary, dtype=torch.float32, device=device)
    weights = compute_center_weights_torch(
        keypoints=keypoints,
        image_shape=image_shape,
        min_weight=min_weight,
        max_weight=max_weight,
        device=device,
    )
    histogram = torch.zeros((num_words,), dtype=torch.float32, device=device)

    chunk_size = max(1, int(config.descriptor_chunk_size))
    word_chunks = []
    for start in range(0, descriptor_tensor.shape[0], chunk_size):
        descriptor_chunk = descriptor_tensor[start : start + chunk_size]
        distances = torch.cdist(descriptor_chunk, vocabulary_tensor)
        word_chunks.append(torch.argmin(distances, dim=1))
    words = torch.cat(word_chunks, dim=0)
    histogram.scatter_add_(0, words, weights)

    total = float(histogram.sum().item())
    if total > 0.0:
        histogram /= total
    return histogram.cpu().numpy().astype(np.float32, copy=False)


def _encode_weighted_bovw_task(task: tuple[np.ndarray, Sequence, Sequence[int], np.ndarray, float, float]) -> np.ndarray:
    descriptors, keypoints, image_shape, vocabulary, min_weight, max_weight = task
    return encode_weighted_bovw(
        descriptors=descriptors,
        keypoints=keypoints,
        image_shape=image_shape,
        vocabulary=vocabulary,
        min_weight=min_weight,
        max_weight=max_weight,
    )


def _encode_weighted_bovw_torch_task(
    task: tuple[np.ndarray, Sequence, Sequence[int], np.ndarray, float, float, TorchBoVWConfig]
) -> np.ndarray:
    descriptors, keypoints, image_shape, vocabulary, min_weight, max_weight, config = task
    return encode_weighted_bovw_torch(
        descriptors=descriptors,
        keypoints=keypoints,
        image_shape=image_shape,
        vocabulary=vocabulary,
        min_weight=min_weight,
        max_weight=max_weight,
        config=config,
    )


def encode_bovw_batch(
    descriptor_sets: Sequence[np.ndarray],
    keypoint_sets: Sequence[Sequence],
    image_shapes: Sequence[Sequence[int]],
    vocabulary: np.ndarray,
    min_weight: float = 0.1,
    max_weight: float = 1.5,
    num_workers: int = 1,
    chunksize: int = 8,
    backend: str = "numpy",
    torch_device: str = "auto",
    descriptor_chunk_size: int = 8192,
) -> np.ndarray:
    histograms = []
    num_workers = max(1, int(num_workers))
    chunksize = max(1, int(chunksize))
    backend = backend.lower()

    if backend == "torch":
        torch_config = TorchBoVWConfig(
            device=torch_device,
            descriptor_chunk_size=descriptor_chunk_size,
        )
        tasks = [
            (
                descriptors,
                keypoints,
                image_shape,
                vocabulary,
                min_weight,
                max_weight,
                torch_config,
            )
            for descriptors, keypoints, image_shape in zip(descriptor_sets, keypoint_sets, image_shapes)
        ]
        for task in tqdm(tasks, total=len(tasks), desc="编码 BoVW (torch)", unit="img"):
            histograms.append(_encode_weighted_bovw_torch_task(task))
    else:
        tasks = [
            (
                descriptors,
                keypoints,
                image_shape,
                vocabulary,
                min_weight,
                max_weight,
            )
            for descriptors, keypoints, image_shape in zip(descriptor_sets, keypoint_sets, image_shapes)
        ]

        if num_workers == 1 or len(tasks) <= 1:
            for task in tqdm(tasks, total=len(tasks), desc="编码 BoVW", unit="img"):
                histograms.append(_encode_weighted_bovw_task(task))
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                iterator = executor.map(_encode_weighted_bovw_task, tasks, chunksize=chunksize)
                for histogram in tqdm(iterator, total=len(tasks), desc="编码 BoVW", unit="img"):
                    histograms.append(histogram)

    if not histograms:
        return np.zeros((0, vocabulary.shape[0]), dtype=np.float32)
    return np.vstack(histograms).astype(np.float32)


def save_artifact(data: object, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(data, path)


def load_artifact(path: Path | str) -> object:
    return joblib.load(path)
