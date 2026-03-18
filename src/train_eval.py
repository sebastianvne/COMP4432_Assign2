from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


def fuse_features(
    color_features: np.ndarray,
    bovw_features: np.ndarray,
    color_weight: float = 1.0,
    bovw_weight: float = 1.0,
) -> np.ndarray:
    color_features = np.asarray(color_features, dtype=np.float32)
    bovw_features = np.asarray(bovw_features, dtype=np.float32)

    if color_features.ndim == 1:
        color_features = color_features[None, :]
    if bovw_features.ndim == 1:
        bovw_features = bovw_features[None, :]

    weighted_color = color_features * float(color_weight)
    weighted_bovw = bovw_features * float(bovw_weight)
    fused = np.concatenate([weighted_color, weighted_bovw], axis=1)
    return normalize(fused, norm="l2").astype(np.float32)


def fit_knn_index(
    train_embeddings: np.ndarray,
    metric: str = "cosine",
) -> NearestNeighbors:
    train_embeddings = np.asarray(train_embeddings, dtype=np.float32)
    algorithm = "brute" if metric == "cosine" else "auto"
    index = NearestNeighbors(metric=metric, algorithm=algorithm)
    index.fit(train_embeddings)
    return index


def retrieve_neighbors(
    index: NearestNeighbors,
    query_embeddings: np.ndarray,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
    if query_embeddings.ndim == 1:
        query_embeddings = query_embeddings[None, :]

    effective_k = max(1, min(top_k, index.n_samples_fit_))
    distances, indices = index.kneighbors(query_embeddings, n_neighbors=effective_k)
    return distances, indices


def majority_vote(neighbor_labels: Sequence[int]) -> int:
    counts = Counter(neighbor_labels)
    best_count = max(counts.values())
    for label in neighbor_labels:
        if counts[label] == best_count:
            return int(label)
    return int(neighbor_labels[0])


def classify_by_knn(
    index: NearestNeighbors,
    train_labels: Sequence[int],
    query_embeddings: np.ndarray,
    top_k: int,
) -> np.ndarray:
    _, indices = retrieve_neighbors(index, query_embeddings, top_k=top_k)
    predictions = []
    train_labels = np.asarray(train_labels)
    for row in indices:
        neighbor_labels = train_labels[row].tolist()
        predictions.append(majority_vote(neighbor_labels))
    return np.asarray(predictions, dtype=np.int32)


def evaluate_classification(
    index: NearestNeighbors,
    train_labels: Sequence[int],
    test_embeddings: np.ndarray,
    test_labels: Sequence[int],
    top_k: int,
) -> Dict[str, object]:
    predictions = classify_by_knn(
        index=index,
        train_labels=train_labels,
        query_embeddings=test_embeddings,
        top_k=top_k,
    )
    test_labels = np.asarray(test_labels, dtype=np.int32)
    return {
        "accuracy": float(accuracy_score(test_labels, predictions)),
        "macro_f1": float(f1_score(test_labels, predictions, average="macro")),
        "confusion_matrix": confusion_matrix(test_labels, predictions).tolist(),
        "predictions": predictions.tolist(),
    }


def compute_mean_average_precision(
    index: NearestNeighbors,
    train_labels: Sequence[int],
    test_embeddings: np.ndarray,
    test_labels: Sequence[int],
) -> float:
    train_labels = np.asarray(train_labels, dtype=np.int32)
    test_labels = np.asarray(test_labels, dtype=np.int32)
    _, indices = retrieve_neighbors(index, test_embeddings, top_k=len(train_labels))

    average_precisions: List[float] = []
    for label, ranked_indices in zip(test_labels, indices):
        hits = 0
        precisions: List[float] = []
        for rank, index_value in enumerate(ranked_indices, start=1):
            if train_labels[index_value] == label:
                hits += 1
                precisions.append(hits / rank)
        average_precisions.append(float(np.mean(precisions)) if precisions else 0.0)

    return float(np.mean(average_precisions)) if average_precisions else 0.0


def evaluate_retrieval(
    index: NearestNeighbors,
    train_labels: Sequence[int],
    test_embeddings: np.ndarray,
    test_labels: Sequence[int],
    top_ks: Sequence[int] = (1, 5),
) -> Dict[str, object]:
    train_labels = np.asarray(train_labels, dtype=np.int32)
    test_labels = np.asarray(test_labels, dtype=np.int32)
    max_top_k = max(top_ks)
    _, indices = retrieve_neighbors(index, test_embeddings, top_k=max_top_k)

    metrics: Dict[str, object] = {}
    for top_k in top_ks:
        hits = 0
        for label, ranked_indices in zip(test_labels, indices):
            retrieved_labels = train_labels[ranked_indices[:top_k]]
            if np.any(retrieved_labels == label):
                hits += 1
        metrics[f"top_{top_k}_accuracy"] = float(hits / len(test_labels))

    metrics["mAP"] = compute_mean_average_precision(
        index=index,
        train_labels=train_labels,
        test_embeddings=test_embeddings,
        test_labels=test_labels,
    )
    return metrics


def build_prediction_payload(
    index: NearestNeighbors,
    train_embeddings: np.ndarray,
    train_labels: Sequence[int],
    train_paths: Sequence[str],
    label_to_name: Dict[int, str],
    query_embedding: np.ndarray,
    top_k: int,
    metric: str = "cosine",
) -> Dict[str, object]:
    distances, indices = retrieve_neighbors(index, query_embedding, top_k=top_k)
    distances = distances[0]
    indices = indices[0]

    neighbor_labels = [int(train_labels[idx]) for idx in indices]
    prediction = majority_vote(neighbor_labels)

    neighbors = []
    for idx, distance in zip(indices, distances):
        score = float(1.0 - distance) if metric == "cosine" else float(distance)
        label = int(train_labels[idx])
        neighbors.append(
            {
                "path": str(train_paths[idx]),
                "label": label,
                "cn_name": label_to_name.get(label, str(label)),
                "score": score,
            }
        )

    return {
        "predicted_label": prediction,
        "predicted_name": label_to_name.get(prediction, str(prediction)),
        "neighbors": neighbors,
    }


def save_json(data: Dict[str, object], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
