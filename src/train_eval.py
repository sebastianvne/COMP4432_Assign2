from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.svm import SVC

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency in some environments
    torch = None


@dataclass(frozen=True)
class TorchKNNIndex:
    train_embeddings: "torch.Tensor"
    metric: str
    train_size: int
    query_batch_size: int


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


def resolve_torch_device(device: str) -> "torch.device":
    if torch is None:
        raise RuntimeError("当前环境未安装 torch，无法使用 torch 后端。")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求使用 CUDA，但当前环境不可用。")
    return torch.device(device)


def fit_knn_index_torch(
    train_embeddings: np.ndarray,
    metric: str = "cosine",
    device: str = "auto",
    query_batch_size: int = 256,
) -> TorchKNNIndex:
    if torch is None:
        raise RuntimeError("当前环境未安装 torch，无法使用 torch KNN 后端。")
    torch_device = resolve_torch_device(device)
    train_tensor = torch.as_tensor(train_embeddings, dtype=torch.float32, device=torch_device)
    if metric == "cosine":
        train_tensor = torch.nn.functional.normalize(train_tensor, p=2, dim=1)
    return TorchKNNIndex(
        train_embeddings=train_tensor,
        metric=metric,
        train_size=int(train_tensor.shape[0]),
        query_batch_size=max(1, int(query_batch_size)),
    )


def build_knn_index(
    train_embeddings: np.ndarray,
    metric: str = "cosine",
    backend: str = "sklearn",
    torch_device: str = "auto",
    query_batch_size: int = 256,
) -> NearestNeighbors | TorchKNNIndex:
    backend = backend.lower()
    if backend == "torch":
        return fit_knn_index_torch(
            train_embeddings=train_embeddings,
            metric=metric,
            device=torch_device,
            query_batch_size=query_batch_size,
        )
    return fit_knn_index(
        train_embeddings=train_embeddings,
        metric=metric,
    )


def _retrieve_neighbors_torch(
    index: TorchKNNIndex,
    query_embeddings: np.ndarray,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if torch is None:
        raise RuntimeError("当前环境未安装 torch，无法使用 torch KNN 后端。")

    query_array = np.asarray(query_embeddings, dtype=np.float32)
    if query_array.ndim == 1:
        query_array = query_array[None, :]

    effective_k = max(1, min(top_k, index.train_size))
    query_tensor = torch.as_tensor(query_array, dtype=torch.float32, device=index.train_embeddings.device)
    if index.metric == "cosine":
        query_tensor = torch.nn.functional.normalize(query_tensor, p=2, dim=1)

    distance_batches: List[torch.Tensor] = []
    index_batches: List[torch.Tensor] = []
    batch_size = max(1, int(index.query_batch_size))
    for start in range(0, query_tensor.shape[0], batch_size):
        query_batch = query_tensor[start : start + batch_size]
        if index.metric == "cosine":
            similarity = query_batch @ index.train_embeddings.T
            values, indices = torch.topk(similarity, k=effective_k, dim=1, largest=True)
            distances = 1.0 - values
        else:
            all_distances = torch.cdist(query_batch, index.train_embeddings)
            distances, indices = torch.topk(all_distances, k=effective_k, dim=1, largest=False)
        distance_batches.append(distances)
        index_batches.append(indices)

    distances = torch.cat(distance_batches, dim=0).cpu().numpy().astype(np.float32, copy=False)
    indices = torch.cat(index_batches, dim=0).cpu().numpy().astype(np.int32, copy=False)
    return distances, indices


def retrieve_neighbors(
    index: NearestNeighbors | TorchKNNIndex,
    query_embeddings: np.ndarray,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(index, TorchKNNIndex):
        return _retrieve_neighbors_torch(index, query_embeddings=query_embeddings, top_k=top_k)

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
    precomputed_indices: np.ndarray | None = None,
) -> np.ndarray:
    if precomputed_indices is None:
        _, indices = retrieve_neighbors(index, query_embeddings, top_k=top_k)
    else:
        indices = np.asarray(precomputed_indices, dtype=np.int32)
        if indices.ndim == 1:
            indices = indices[None, :]
        indices = indices[:, :top_k]
    predictions = []
    train_labels = np.asarray(train_labels)
    for row in indices:
        neighbor_labels = train_labels[row].tolist()
        predictions.append(majority_vote(neighbor_labels))
    return np.asarray(predictions, dtype=np.int32)


def fit_svm_classifier(
    train_embeddings: np.ndarray,
    train_labels: Sequence[int],
    kernel: str = "rbf",
    c_value: float = 1.0,
    gamma: str | float = "scale",
) -> SVC:
    parsed_gamma: str | float = gamma
    if isinstance(gamma, str) and gamma not in {"scale", "auto"}:
        parsed_gamma = float(gamma)
    classifier = SVC(
        kernel=kernel,
        C=float(c_value),
        gamma=parsed_gamma,
        class_weight="balanced",
        decision_function_shape="ovr",
    )
    classifier.fit(np.asarray(train_embeddings, dtype=np.float32), np.asarray(train_labels, dtype=np.int32))
    return classifier


def fit_rf_classifier(
    train_embeddings: np.ndarray,
    train_labels: Sequence[int],
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    random_state: int = 42,
) -> RandomForestClassifier:
    classifier = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=max_depth,
        min_samples_leaf=int(min_samples_leaf),
        class_weight="balanced",
        random_state=int(random_state),
        n_jobs=-1,
    )
    classifier.fit(np.asarray(train_embeddings, dtype=np.float32), np.asarray(train_labels, dtype=np.int32))
    return classifier


def vote_predictions(prediction_groups: Sequence[Sequence[int]]) -> np.ndarray:
    if not prediction_groups:
        raise ValueError("至少需要一组预测结果才能执行投票。")
    stacked = np.vstack([np.asarray(group, dtype=np.int32) for group in prediction_groups])
    voted = [majority_vote(stacked[:, idx].tolist()) for idx in range(stacked.shape[1])]
    return np.asarray(voted, dtype=np.int32)


def evaluate_predictions(
    predictions: Sequence[int],
    test_labels: Sequence[int],
    extra: Dict[str, object] | None = None,
) -> Dict[str, object]:
    test_labels = np.asarray(test_labels, dtype=np.int32)
    predictions = np.asarray(predictions, dtype=np.int32)
    payload = dict(extra or {})
    payload.update(
        {
            "accuracy": float(accuracy_score(test_labels, predictions)),
            "macro_precision": float(
                precision_score(test_labels, predictions, average="macro", zero_division=0)
            ),
            "macro_recall": float(
                recall_score(test_labels, predictions, average="macro", zero_division=0)
            ),
            "macro_f1": float(f1_score(test_labels, predictions, average="macro")),
            "confusion_matrix": confusion_matrix(test_labels, predictions).tolist(),
            "predictions": predictions.tolist(),
        }
    )
    return payload


def evaluate_knn_classification(
    index: NearestNeighbors,
    train_labels: Sequence[int],
    test_embeddings: np.ndarray,
    test_labels: Sequence[int],
    top_k: int,
    precomputed_indices: np.ndarray | None = None,
) -> Dict[str, object]:
    predictions = classify_by_knn(
        index=index,
        train_labels=train_labels,
        query_embeddings=test_embeddings,
        top_k=top_k,
        precomputed_indices=precomputed_indices,
    )
    return evaluate_predictions(
        predictions=predictions,
        test_labels=test_labels,
        extra={"classifier": "knn", "top_k": int(top_k)},
    )


def evaluate_svm_classification(
    classifier: SVC,
    test_embeddings: np.ndarray,
    test_labels: Sequence[int],
) -> Dict[str, object]:
    predictions = classifier.predict(np.asarray(test_embeddings, dtype=np.float32))
    return evaluate_predictions(
        predictions=predictions,
        test_labels=test_labels,
        extra={"classifier": "svm"},
    )


def evaluate_rf_classification(
    classifier: RandomForestClassifier,
    test_embeddings: np.ndarray,
    test_labels: Sequence[int],
) -> Dict[str, object]:
    predictions = classifier.predict(np.asarray(test_embeddings, dtype=np.float32))
    return evaluate_predictions(
        predictions=predictions,
        test_labels=test_labels,
        extra={"classifier": "rf"},
    )


def evaluate_ensemble_classification(
    index: NearestNeighbors,
    train_labels: Sequence[int],
    test_embeddings: np.ndarray,
    test_labels: Sequence[int],
    top_k: int,
    svm_classifier: SVC,
    rf_classifier: RandomForestClassifier,
    precomputed_indices: np.ndarray | None = None,
) -> Dict[str, object]:
    knn_predictions = classify_by_knn(
        index=index,
        train_labels=train_labels,
        query_embeddings=test_embeddings,
        top_k=top_k,
        precomputed_indices=precomputed_indices,
    )
    svm_predictions = svm_classifier.predict(np.asarray(test_embeddings, dtype=np.float32))
    rf_predictions = rf_classifier.predict(np.asarray(test_embeddings, dtype=np.float32))
    predictions = vote_predictions([knn_predictions, svm_predictions, rf_predictions])
    return evaluate_predictions(
        predictions=predictions,
        test_labels=test_labels,
        extra={
            "classifier": "ensemble",
            "members": ["knn", "svm", "rf"],
            "top_k": int(top_k),
        },
    )


def predict_labels_by_classifier(
    classifier_name: str,
    index: NearestNeighbors,
    train_labels: Sequence[int],
    query_embeddings: np.ndarray,
    top_k: int,
    svm_classifier: SVC | None = None,
    rf_classifier: RandomForestClassifier | None = None,
    precomputed_indices: np.ndarray | None = None,
) -> np.ndarray:
    query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
    if query_embeddings.ndim == 1:
        query_embeddings = query_embeddings[None, :]

    if classifier_name == "svm":
        if svm_classifier is None:
            raise ValueError("classifier_name=svm 时必须提供 svm_classifier。")
        return np.asarray(svm_classifier.predict(query_embeddings), dtype=np.int32)

    if classifier_name == "rf":
        if rf_classifier is None:
            raise ValueError("classifier_name=rf 时必须提供 rf_classifier。")
        return np.asarray(rf_classifier.predict(query_embeddings), dtype=np.int32)

    if classifier_name == "ensemble":
        if svm_classifier is None or rf_classifier is None:
            raise ValueError("classifier_name=ensemble 时必须同时提供 svm_classifier 和 rf_classifier。")
        knn_predictions = classify_by_knn(
            index=index,
            train_labels=train_labels,
            query_embeddings=query_embeddings,
            top_k=top_k,
            precomputed_indices=precomputed_indices,
        )
        svm_predictions = np.asarray(svm_classifier.predict(query_embeddings), dtype=np.int32)
        rf_predictions = np.asarray(rf_classifier.predict(query_embeddings), dtype=np.int32)
        return vote_predictions([knn_predictions, svm_predictions, rf_predictions])

    return classify_by_knn(
        index=index,
        train_labels=train_labels,
        query_embeddings=query_embeddings,
        top_k=top_k,
        precomputed_indices=precomputed_indices,
    )


def evaluate_classification(
    classifier_name: str,
    index: NearestNeighbors,
    train_labels: Sequence[int],
    test_embeddings: np.ndarray,
    test_labels: Sequence[int],
    top_k: int,
    svm_classifier: SVC | None = None,
    rf_classifier: RandomForestClassifier | None = None,
    precomputed_distances: np.ndarray | None = None,
    precomputed_indices: np.ndarray | None = None,
) -> Dict[str, object]:
    if classifier_name == "svm":
        if svm_classifier is None:
            raise ValueError("classifier_name=svm 时必须提供 svm_classifier。")
        return evaluate_svm_classification(
            classifier=svm_classifier,
            test_embeddings=test_embeddings,
            test_labels=test_labels,
        )

    if classifier_name == "rf":
        if rf_classifier is None:
            raise ValueError("classifier_name=rf 时必须提供 rf_classifier。")
        return evaluate_rf_classification(
            classifier=rf_classifier,
            test_embeddings=test_embeddings,
            test_labels=test_labels,
        )

    if classifier_name == "ensemble":
        if svm_classifier is None or rf_classifier is None:
            raise ValueError("classifier_name=ensemble 时必须同时提供 svm_classifier 和 rf_classifier。")
        return evaluate_ensemble_classification(
            index=index,
            train_labels=train_labels,
            test_embeddings=test_embeddings,
            test_labels=test_labels,
            top_k=top_k,
            svm_classifier=svm_classifier,
            rf_classifier=rf_classifier,
            precomputed_indices=precomputed_indices,
        )

    return evaluate_knn_classification(
        index=index,
        train_labels=train_labels,
        test_embeddings=test_embeddings,
        test_labels=test_labels,
        top_k=top_k,
        precomputed_indices=precomputed_indices,
    )


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
    precomputed_indices: np.ndarray | None = None,
) -> Dict[str, object]:
    train_labels = np.asarray(train_labels, dtype=np.int32)
    test_labels = np.asarray(test_labels, dtype=np.int32)
    max_top_k = max(top_ks)
    if precomputed_indices is None:
        _, indices = retrieve_neighbors(index, test_embeddings, top_k=max_top_k)
    else:
        indices = np.asarray(precomputed_indices, dtype=np.int32)
        if indices.ndim == 1:
            indices = indices[None, :]
        indices = indices[:, :max_top_k]

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
    query_embedding: np.ndarray | None,
    top_k: int,
    metric: str = "cosine",
    predicted_label: int | None = None,
    precomputed_distances: np.ndarray | None = None,
    precomputed_indices: np.ndarray | None = None,
) -> Dict[str, object]:
    if precomputed_distances is not None and precomputed_indices is not None:
        distances = np.asarray(precomputed_distances, dtype=np.float32).reshape(-1)[:top_k]
        indices = np.asarray(precomputed_indices, dtype=np.int32).reshape(-1)[:top_k]
    else:
        if query_embedding is None:
            raise ValueError("未提供 query_embedding，且没有传入预计算的邻居结果。")
        distances, indices = retrieve_neighbors(index, query_embedding, top_k=top_k)
        distances = distances[0]
        indices = indices[0]

    neighbor_labels = [int(train_labels[idx]) for idx in indices]
    prediction = int(predicted_label) if predicted_label is not None else majority_vote(neighbor_labels)

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
