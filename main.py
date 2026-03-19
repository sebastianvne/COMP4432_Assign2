from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def sanitize_omp_num_threads() -> None:
    raw_value = os.environ.get("OMP_NUM_THREADS")
    if raw_value is None:
        return
    try:
        if int(raw_value) > 0:
            return
    except ValueError:
        pass
    os.environ["OMP_NUM_THREADS"] = "1"


sanitize_omp_num_threads()


import joblib
import matplotlib
import numpy as np
from tqdm.auto import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.bovw import encode_bovw_batch, save_artifact, train_visual_vocabulary
from src.dataset import build_label_to_name, load_samples, stratified_split, summarize_labels
from src.features import extract_feature_bundle, resolve_sift_backend
from src.preprocess import preprocess_image
from src.train_eval import (
    build_knn_index,
    build_prediction_payload,
    evaluate_classification,
    evaluate_retrieval,
    fit_rf_classifier,
    predict_labels_by_classifier,
    fit_svm_classifier,
    fuse_features,
    retrieve_neighbors,
    save_json,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}


def resolve_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (project_root / path).resolve()


def default_num_workers() -> int:
    return max(1, min(4, os.cpu_count() or 1))


def keypoints_to_points(keypoints: Sequence) -> np.ndarray:
    if isinstance(keypoints, np.ndarray):
        return np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
    if not keypoints:
        return np.zeros((0, 2), dtype=np.float32)
    points = np.array([keypoint.pt for keypoint in keypoints], dtype=np.float32)
    return points.reshape(-1, 2)


def build_feature_cache_signature(sample, args) -> str:
    file_stat = sample.path.stat()
    payload = {
        "path": sample.relative_path,
        "size": int(file_stat.st_size),
        "mtime_ns": int(file_stat.st_mtime_ns),
        "max_side": int(args.max_side),
        "h_bins": int(args.h_bins),
        "s_bins": int(args.s_bins),
        "sift_nfeatures": int(args.sift_nfeatures),
        "sift_contrast_threshold": float(args.sift_contrast_threshold),
        "sift_edge_threshold": float(args.sift_edge_threshold),
        "sift_sigma": float(args.sift_sigma),
        "sift_backend": str(getattr(args, "sift_backend", "opencv")),
        "cudasift_max_points": int(getattr(args, "cudasift_max_points", 32768)),
        "cudasift_num_octaves": int(getattr(args, "cudasift_num_octaves", 5)),
        "cudasift_lowest_scale": float(getattr(args, "cudasift_lowest_scale", 0.0)),
        "cudasift_upscale": bool(getattr(args, "cudasift_upscale", False)),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def build_feature_cache_path(project_root: Path, sample, args) -> Path:
    cache_dir = project_root / "feature"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{build_feature_cache_signature(sample, args)}.joblib"


def should_use_spawn_for_feature_workers(config) -> bool:
    backend = getattr(config, "sift_backend", "opencv")
    try:
        resolved_backend = resolve_sift_backend(backend)
    except Exception:
        resolved_backend = str(backend).lower()
    return resolved_backend == "cudasift"


def effective_feature_worker_count(config, requested_workers: int) -> int:
    requested_workers = max(1, int(requested_workers))
    if should_use_spawn_for_feature_workers(config):
        return 1
    return requested_workers


def extract_or_load_record(project_root: Path, sample, args) -> tuple[Dict[str, object], bool]:
    cache_path = build_feature_cache_path(project_root, sample, args)
    if cache_path.exists():
        cached = joblib.load(cache_path)
        record = {
            "sample": sample,
            "color_hist": np.asarray(cached["color_hist"], dtype=np.float32),
            "keypoints": np.asarray(cached["keypoints"], dtype=np.float32).reshape(-1, 2),
            "descriptors": np.asarray(cached["descriptors"], dtype=np.float32).reshape(-1, 128),
            "image_shape": tuple(cached["image_shape"]),
        }
        return record, True

    preprocess_result = preprocess_image(
        sample.path,
        max_side=args.max_side,
    )
    feature_bundle = extract_feature_bundle(
        preprocess_result.image_bgr,
        h_bins=args.h_bins,
        s_bins=args.s_bins,
        sift_nfeatures=args.sift_nfeatures,
        sift_contrast_threshold=args.sift_contrast_threshold,
        sift_edge_threshold=args.sift_edge_threshold,
        sift_sigma=args.sift_sigma,
        sift_backend=args.sift_backend,
        cudasift_max_points=args.cudasift_max_points,
        cudasift_num_octaves=args.cudasift_num_octaves,
        cudasift_lowest_scale=args.cudasift_lowest_scale,
        cudasift_upscale=args.cudasift_upscale,
    )
    keypoint_points = keypoints_to_points(feature_bundle.keypoints)
    record = {
        "sample": sample,
        "color_hist": feature_bundle.color_hist.astype(np.float32, copy=False),
        "keypoints": keypoint_points,
        "descriptors": feature_bundle.descriptors.astype(np.float32, copy=False),
        "image_shape": feature_bundle.image_shape,
    }
    joblib.dump(
        {
            "path": sample.relative_path,
            "color_hist": record["color_hist"],
            "keypoints": record["keypoints"],
            "descriptors": record["descriptors"],
            "image_shape": record["image_shape"],
        },
        cache_path,
    )
    return record, False


def _extract_record_task(project_root: Path, sample, args) -> tuple[Dict[str, object], bool]:
    return extract_or_load_record(project_root, sample, args)


def extract_records(samples, args, project_root: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    cache_hits = 0
    requested_workers = max(1, int(getattr(args, "num_workers", 1)))
    num_workers = effective_feature_worker_count(args, requested_workers)
    chunksize = max(1, int(getattr(args, "chunksize", 1)))

    if num_workers != requested_workers:
        print(
            f"检测到 cudasift 后端，特征提取 worker 从 {requested_workers} 调整为 {num_workers}，"
            "避免多进程同时占用 GPU 导致显存溢出。"
        )

    if num_workers == 1 or len(samples) <= 1:
        for sample in tqdm(samples, desc="提取图像特征", unit="img"):
            record, loaded_from_cache = extract_or_load_record(project_root, sample, args)
            cache_hits += int(loaded_from_cache)
            records.append(record)
    else:
        executor_kwargs = {"max_workers": num_workers}
        if should_use_spawn_for_feature_workers(args):
            executor_kwargs["mp_context"] = mp.get_context("spawn")
        with ProcessPoolExecutor(**executor_kwargs) as executor:
            iterator = executor.map(
                _extract_record_task,
                repeat(project_root),
                samples,
                repeat(args),
                chunksize=chunksize,
            )
            for record, loaded_from_cache in tqdm(iterator, total=len(samples), desc="提取图像特征", unit="img"):
                cache_hits += int(loaded_from_cache)
                records.append(record)

    cache_misses = len(samples) - cache_hits
    print(f"特征缓存命中: {cache_hits} | 新提取: {cache_misses} | 缓存目录: {project_root / 'feature'}")
    return records


def build_feature_matrices(records, vocabulary: np.ndarray, args) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    color_features = np.vstack([record["color_hist"] for record in records]).astype(np.float32)
    bovw_features = encode_bovw_batch(
        descriptor_sets=[record["descriptors"] for record in records],
        keypoint_sets=[record["keypoints"] for record in records],
        image_shapes=[record["image_shape"] for record in records],
        vocabulary=vocabulary,
        min_weight=args.min_spatial_weight,
        max_weight=args.max_spatial_weight,
        num_workers=getattr(args, "num_workers", 1),
        chunksize=getattr(args, "chunksize", 8),
        backend=getattr(args, "bovw_backend", "numpy"),
        torch_device=getattr(args, "torch_device", "auto"),
        descriptor_chunk_size=getattr(args, "torch_descriptor_chunk_size", 8192),
    )
    embeddings = fuse_features(
        color_features=color_features,
        bovw_features=bovw_features,
        color_weight=args.color_weight,
        bovw_weight=args.bovw_weight,
    )
    return color_features, bovw_features, embeddings


def build_embeddings(records, vocabulary: np.ndarray, args) -> np.ndarray:
    _, _, embeddings = build_feature_matrices(records, vocabulary=vocabulary, args=args)
    return embeddings


def extract_query_record(image_path: Path, config) -> Dict[str, object]:
    preprocess_result = preprocess_image(
        image_path,
        max_side=config.max_side,
    )
    feature_bundle = extract_feature_bundle(
        preprocess_result.image_bgr,
        h_bins=config.h_bins,
        s_bins=config.s_bins,
        sift_nfeatures=config.sift_nfeatures,
        sift_contrast_threshold=config.sift_contrast_threshold,
        sift_edge_threshold=config.sift_edge_threshold,
        sift_sigma=config.sift_sigma,
        sift_backend=getattr(config, "sift_backend", "opencv"),
        cudasift_max_points=getattr(config, "cudasift_max_points", 32768),
        cudasift_num_octaves=getattr(config, "cudasift_num_octaves", 5),
        cudasift_lowest_scale=getattr(config, "cudasift_lowest_scale", 0.0),
        cudasift_upscale=getattr(config, "cudasift_upscale", False),
    )
    return {
        "path": str(image_path),
        "color_hist": feature_bundle.color_hist.astype(np.float32, copy=False),
        "keypoints": keypoints_to_points(feature_bundle.keypoints),
        "descriptors": feature_bundle.descriptors.astype(np.float32, copy=False),
        "image_shape": feature_bundle.image_shape,
    }


def _extract_query_record_task(image_path: Path, config) -> Dict[str, object]:
    return extract_query_record(image_path, config)


def extract_query_records(image_paths: Sequence[Path], config, num_workers: int, chunksize: int) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    requested_workers = max(1, int(num_workers))
    num_workers = effective_feature_worker_count(config, requested_workers)
    chunksize = max(1, int(chunksize))

    if num_workers != requested_workers:
        print(
            f"检测到 cudasift 后端，查询特征提取 worker 从 {requested_workers} 调整为 {num_workers}，"
            "避免多进程同时占用 GPU 导致显存溢出。"
        )

    if num_workers == 1 or len(image_paths) <= 1:
        for image_path in tqdm(image_paths, desc="提取查询特征", unit="img"):
            records.append(extract_query_record(image_path, config))
    else:
        executor_kwargs = {"max_workers": num_workers}
        if should_use_spawn_for_feature_workers(config):
            executor_kwargs["mp_context"] = mp.get_context("spawn")
        with ProcessPoolExecutor(**executor_kwargs) as executor:
            iterator = executor.map(
                _extract_query_record_task,
                image_paths,
                repeat(config),
                chunksize=chunksize,
            )
            for record in tqdm(iterator, total=len(image_paths), desc="提取查询特征", unit="img"):
                records.append(record)
    return records


def list_image_paths(image_dir: Path, recursive: bool = False) -> List[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(
        [
            path
            for path in image_dir.glob(pattern)
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: path.as_posix(),
    )


def save_partition_records(partition_dir: Path, filename: str, samples: Sequence) -> Path:
    partition_dir.mkdir(parents=True, exist_ok=True)
    output_path = partition_dir / filename
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["path", "label", "time", "cn_name"])
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "path": sample.relative_path,
                    "label": sample.label,
                    "time": sample.time,
                    "cn_name": sample.cn_name,
                }
            )
    return output_path


def save_partition_outputs(output_dir: Path, train_samples: Sequence, test_samples: Sequence) -> Dict[str, str]:
    partition_dir = output_dir / "partition"
    train_path = save_partition_records(partition_dir, "train_split.csv", train_samples)
    test_path = save_partition_records(partition_dir, "test_split.csv", test_samples)
    metadata = {
        "train_file": str(train_path),
        "test_file": str(test_path),
        "num_train": len(train_samples),
        "num_test": len(test_samples),
        "train_labels": summarize_labels(train_samples),
        "test_labels": summarize_labels(test_samples),
    }
    save_json(metadata, partition_dir / "summary.json")
    return metadata


def parse_eval_ks(raw_value: str | None, max_allowed_k: int) -> List[int]:
    if raw_value:
        values = []
        for token in raw_value.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                value = int(token)
            except ValueError as exc:
                raise ValueError(f"无效的 k 值: {token}") from exc
            if value <= 0:
                raise ValueError("k 必须是正整数。")
            if value > max_allowed_k:
                raise ValueError(
                    f"k={value} 超过训练样本数 {max_allowed_k}，请缩小 --eval-ks 或 --max-eval-k。"
                )
            values.append(value)
        if not values:
            raise ValueError("--eval-ks 不能为空。")
        return sorted(set(values))
    return list(range(1, max_allowed_k + 1))


def evaluate_k_sweep(
    index,
    train_labels: Sequence[int],
    test_embeddings: np.ndarray,
    test_labels: Sequence[int],
    eval_ks: Iterable[int],
    precomputed_indices: np.ndarray | None = None,
) -> Dict[str, object]:
    per_k: List[Dict[str, object]] = []
    best_accuracy: Dict[str, object] | None = None
    best_recall: Dict[str, object] | None = None

    for k in eval_ks:
        metrics = evaluate_classification(
            classifier_name="knn",
            index=index,
            train_labels=train_labels,
            test_embeddings=test_embeddings,
            test_labels=test_labels,
            top_k=k,
            precomputed_indices=precomputed_indices,
        )
        record = {
            "k": int(k),
            "accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
        }
        per_k.append(record)

        if best_accuracy is None or record["accuracy"] > best_accuracy["accuracy"]:
            best_accuracy = record
        if best_recall is None or record["macro_recall"] > best_recall["macro_recall"]:
            best_recall = record

    return {
        "per_k": per_k,
        "best_accuracy": best_accuracy,
        "best_recall": best_recall,
    }


def save_k_sweep_csv(records: Sequence[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["k", "accuracy", "macro_precision", "macro_recall", "macro_f1"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def plot_k_sweep(records: Sequence[Dict[str, object]], path: Path) -> None:
    if not records:
        raise ValueError("没有可绘制的 k-sweep 结果。")

    ks = [int(record["k"]) for record in records]
    accuracies = [float(record["accuracy"]) for record in records]
    recalls = [float(record["macro_recall"]) for record in records]

    best_accuracy_index = int(np.argmax(accuracies))
    best_recall_index = int(np.argmax(recalls))

    plt.figure(figsize=(10, 6))
    plt.plot(ks, accuracies, marker="o", linewidth=2, label="Accuracy")
    plt.plot(ks, recalls, marker="s", linewidth=2, label="Macro Recall")
    plt.scatter(
        ks[best_accuracy_index],
        accuracies[best_accuracy_index],
        color="tab:blue",
        s=90,
    )
    plt.scatter(
        ks[best_recall_index],
        recalls[best_recall_index],
        color="tab:orange",
        s=90,
    )
    plt.annotate(
        f"best acc: k={ks[best_accuracy_index]} ({accuracies[best_accuracy_index]:.3f})",
        (ks[best_accuracy_index], accuracies[best_accuracy_index]),
        textcoords="offset points",
        xytext=(8, 10),
    )
    plt.annotate(
        f"best recall: k={ks[best_recall_index]} ({recalls[best_recall_index]:.3f})",
        (ks[best_recall_index], recalls[best_recall_index]),
        textcoords="offset points",
        xytext=(8, -18),
    )
    plt.xlabel("k")
    plt.ylabel("Score")
    plt.title("KNN classification metrics across different k")
    plt.xticks(ks)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def save_training_outputs(
    output_dir: Path,
    vocabulary: np.ndarray,
    train_embeddings: np.ndarray,
    train_bovw_features: np.ndarray,
    train_color_features: np.ndarray,
    train_records,
    test_embeddings: np.ndarray,
    test_records,
    metrics: Dict[str, object],
    label_to_name: Dict[int, str],
    args,
    classifier_artifacts: Dict[str, object] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_labels = np.array([record["sample"].label for record in train_records], dtype=np.int32)
    test_labels = np.array([record["sample"].label for record in test_records], dtype=np.int32)
    train_paths = [record["sample"].relative_path for record in train_records]
    test_paths = [record["sample"].relative_path for record in test_records]

    np.save(output_dir / "train_features.npy", train_embeddings)
    np.save(output_dir / "train_bovw_features.npy", train_bovw_features)
    np.save(output_dir / "train_color_features.npy", train_color_features)
    np.save(output_dir / "train_labels.npy", train_labels)
    np.save(output_dir / "test_features.npy", test_embeddings)
    np.save(output_dir / "test_labels.npy", test_labels)
    save_artifact(vocabulary, output_dir / "vocab.joblib")

    save_json({str(k): v for k, v in label_to_name.items()}, output_dir / "label_map.json")
    save_json(metrics, output_dir / "metrics.json")
    save_json(
        {"train_paths": train_paths, "test_paths": test_paths},
        output_dir / "sample_paths.json",
    )

    model_bundle = {
        "vocabulary": vocabulary,
        "train_embeddings": train_embeddings,
        "train_labels": train_labels,
        "train_paths": train_paths,
        "label_to_name": {int(k): v for k, v in label_to_name.items()},
        "classifier_name": args.classifier,
        "classifier_artifacts": classifier_artifacts or {},
        "config": vars(args),
    }
    model_path = output_dir / "model_bundle.joblib"
    joblib.dump(model_bundle, model_path)
    return model_path


def run_train(args) -> None:
    project_root = Path(__file__).resolve().parent
    csv_path = resolve_path(project_root, args.csv)
    output_dir = resolve_path(project_root, args.output_dir)

    samples = load_samples(csv_path, project_root=project_root)
    label_to_name = build_label_to_name(samples)
    train_samples, test_samples = stratified_split(
        samples,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print(f"总样本数: {len(samples)}")
    print(f"类别分布: {summarize_labels(samples)}")
    print(f"训练集: {len(train_samples)} | 测试集: {len(test_samples)}")

    partition_info = save_partition_outputs(output_dir, train_samples, test_samples)
    print(f"数据划分已保存到: {output_dir / 'partition'}")

    print("正在提取训练集特征...")
    train_records = extract_records(train_samples, args, project_root=project_root)
    print("正在训练视觉词典...")
    kmeans = train_visual_vocabulary(
        descriptor_sets=[record["descriptors"] for record in train_records],
        num_words=args.num_words,
        random_state=args.random_state,
        batch_size=args.kmeans_batch_size,
        max_iter=args.kmeans_max_iter,
    )
    vocabulary = kmeans.cluster_centers_.astype(np.float32)

    print("正在编码训练集 BoVW...")
    train_color_features, train_bovw_features, train_embeddings = build_feature_matrices(
        train_records,
        vocabulary=vocabulary,
        args=args,
    )

    print("正在提取测试集特征...")
    test_records = extract_records(test_samples, args, project_root=project_root)
    _, test_bovw_features, test_embeddings = build_feature_matrices(
        test_records,
        vocabulary=vocabulary,
        args=args,
    )

    print("正在建立 KNN 检索器并评估...")
    knn_index = build_knn_index(
        train_embeddings=train_embeddings,
        metric=args.metric,
        backend=getattr(args, "knn_backend", "sklearn"),
        torch_device=getattr(args, "torch_device", "auto"),
        query_batch_size=getattr(args, "torch_query_batch_size", 256),
    )
    train_labels = [record["sample"].label for record in train_records]
    test_labels = [record["sample"].label for record in test_records]
    retrieval_top_ks = tuple(sorted({1, min(5, args.top_k)}))
    eval_ks = None
    if args.classifier == "knn":
        eval_ks = parse_eval_ks(
            raw_value=args.eval_ks,
            max_allowed_k=min(args.max_eval_k, len(train_labels)),
        )
    max_precomputed_k = max([args.top_k, *retrieval_top_ks, *(eval_ks or [args.top_k])])
    knn_distances, knn_indices = retrieve_neighbors(knn_index, test_embeddings, top_k=max_precomputed_k)

    svm_classifier = None
    if args.classifier in {"svm", "ensemble"}:
        print("正在训练 SVM 分类器...")
        svm_classifier = fit_svm_classifier(
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            kernel=args.svm_kernel,
            c_value=args.svm_c,
            gamma=args.svm_gamma,
        )
    rf_classifier = None
    if args.classifier in {"rf", "ensemble"}:
        print("正在训练 Random Forest 分类器...")
        rf_classifier = fit_rf_classifier(
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            n_estimators=args.rf_n_estimators,
            max_depth=args.rf_max_depth,
            min_samples_leaf=args.rf_min_samples_leaf,
            random_state=args.random_state,
        )

    classification_metrics = evaluate_classification(
        classifier_name=args.classifier,
        index=knn_index,
        train_labels=train_labels,
        test_embeddings=test_embeddings,
        test_labels=test_labels,
        top_k=args.top_k,
        svm_classifier=svm_classifier,
        rf_classifier=rf_classifier,
        precomputed_distances=knn_distances,
        precomputed_indices=knn_indices,
    )
    retrieval_metrics = evaluate_retrieval(
        index=knn_index,
        train_labels=train_labels,
        test_embeddings=test_embeddings,
        test_labels=test_labels,
        top_ks=retrieval_top_ks,
        precomputed_indices=knn_indices,
    )
    k_sweep_metrics = None
    if args.classifier == "knn":
        k_sweep_metrics = evaluate_k_sweep(
            index=knn_index,
            train_labels=train_labels,
            test_embeddings=test_embeddings,
            test_labels=test_labels,
            eval_ks=eval_ks,
            precomputed_indices=knn_indices,
        )
        save_json(k_sweep_metrics, output_dir / "k_sweep_metrics.json")
        save_k_sweep_csv(k_sweep_metrics["per_k"], output_dir / "k_sweep_metrics.csv")
        plot_k_sweep(k_sweep_metrics["per_k"], output_dir / "k_sweep_accuracy_recall.png")

    classification_predictions = np.asarray(classification_metrics["predictions"], dtype=np.int32)
    train_paths = [sample["sample"].relative_path for sample in train_records]
    retrieval_examples = []
    for record, predicted_label, row_distances, row_indices in zip(
        test_records,
        classification_predictions,
        knn_distances,
        knn_indices,
    ):
        prediction = build_prediction_payload(
            index=knn_index,
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            train_paths=train_paths,
            label_to_name=label_to_name,
            query_embedding=None,
            top_k=args.top_k,
            metric=args.metric,
            predicted_label=int(predicted_label),
            precomputed_distances=row_distances,
            precomputed_indices=row_indices,
        )
        retrieval_examples.append(
            {
                "query_path": record["sample"].relative_path,
                "query_label": record["sample"].label,
                "query_name": record["sample"].cn_name,
                **prediction,
            }
        )

    metrics = {
        "classification": classification_metrics,
        "retrieval": retrieval_metrics,
        "dataset_summary": {
            "num_samples": len(samples),
            "num_train": len(train_samples),
            "num_test": len(test_samples),
            "labels": summarize_labels(samples),
        },
        "partition": partition_info,
    }
    if k_sweep_metrics is not None:
        metrics["k_sweep"] = k_sweep_metrics
    else:
        metrics["k_sweep"] = {
            "enabled": False,
            "reason": f"当前分类器为 {args.classifier}，k-sweep 仅适用于 knn 分类。",
        }

    model_path = save_training_outputs(
        output_dir=output_dir,
        vocabulary=vocabulary,
        train_embeddings=train_embeddings,
        train_bovw_features=train_bovw_features,
        train_color_features=train_color_features,
        train_records=train_records,
        test_embeddings=test_embeddings,
        test_records=test_records,
        metrics=metrics,
        label_to_name=label_to_name,
        args=args,
        classifier_artifacts={
            name: artifact
            for name, artifact in {"svm": svm_classifier, "rf": rf_classifier}.items()
            if artifact is not None
        },
    )
    save_json({"examples": retrieval_examples}, output_dir / "retrieval_examples.json")

    print("训练完成。")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"模型已保存到: {model_path}")
    if k_sweep_metrics is not None:
        print(f"k-sweep 图表已保存到: {output_dir / 'k_sweep_accuracy_recall.png'}")


def run_predict(args) -> None:
    project_root = Path(__file__).resolve().parent
    image_path = resolve_path(project_root, args.image)
    model_path = resolve_path(project_root, args.model)

    bundle = joblib.load(model_path)
    config = bundle["config"]
    vocabulary = np.asarray(bundle["vocabulary"], dtype=np.float32)
    train_embeddings = np.asarray(bundle["train_embeddings"], dtype=np.float32)
    train_labels = np.asarray(bundle["train_labels"], dtype=np.int32)
    train_paths = bundle["train_paths"]
    label_to_name = {int(k): v for k, v in bundle["label_to_name"].items()}
    classifier_name = bundle.get("classifier_name", "knn")
    classifier_artifacts = bundle.get("classifier_artifacts", {})
    if not classifier_artifacts:
        legacy_artifact = bundle.get("classifier_artifact")
        if legacy_artifact is not None and classifier_name == "svm":
            classifier_artifacts = {"svm": legacy_artifact}

    namespace = argparse.Namespace(**config)
    preprocess_result = preprocess_image(
        image_path,
        max_side=namespace.max_side,
    )
    feature_bundle = extract_feature_bundle(
        preprocess_result.image_bgr,
        h_bins=namespace.h_bins,
        s_bins=namespace.s_bins,
        sift_nfeatures=namespace.sift_nfeatures,
        sift_contrast_threshold=namespace.sift_contrast_threshold,
        sift_edge_threshold=namespace.sift_edge_threshold,
        sift_sigma=namespace.sift_sigma,
        sift_backend=getattr(namespace, "sift_backend", "opencv"),
        cudasift_max_points=getattr(namespace, "cudasift_max_points", 32768),
        cudasift_num_octaves=getattr(namespace, "cudasift_num_octaves", 5),
        cudasift_lowest_scale=getattr(namespace, "cudasift_lowest_scale", 0.0),
        cudasift_upscale=getattr(namespace, "cudasift_upscale", False),
    )
    bovw = encode_bovw_batch(
        descriptor_sets=[feature_bundle.descriptors],
        keypoint_sets=[feature_bundle.keypoints],
        image_shapes=[feature_bundle.image_shape],
        vocabulary=vocabulary,
        min_weight=namespace.min_spatial_weight,
        max_weight=namespace.max_spatial_weight,
        backend=getattr(namespace, "bovw_backend", "numpy"),
        torch_device=getattr(namespace, "torch_device", "auto"),
        descriptor_chunk_size=getattr(namespace, "torch_descriptor_chunk_size", 8192),
    )[0]
    embedding = fuse_features(
        color_features=feature_bundle.color_hist,
        bovw_features=bovw,
        color_weight=namespace.color_weight,
        bovw_weight=namespace.bovw_weight,
    )
    knn_index = build_knn_index(
        train_embeddings=train_embeddings,
        metric=namespace.metric,
        backend=getattr(namespace, "knn_backend", "sklearn"),
        torch_device=getattr(namespace, "torch_device", "auto"),
        query_batch_size=getattr(namespace, "torch_query_batch_size", 256),
    )
    predicted_label = int(
        predict_labels_by_classifier(
            classifier_name=classifier_name,
            index=knn_index,
            train_labels=train_labels,
            query_embeddings=embedding,
            top_k=args.top_k,
            svm_classifier=classifier_artifacts.get("svm"),
            rf_classifier=classifier_artifacts.get("rf"),
        )[0]
    )
    prediction = build_prediction_payload(
        index=knn_index,
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        train_paths=train_paths,
        label_to_name=label_to_name,
        query_embedding=embedding,
        top_k=args.top_k,
        metric=namespace.metric,
        predicted_label=predicted_label,
    )
    print(json.dumps(prediction, ensure_ascii=False, indent=2))


def run_predict_batch(args) -> None:
    project_root = Path(__file__).resolve().parent
    image_dir = resolve_path(project_root, args.image_dir)
    model_path = resolve_path(project_root, args.model)
    output_path = resolve_path(project_root, args.output)

    if not image_dir.exists():
        raise FileNotFoundError(f"图片目录不存在: {image_dir}")

    image_paths = list_image_paths(image_dir, recursive=args.recursive)
    if not image_paths:
        raise ValueError(f"未在目录中找到可处理图片: {image_dir}")

    bundle = joblib.load(model_path)
    config = argparse.Namespace(**bundle["config"])
    vocabulary = np.asarray(bundle["vocabulary"], dtype=np.float32)
    train_embeddings = np.asarray(bundle["train_embeddings"], dtype=np.float32)
    train_labels = np.asarray(bundle["train_labels"], dtype=np.int32)
    train_paths = bundle["train_paths"]
    label_to_name = {int(k): v for k, v in bundle["label_to_name"].items()}
    classifier_name = bundle.get("classifier_name", "knn")
    classifier_artifacts = bundle.get("classifier_artifacts", {})

    print(f"批量推理图片数: {len(image_paths)}")
    query_records = extract_query_records(
        image_paths=image_paths,
        config=config,
        num_workers=args.num_workers,
        chunksize=args.chunksize,
    )
    predict_args = argparse.Namespace(**vars(config))
    predict_args.num_workers = args.num_workers
    predict_args.chunksize = args.chunksize
    query_embeddings = build_embeddings(query_records, vocabulary=vocabulary, args=predict_args)
    knn_index = build_knn_index(
        train_embeddings=train_embeddings,
        metric=config.metric,
        backend=getattr(config, "knn_backend", "sklearn"),
        torch_device=getattr(config, "torch_device", "auto"),
        query_batch_size=getattr(config, "torch_query_batch_size", 256),
    )
    distances, indices = retrieve_neighbors(knn_index, query_embeddings, top_k=args.top_k)
    predicted_labels = predict_labels_by_classifier(
        classifier_name=classifier_name,
        index=knn_index,
        train_labels=train_labels,
        query_embeddings=query_embeddings,
        top_k=args.top_k,
        svm_classifier=classifier_artifacts.get("svm"),
        rf_classifier=classifier_artifacts.get("rf"),
        precomputed_indices=indices,
    )

    predictions = []
    for image_path, predicted_label, row_distances, row_indices in zip(
        image_paths,
        predicted_labels,
        distances,
        indices,
    ):
        payload = build_prediction_payload(
            index=knn_index,
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            train_paths=train_paths,
            label_to_name=label_to_name,
            query_embedding=None,
            top_k=args.top_k,
            metric=config.metric,
            predicted_label=int(predicted_label),
            precomputed_distances=row_distances,
            precomputed_indices=row_indices,
        )
        predictions.append(
            {
                "image": str(image_path),
                **payload,
            }
        )

    output_payload = {"predictions": predictions}
    save_json(output_payload, output_path)
    print(f"批量推理结果已保存到: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assign2 校园植被识别与检索系统")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="训练并评估系统")
    train_parser.add_argument("--csv", default="dataset/dataset.csv")
    train_parser.add_argument("--output-dir", default="outputs")
    train_parser.add_argument("--test-size", type=float, default=0.3)
    train_parser.add_argument("--random-state", type=int, default=42)
    train_parser.add_argument("--max-side", type=int, default=960)
    train_parser.add_argument("--h-bins", type=int, default=30)
    train_parser.add_argument("--s-bins", type=int, default=32)
    train_parser.add_argument("--sift-nfeatures", type=int, default=0)
    train_parser.add_argument("--sift-contrast-threshold", type=float, default=0.04)
    train_parser.add_argument("--sift-edge-threshold", type=float, default=10.0)
    train_parser.add_argument("--sift-sigma", type=float, default=1.6)
    train_parser.add_argument("--sift-backend", choices=["auto", "opencv", "cudasift"], default="auto")
    train_parser.add_argument("--cudasift-max-points", type=int, default=32768)
    train_parser.add_argument("--cudasift-num-octaves", type=int, default=5)
    train_parser.add_argument("--cudasift-lowest-scale", type=float, default=0.0)
    train_parser.add_argument("--cudasift-upscale", action="store_true")
    train_parser.add_argument("--num-words", type=int, default=300)
    train_parser.add_argument("--kmeans-batch-size", type=int, default=2048)
    train_parser.add_argument("--kmeans-max-iter", type=int, default=100)
    train_parser.add_argument("--min-spatial-weight", type=float, default=0.1)
    train_parser.add_argument("--max-spatial-weight", type=float, default=1.5)
    train_parser.add_argument("--color-weight", type=float, default=1.0)
    train_parser.add_argument("--bovw-weight", type=float, default=1.0)
    train_parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    train_parser.add_argument(
        "--classifier",
        choices=["knn", "svm", "rf", "ensemble"],
        default="knn",
    )
    train_parser.add_argument("--top-k", type=int, default=5)
    train_parser.add_argument(
        "--eval-ks",
        default=None,
        help="用于扫描评估的 k 列表，例如 1,3,5,7；默认从 1 扫描到 --max-eval-k。",
    )
    train_parser.add_argument(
        "--max-eval-k",
        type=int,
        default=10,
        help="未指定 --eval-ks 时，评估 k 的最大值。",
    )
    train_parser.add_argument("--svm-kernel", choices=["linear", "rbf"], default="rbf")
    train_parser.add_argument("--svm-c", type=float, default=1.0)
    train_parser.add_argument(
        "--svm-gamma",
        default="scale",
        help="SVM 的 gamma 参数，支持 scale、auto 或具体数值字符串。",
    )
    train_parser.add_argument("--rf-n-estimators", type=int, default=300)
    train_parser.add_argument("--rf-max-depth", type=int, default=None)
    train_parser.add_argument("--rf-min-samples-leaf", type=int, default=1)
    train_parser.add_argument("--num-workers", type=int, default=default_num_workers())
    train_parser.add_argument("--chunksize", type=int, default=4)
    train_parser.add_argument("--bovw-backend", choices=["numpy", "torch"], default="numpy")
    train_parser.add_argument("--knn-backend", choices=["sklearn", "torch"], default="sklearn")
    train_parser.add_argument("--torch-device", choices=["auto", "cpu", "cuda"], default="auto")
    train_parser.add_argument("--torch-query-batch-size", type=int, default=256)
    train_parser.add_argument("--torch-descriptor-chunk-size", type=int, default=8192)
    train_parser.set_defaults(func=run_train)

    predict_parser = subparsers.add_parser("predict", help="加载模型并预测单张图片")
    predict_parser.add_argument("--image", required=True)
    predict_parser.add_argument("--model", default="outputs/model_bundle.joblib")
    predict_parser.add_argument("--top-k", type=int, default=5)
    predict_parser.set_defaults(func=run_predict)

    predict_batch_parser = subparsers.add_parser("predict-batch", help="批量预测目录中的图片")
    predict_batch_parser.add_argument("--image-dir", required=True)
    predict_batch_parser.add_argument("--model", default="outputs/model_bundle.joblib")
    predict_batch_parser.add_argument("--top-k", type=int, default=5)
    predict_batch_parser.add_argument("--output", default="outputs/predictions.json")
    predict_batch_parser.add_argument("--recursive", action="store_true")
    predict_batch_parser.add_argument("--num-workers", type=int, default=default_num_workers())
    predict_batch_parser.add_argument("--chunksize", type=int, default=4)
    predict_batch_parser.set_defaults(func=run_predict_batch)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
