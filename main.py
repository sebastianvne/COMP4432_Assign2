from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import joblib
import matplotlib
import numpy as np
from tqdm.auto import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.bovw import encode_bovw_batch, encode_weighted_bovw, save_artifact, train_visual_vocabulary
from src.dataset import build_label_to_name, load_samples, stratified_split, summarize_labels
from src.features import extract_feature_bundle
from src.preprocess import preprocess_image
from src.train_eval import (
    build_prediction_payload,
    evaluate_classification,
    evaluate_retrieval,
    fit_knn_index,
    fit_svm_classifier,
    fuse_features,
    save_json,
)


def resolve_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (project_root / path).resolve()


def extract_records(samples, args) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for sample in tqdm(samples, desc="提取图像特征", unit="img"):
        preprocess_result = preprocess_image(
            sample.path,
            max_side=args.max_side,
            grabcut_margin_ratio=args.grabcut_margin_ratio,
            grabcut_iter_count=args.grabcut_iter_count,
        )
        feature_bundle = extract_feature_bundle(
            preprocess_result.image_bgr,
            foreground_mask=preprocess_result.foreground_mask,
            h_bins=args.h_bins,
            s_bins=args.s_bins,
            sift_nfeatures=args.sift_nfeatures,
            sift_contrast_threshold=args.sift_contrast_threshold,
            sift_edge_threshold=args.sift_edge_threshold,
            sift_sigma=args.sift_sigma,
        )
        records.append(
            {
                "sample": sample,
                "color_hist": feature_bundle.color_hist,
                "keypoints": feature_bundle.keypoints,
                "descriptors": feature_bundle.descriptors,
                "image_shape": feature_bundle.image_shape,
            }
        )
    return records


def build_embeddings(records, vocabulary: np.ndarray, args) -> np.ndarray:
    color_features = np.vstack([record["color_hist"] for record in records]).astype(np.float32)
    bovw_features = encode_bovw_batch(
        descriptor_sets=[record["descriptors"] for record in records],
        keypoint_sets=[record["keypoints"] for record in records],
        image_shapes=[record["image_shape"] for record in records],
        vocabulary=vocabulary,
        min_weight=args.min_spatial_weight,
        max_weight=args.max_spatial_weight,
    )
    embeddings = fuse_features(
        color_features=color_features,
        bovw_features=bovw_features,
        color_weight=args.color_weight,
        bovw_weight=args.bovw_weight,
    )
    return embeddings


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
    classifier_artifact=None,
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
        "classifier_artifact": classifier_artifact,
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
    train_records = extract_records(train_samples, args)
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
    train_bovw_features = encode_bovw_batch(
        descriptor_sets=[record["descriptors"] for record in train_records],
        keypoint_sets=[record["keypoints"] for record in train_records],
        image_shapes=[record["image_shape"] for record in train_records],
        vocabulary=vocabulary,
        min_weight=args.min_spatial_weight,
        max_weight=args.max_spatial_weight,
    )
    train_color_features = np.vstack([record["color_hist"] for record in train_records]).astype(np.float32)
    train_embeddings = fuse_features(
        color_features=train_color_features,
        bovw_features=train_bovw_features,
        color_weight=args.color_weight,
        bovw_weight=args.bovw_weight,
    )

    print("正在提取测试集特征...")
    test_records = extract_records(test_samples, args)
    test_bovw_features = encode_bovw_batch(
        descriptor_sets=[record["descriptors"] for record in test_records],
        keypoint_sets=[record["keypoints"] for record in test_records],
        image_shapes=[record["image_shape"] for record in test_records],
        vocabulary=vocabulary,
        min_weight=args.min_spatial_weight,
        max_weight=args.max_spatial_weight,
    )
    test_color_features = np.vstack([record["color_hist"] for record in test_records]).astype(np.float32)
    test_embeddings = fuse_features(
        color_features=test_color_features,
        bovw_features=test_bovw_features,
        color_weight=args.color_weight,
        bovw_weight=args.bovw_weight,
    )

    print("正在建立 KNN 检索器并评估...")
    knn_index = fit_knn_index(train_embeddings, metric=args.metric)
    train_labels = [record["sample"].label for record in train_records]
    test_labels = [record["sample"].label for record in test_records]

    svm_classifier = None
    if args.classifier == "svm":
        print("正在训练 SVM 分类器...")
        svm_classifier = fit_svm_classifier(
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            kernel=args.svm_kernel,
            c_value=args.svm_c,
            gamma=args.svm_gamma,
        )

    classification_metrics = evaluate_classification(
        classifier_name=args.classifier,
        index=knn_index,
        train_labels=train_labels,
        test_embeddings=test_embeddings,
        test_labels=test_labels,
        top_k=args.top_k,
        svm_classifier=svm_classifier,
    )
    retrieval_metrics = evaluate_retrieval(
        index=knn_index,
        train_labels=train_labels,
        test_embeddings=test_embeddings,
        test_labels=test_labels,
        top_ks=(1, min(5, args.top_k)),
    )
    k_sweep_metrics = None
    if args.classifier == "knn":
        eval_ks = parse_eval_ks(
            raw_value=args.eval_ks,
            max_allowed_k=min(args.max_eval_k, len(train_labels)),
        )
        k_sweep_metrics = evaluate_k_sweep(
            index=knn_index,
            train_labels=train_labels,
            test_embeddings=test_embeddings,
            test_labels=test_labels,
            eval_ks=eval_ks,
        )
        save_json(k_sweep_metrics, output_dir / "k_sweep_metrics.json")
        save_k_sweep_csv(k_sweep_metrics["per_k"], output_dir / "k_sweep_metrics.csv")
        plot_k_sweep(k_sweep_metrics["per_k"], output_dir / "k_sweep_accuracy_recall.png")

    retrieval_examples = []
    for record, query_embedding in zip(test_records, test_embeddings):
        predicted_label = None
        if args.classifier == "svm" and svm_classifier is not None:
            predicted_label = int(
                svm_classifier.predict(np.asarray(query_embedding, dtype=np.float32)[None, :])[0]
            )
        prediction = build_prediction_payload(
            index=knn_index,
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            train_paths=[sample["sample"].relative_path for sample in train_records],
            label_to_name=label_to_name,
            query_embedding=query_embedding,
            top_k=args.top_k,
            metric=args.metric,
            predicted_label=predicted_label,
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
            "reason": "当前分类器为 svm，k-sweep 仅适用于 knn 分类。",
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
        classifier_artifact=svm_classifier,
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
    classifier_artifact = bundle.get("classifier_artifact")

    namespace = argparse.Namespace(**config)
    preprocess_result = preprocess_image(
        image_path,
        max_side=namespace.max_side,
        grabcut_margin_ratio=namespace.grabcut_margin_ratio,
        grabcut_iter_count=namespace.grabcut_iter_count,
    )
    feature_bundle = extract_feature_bundle(
        preprocess_result.image_bgr,
        foreground_mask=preprocess_result.foreground_mask,
        h_bins=namespace.h_bins,
        s_bins=namespace.s_bins,
        sift_nfeatures=namespace.sift_nfeatures,
        sift_contrast_threshold=namespace.sift_contrast_threshold,
        sift_edge_threshold=namespace.sift_edge_threshold,
        sift_sigma=namespace.sift_sigma,
    )
    bovw = encode_weighted_bovw(
        descriptors=feature_bundle.descriptors,
        keypoints=feature_bundle.keypoints,
        image_shape=feature_bundle.image_shape,
        vocabulary=vocabulary,
        min_weight=namespace.min_spatial_weight,
        max_weight=namespace.max_spatial_weight,
    )
    embedding = fuse_features(
        color_features=feature_bundle.color_hist,
        bovw_features=bovw,
        color_weight=namespace.color_weight,
        bovw_weight=namespace.bovw_weight,
    )
    knn_index = fit_knn_index(train_embeddings, metric=namespace.metric)
    predicted_label = None
    if classifier_name == "svm" and classifier_artifact is not None:
        predicted_label = int(classifier_artifact.predict(np.asarray(embedding, dtype=np.float32))[0])
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assign2 校园植被识别与检索系统")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="训练并评估系统")
    train_parser.add_argument("--csv", default="dataset/dataset.csv")
    train_parser.add_argument("--output-dir", default="outputs")
    train_parser.add_argument("--test-size", type=float, default=0.3)
    train_parser.add_argument("--random-state", type=int, default=42)
    train_parser.add_argument("--max-side", type=int, default=960)
    train_parser.add_argument("--grabcut-margin-ratio", type=float, default=0.1)
    train_parser.add_argument("--grabcut-iter-count", type=int, default=5)
    train_parser.add_argument("--h-bins", type=int, default=30)
    train_parser.add_argument("--s-bins", type=int, default=32)
    train_parser.add_argument("--sift-nfeatures", type=int, default=0)
    train_parser.add_argument("--sift-contrast-threshold", type=float, default=0.04)
    train_parser.add_argument("--sift-edge-threshold", type=float, default=10.0)
    train_parser.add_argument("--sift-sigma", type=float, default=1.6)
    train_parser.add_argument("--num-words", type=int, default=300)
    train_parser.add_argument("--kmeans-batch-size", type=int, default=2048)
    train_parser.add_argument("--kmeans-max-iter", type=int, default=100)
    train_parser.add_argument("--min-spatial-weight", type=float, default=0.1)
    train_parser.add_argument("--max-spatial-weight", type=float, default=1.5)
    train_parser.add_argument("--color-weight", type=float, default=1.0)
    train_parser.add_argument("--bovw-weight", type=float, default=1.0)
    train_parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    train_parser.add_argument("--classifier", choices=["knn", "svm"], default="knn")
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
    train_parser.set_defaults(func=run_train)

    predict_parser = subparsers.add_parser("predict", help="加载模型并预测单张图片")
    predict_parser.add_argument("--image", required=True)
    predict_parser.add_argument("--model", default="outputs/model_bundle.joblib")
    predict_parser.add_argument("--top-k", type=int, default=5)
    predict_parser.set_defaults(func=run_predict)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
