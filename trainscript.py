from __future__ import annotations

import argparse
from pathlib import Path

from main import run_train


def resolve_csv_path(dataset_root: str | None, csv_path: str | None) -> str:
    if csv_path:
        return csv_path
    if dataset_root:
        return str((Path(dataset_root) / "dataset.csv").as_posix())
    return "dataset/dataset.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="针对指定数据集执行训练与测试的快捷脚本"
    )
    parser.add_argument(
        "--dataset-root",
        default="dataset",
        help="数据集目录路径，目录中应包含 dataset.csv 和图像文件",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="可直接指定 dataset.csv 路径；指定后优先于 --dataset-root",
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-side", type=int, default=960)
    parser.add_argument("--h-bins", type=int, default=30)
    parser.add_argument("--s-bins", type=int, default=32)
    parser.add_argument("--sift-nfeatures", type=int, default=0)
    parser.add_argument("--sift-contrast-threshold", type=float, default=0.04)
    parser.add_argument("--sift-edge-threshold", type=float, default=10.0)
    parser.add_argument("--sift-sigma", type=float, default=1.6)
    parser.add_argument("--sift-backend", choices=["auto", "opencv", "cudasift"], default="auto")
    parser.add_argument("--cudasift-max-points", type=int, default=32768)
    parser.add_argument("--cudasift-num-octaves", type=int, default=5)
    parser.add_argument("--cudasift-lowest-scale", type=float, default=0.0)
    parser.add_argument("--cudasift-upscale", action="store_true")
    parser.add_argument("--num-words", type=int, default=300)
    parser.add_argument("--kmeans-batch-size", type=int, default=2048)
    parser.add_argument("--kmeans-max-iter", type=int, default=100)
    parser.add_argument("--min-spatial-weight", type=float, default=0.1)
    parser.add_argument("--max-spatial-weight", type=float, default=1.5)
    parser.add_argument("--color-weight", type=float, default=1.0)
    parser.add_argument("--bovw-weight", type=float, default=1.0)
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--classifier", choices=["knn", "svm", "rf", "ensemble"], default="knn")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--eval-ks",
        default=None,
        help="用于扫描评估的 k 列表，例如 1,3,5,7；默认从 1 扫描到 --max-eval-k。",
    )
    parser.add_argument(
        "--max-eval-k",
        type=int,
        default=10,
        help="未指定 --eval-ks 时，评估 k 的最大值。",
    )
    parser.add_argument("--svm-kernel", choices=["linear", "rbf"], default="rbf")
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument(
        "--svm-gamma",
        default="scale",
        help="SVM 的 gamma 参数，支持 scale、auto 或具体数值字符串。",
    )
    parser.add_argument("--rf-n-estimators", type=int, default=300)
    parser.add_argument("--rf-max-depth", type=int, default=None)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--chunksize", type=int, default=4)
    parser.add_argument("--bovw-backend", choices=["numpy", "torch"], default="numpy")
    parser.add_argument("--knn-backend", choices=["sklearn", "torch"], default="sklearn")
    parser.add_argument("--torch-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--torch-query-batch-size", type=int, default=256)
    parser.add_argument("--torch-descriptor-chunk-size", type=int, default=8192)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.csv = resolve_csv_path(args.dataset_root, args.csv)
    run_train(args)


if __name__ == "__main__":
    main()
