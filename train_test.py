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
    parser.add_argument("--grabcut-margin-ratio", type=float, default=0.1)
    parser.add_argument("--grabcut-iter-count", type=int, default=5)
    parser.add_argument("--h-bins", type=int, default=30)
    parser.add_argument("--s-bins", type=int, default=32)
    parser.add_argument("--sift-nfeatures", type=int, default=0)
    parser.add_argument("--sift-contrast-threshold", type=float, default=0.04)
    parser.add_argument("--sift-edge-threshold", type=float, default=10.0)
    parser.add_argument("--sift-sigma", type=float, default=1.6)
    parser.add_argument("--num-words", type=int, default=300)
    parser.add_argument("--kmeans-batch-size", type=int, default=2048)
    parser.add_argument("--kmeans-max-iter", type=int, default=100)
    parser.add_argument("--min-spatial-weight", type=float, default=0.1)
    parser.add_argument("--max-spatial-weight", type=float, default=1.5)
    parser.add_argument("--color-weight", type=float, default=1.0)
    parser.add_argument("--bovw-weight", type=float, default=1.0)
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--top-k", type=int, default=5)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.csv = resolve_csv_path(args.dataset_root, args.csv)
    run_train(args)


if __name__ == "__main__":
    main()
