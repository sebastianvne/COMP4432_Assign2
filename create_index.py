from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Tuple

from tqdm.auto import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}


def list_class_dirs(dataset_root: Path) -> List[Path]:
    return sorted([path for path in dataset_root.iterdir() if path.is_dir()], key=lambda p: p.name.lower())


def collect_rows(project_root: Path, dataset_root: Path) -> Tuple[List[dict], List[Tuple[int, str, int]]]:
    rows: List[dict] = []
    summary: List[Tuple[int, str, int]] = []
    class_dirs = list_class_dirs(dataset_root)

    label = 0
    for class_dir in tqdm(class_dirs, desc="扫描类别目录", unit="class"):
        image_paths = sorted(
            [path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
            key=lambda p: p.name.lower(),
        )
        if not image_paths:
            continue

        summary.append((label, class_dir.name, len(image_paths)))

        for image_path in image_paths:
            relative_path = image_path.relative_to(project_root).as_posix()
            rows.append(
                {
                    "path": f"./{relative_path}",
                    "label": label,
                    "cn_name": class_dir.name,
                }
            )
        label += 1

    return rows, summary


def write_index(csv_path: Path, rows: Iterable[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["path", "label", "cn_name"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="根据子文件夹生成数据集 index.csv")
    parser.add_argument("--dataset-root", default="dataset_large", help="按类别子目录组织的数据集根目录")
    parser.add_argument("--output-csv", default=None, help="输出 csv 路径，默认写到 <dataset-root>/index.csv")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    dataset_root = (project_root / args.dataset_root).resolve()
    output_csv = (project_root / args.output_csv).resolve() if args.output_csv else dataset_root / "index.csv"

    if not dataset_root.exists():
        raise FileNotFoundError(f"数据集目录不存在: {dataset_root}")

    rows, summary = collect_rows(project_root=project_root, dataset_root=dataset_root)
    if not rows:
        raise ValueError("没有扫描到任何图片，无法生成 index.csv。")

    write_index(output_csv, rows)
    print(f"已生成索引文件: {output_csv}")
    print("类别统计:")
    for label, class_name, count in summary:
        print(f"  label={label:02d} | class={class_name} | images={count}")
    print(f"总图片数: {len(rows)}")


if __name__ == "__main__":
    main()
