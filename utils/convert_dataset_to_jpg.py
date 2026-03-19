from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from tqdm.auto import tqdm


register_heif_opener()

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}


@dataclass
class ConversionStats:
    converted: int = 0
    copied: int = 0
    skipped: int = 0


def list_image_files(dataset_root: Path) -> List[Path]:
    return sorted(
        [
            path
            for path in dataset_root.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ],
        key=lambda p: p.as_posix().lower(),
    )


def make_unique_target_path(target_path: Path) -> Path:
    if not target_path.exists():
        return target_path

    stem = target_path.stem
    suffix = target_path.suffix
    parent = target_path.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def convert_image_to_jpg(source_path: Path, target_path: Path, quality: int) -> None:
    with Image.open(source_path) as image:
        image = ImageOps.exif_transpose(image)
        rgb_image = image.convert("RGB")
        rgb_image.save(target_path, format="JPEG", quality=quality, optimize=True)


def convert_dataset(
    source_root: Path,
    target_root: Path,
    overwrite: bool = False,
    quality: int = 95,
) -> ConversionStats:
    stats = ConversionStats()
    image_files = list_image_files(source_root)
    if not image_files:
        raise ValueError(f"在 {source_root} 中没有找到可转换图片。")

    for source_path in tqdm(image_files, desc="转换为 JPG", unit="img"):
        relative_path = source_path.relative_to(source_root)
        target_dir = target_root / relative_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        target_name = f"{source_path.stem}.jpg"
        target_path = target_dir / target_name
        if not overwrite:
            target_path = make_unique_target_path(target_path)

        if target_path.exists() and not overwrite:
            stats.skipped += 1
            continue

        source_ext = source_path.suffix.lower()
        if source_ext in {".jpg", ".jpeg"} and source_path.suffix == ".jpg" and overwrite:
            shutil.copy2(source_path, target_path)
            stats.copied += 1
            continue

        if source_ext in {".jpg", ".jpeg"} and source_path.stem == target_path.stem and source_path.suffix.lower() == ".jpg":
            if overwrite:
                shutil.copy2(source_path, target_path)
                stats.copied += 1
            else:
                convert_image_to_jpg(source_path, target_path, quality=quality)
                stats.converted += 1
            continue

        convert_image_to_jpg(source_path, target_path, quality=quality)
        stats.converted += 1

    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将数据集中的全部图片统一转换为 JPG。")
    parser.add_argument("--source-root", default="dataset_large", help="原始数据集根目录")
    parser.add_argument("--target-root", default="dataset_large_jpg", help="输出 JPG 数据集根目录")
    parser.add_argument("--quality", type=int, default=95, help="JPG 保存质量，默认 95")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖目标文件")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    source_root = (project_root / args.source_root).resolve()
    target_root = (project_root / args.target_root).resolve()

    if not source_root.exists():
        raise FileNotFoundError(f"源数据集目录不存在: {source_root}")

    stats = convert_dataset(
        source_root=source_root,
        target_root=target_root,
        overwrite=args.overwrite,
        quality=args.quality,
    )

    print(f"源目录: {source_root}")
    print(f"目标目录: {target_root}")
    print(f"转换完成: converted={stats.converted}, copied={stats.copied}, skipped={stats.skipped}")


if __name__ == "__main__":
    main()
