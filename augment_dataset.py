from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np
from tqdm.auto import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}


def list_class_dirs(dataset_root: Path) -> List[Path]:
    return sorted(
        [path for path in dataset_root.iterdir() if path.is_dir() and path.name != ".cache"],
        key=lambda path: path.name.lower(),
    )


def list_images(class_dir: Path) -> List[Path]:
    return sorted(
        [path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda path: path.name.lower(),
    )


def load_image_bgr(image_path: Path) -> np.ndarray:
    buffer = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    return image


def save_image_bgr(image: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(output_path.suffix or ".jpg", image)
    if not success:
        raise ValueError(f"无法编码图片: {output_path}")
    encoded.tofile(str(output_path))


def make_image_rng(seed: int, image_path: Path) -> np.random.Generator:
    digest = hashlib.sha1(f"{seed}:{image_path.as_posix()}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big", signed=False)
    return np.random.default_rng(value)


def flip_image(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.flip(image_bgr, 1)


def rotate_image(
    image_bgr: np.ndarray,
    rng: np.random.Generator,
    max_abs_angle_degrees: float = 20.0,
) -> np.ndarray:
    angle_degrees = float(rng.uniform(-max_abs_angle_degrees, max_abs_angle_degrees))
    height, width = image_bgr.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    return cv2.warpAffine(
        image_bgr,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def scale_crop_image(
    image_bgr: np.ndarray,
    rng: np.random.Generator,
    min_scale_factor: float = 1.05,
    max_scale_factor: float = 1.25,
) -> np.ndarray:
    scale_factor = float(rng.uniform(min_scale_factor, max_scale_factor))
    height, width = image_bgr.shape[:2]
    scaled = cv2.resize(
        image_bgr,
        (max(1, int(width * scale_factor)), max(1, int(height * scale_factor))),
        interpolation=cv2.INTER_LINEAR,
    )
    scaled_height, scaled_width = scaled.shape[:2]
    max_offset_y = max(scaled_height - height, 0)
    max_offset_x = max(scaled_width - width, 0)
    start_y = int(rng.integers(0, max_offset_y + 1)) if max_offset_y > 0 else 0
    start_x = int(rng.integers(0, max_offset_x + 1)) if max_offset_x > 0 else 0
    return scaled[start_y : start_y + height, start_x : start_x + width]


def adjust_color_properties(
    image_bgr: np.ndarray,
    rng: np.random.Generator,
    brightness_delta: float = 0.12,
    contrast_delta: float = 0.12,
    saturation_delta: float = 0.15,
    hue_delta: int = 10,
) -> np.ndarray:
    image = image_bgr.astype(np.float32) / 255.0

    contrast = float(rng.uniform(1.0 - contrast_delta, 1.0 + contrast_delta))
    brightness = float(rng.uniform(-brightness_delta, brightness_delta))
    image = np.clip(image * contrast + brightness, 0.0, 1.0)

    hsv = cv2.cvtColor((image * 255.0).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= float(rng.uniform(1.0 - saturation_delta, 1.0 + saturation_delta))
    hsv[..., 1] = np.clip(hsv[..., 1], 0.0, 255.0)
    hsv[..., 0] = (hsv[..., 0] + float(rng.integers(-hue_delta, hue_delta + 1))) % 180.0

    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def inject_noise(
    image_bgr: np.ndarray,
    rng: np.random.Generator,
    gaussian_sigma: float = 12.0,
    salt_pepper_ratio: float = 0.002,
) -> np.ndarray:
    noisy = image_bgr.astype(np.float32)
    gaussian_noise = rng.normal(0.0, gaussian_sigma, size=image_bgr.shape).astype(np.float32)
    noisy = np.clip(noisy + gaussian_noise, 0.0, 255.0)

    output = noisy.astype(np.uint8)
    total_pixels = image_bgr.shape[0] * image_bgr.shape[1]
    num_salt = int(total_pixels * salt_pepper_ratio)
    num_pepper = int(total_pixels * salt_pepper_ratio)

    if num_salt > 0:
        ys = rng.integers(0, image_bgr.shape[0], size=num_salt)
        xs = rng.integers(0, image_bgr.shape[1], size=num_salt)
        output[ys, xs] = 255
    if num_pepper > 0:
        ys = rng.integers(0, image_bgr.shape[0], size=num_pepper)
        xs = rng.integers(0, image_bgr.shape[1], size=num_pepper)
        output[ys, xs] = 0

    return output


def write_augmented_images(
    image_path: Path,
    output_class_dir: Path,
    seed: int,
) -> int:
    image_bgr = load_image_bgr(image_path)
    rng = make_image_rng(seed, image_path)

    stem = image_path.stem
    suffix = image_path.suffix.lower() or ".jpg"

    outputs = {
        stem: image_bgr,
        f"{stem}_flip": flip_image(image_bgr),
        f"{stem}_rotate": rotate_image(image_bgr, rng),
        f"{stem}_scale_crop": scale_crop_image(image_bgr, rng),
        f"{stem}_color_jitter": adjust_color_properties(image_bgr, rng),
        f"{stem}_noise": inject_noise(image_bgr, rng),
    }

    for name, image in outputs.items():
        save_image_bgr(image, output_class_dir / f"{name}{suffix}")

    return len(outputs)


def augment_dataset(dataset_root: Path, output_root: Path, seed: int) -> None:
    if not dataset_root.exists():
        raise FileNotFoundError(f"数据集目录不存在: {dataset_root}")

    class_dirs = list_class_dirs(dataset_root)
    if not class_dirs:
        raise ValueError(f"未在 {dataset_root} 中找到任何类别子目录。")

    total_written = 0
    total_inputs = 0
    for class_dir in tqdm(class_dirs, desc="增强类别", unit="class"):
        images = list_images(class_dir)
        if not images:
            continue

        output_class_dir = output_root / class_dir.name
        for image_path in tqdm(images, desc=f"处理 {class_dir.name}", unit="img", leave=False):
            total_inputs += 1
            total_written += write_augmented_images(
                image_path=image_path,
                output_class_dir=output_class_dir,
                seed=seed,
            )

    print(f"输入图片数: {total_inputs}")
    print(f"输出图片数: {total_written}")
    print(f"增强结果已保存到: {output_root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="对数据集图片执行常见数据增强并输出到新目录")
    parser.add_argument("--dataset-root", default="dataset_large", help="原始数据集目录")
    parser.add_argument("--output-root", default="dataset_augmented", help="增强输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机增强种子")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    dataset_root = (project_root / args.dataset_root).resolve()
    output_root = (project_root / args.output_root).resolve()

    augment_dataset(
        dataset_root=dataset_root,
        output_root=output_root,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
