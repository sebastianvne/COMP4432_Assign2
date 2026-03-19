from __future__ import annotations

import argparse
import hashlib
import math
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List


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


import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
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


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求使用 CUDA，但当前环境不可用。")
    return torch.device(requested_device)


def make_image_rng(seed: int, image_path: Path) -> np.random.Generator:
    digest = hashlib.sha1(f"{seed}:{image_path.as_posix()}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big", signed=False)
    return np.random.default_rng(value)


def load_image_rgb_np(image_path: Path) -> np.ndarray:
    buffer = np.fromfile(str(image_path), dtype=np.uint8)
    image_bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"无法读取图片: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_image_rgb_tensor(image_path: Path, device: torch.device) -> torch.Tensor:
    image_rgb = load_image_rgb_np(image_path)
    return (
        torch.from_numpy(image_rgb)
        .to(device=device, dtype=torch.float32)
        .permute(2, 0, 1)
        .div_(255.0)
    )


def save_image_rgb_tensor(image_rgb: torch.Tensor, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_uint8 = (
        image_rgb.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(dtype=torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(output_path.suffix or ".jpg", image_bgr)
    if not success:
        raise ValueError(f"无法编码图片: {output_path}")
    encoded.tofile(str(output_path))


def flip_image(image_rgb: torch.Tensor) -> torch.Tensor:
    return torch.flip(image_rgb, dims=[2])


def flip_batch(images_rgb: torch.Tensor) -> torch.Tensor:
    return torch.flip(images_rgb, dims=[3])


def rotate_image(
    image_rgb: torch.Tensor,
    rng: np.random.Generator,
    max_abs_angle_degrees: float = 20.0,
) -> torch.Tensor:
    angle_degrees = float(rng.uniform(-max_abs_angle_degrees, max_abs_angle_degrees))
    angle_radians = math.radians(angle_degrees)
    cos_value = math.cos(angle_radians)
    sin_value = math.sin(angle_radians)

    theta = torch.tensor(
        [[cos_value, -sin_value, 0.0], [sin_value, cos_value, 0.0]],
        dtype=image_rgb.dtype,
        device=image_rgb.device,
    ).unsqueeze(0)

    batch_image = image_rgb.unsqueeze(0)
    grid = F.affine_grid(theta, size=batch_image.shape, align_corners=False)
    rotated = F.grid_sample(
        batch_image,
        grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=False,
    )
    return rotated.squeeze(0)


def rotate_batch(
    images_rgb: torch.Tensor,
    rngs: List[np.random.Generator],
    max_abs_angle_degrees: float = 20.0,
) -> torch.Tensor:
    angles_degrees = [float(rng.uniform(-max_abs_angle_degrees, max_abs_angle_degrees)) for rng in rngs]
    angles_radians = torch.tensor(
        [math.radians(angle) for angle in angles_degrees],
        dtype=images_rgb.dtype,
        device=images_rgb.device,
    )
    cos_values = torch.cos(angles_radians)
    sin_values = torch.sin(angles_radians)

    theta = torch.zeros((images_rgb.shape[0], 2, 3), dtype=images_rgb.dtype, device=images_rgb.device)
    theta[:, 0, 0] = cos_values
    theta[:, 0, 1] = -sin_values
    theta[:, 1, 0] = sin_values
    theta[:, 1, 1] = cos_values

    grid = F.affine_grid(theta, size=images_rgb.shape, align_corners=False)
    return F.grid_sample(
        images_rgb,
        grid,
        mode="bilinear",
        padding_mode="reflection",
        align_corners=False,
    )


def scale_crop_image(
    image_rgb: torch.Tensor,
    rng: np.random.Generator,
    min_scale_factor: float = 1.05,
    max_scale_factor: float = 1.25,
) -> torch.Tensor:
    scale_factor = float(rng.uniform(min_scale_factor, max_scale_factor))
    _, height, width = image_rgb.shape
    scaled_height = max(1, int(height * scale_factor))
    scaled_width = max(1, int(width * scale_factor))

    scaled = F.interpolate(
        image_rgb.unsqueeze(0),
        size=(scaled_height, scaled_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    max_offset_y = max(scaled_height - height, 0)
    max_offset_x = max(scaled_width - width, 0)
    start_y = int(rng.integers(0, max_offset_y + 1)) if max_offset_y > 0 else 0
    start_x = int(rng.integers(0, max_offset_x + 1)) if max_offset_x > 0 else 0
    return scaled[:, start_y : start_y + height, start_x : start_x + width]


def scale_crop_batch(
    images_rgb: torch.Tensor,
    rngs: List[np.random.Generator],
    min_scale_factor: float = 1.05,
    max_scale_factor: float = 1.25,
) -> torch.Tensor:
    outputs = [
        scale_crop_image(
            image_rgb=image_rgb,
            rng=rng,
            min_scale_factor=min_scale_factor,
            max_scale_factor=max_scale_factor,
        )
        for image_rgb, rng in zip(images_rgb, rngs)
    ]
    return torch.stack(outputs, dim=0)


def adjust_color_properties(
    image_rgb: torch.Tensor,
    rng: np.random.Generator,
    brightness_delta: float = 0.12,
    contrast_delta: float = 0.12,
    saturation_delta: float = 0.15,
    hue_delta: int = 10,
) -> torch.Tensor:
    contrast = float(rng.uniform(1.0 - contrast_delta, 1.0 + contrast_delta))
    brightness = float(rng.uniform(-brightness_delta, brightness_delta))
    saturation = float(rng.uniform(1.0 - saturation_delta, 1.0 + saturation_delta))
    hue_shift = float(rng.integers(-hue_delta, hue_delta + 1)) / 180.0

    adjusted = torch.clamp(image_rgb * contrast + brightness, 0.0, 1.0)
    adjusted = TF.adjust_saturation(adjusted, saturation)
    adjusted = TF.adjust_hue(adjusted, hue_shift)
    return adjusted.clamp(0.0, 1.0)


def adjust_color_properties_batch(
    images_rgb: torch.Tensor,
    rngs: List[np.random.Generator],
    brightness_delta: float = 0.12,
    contrast_delta: float = 0.12,
    saturation_delta: float = 0.15,
    hue_delta: int = 10,
) -> torch.Tensor:
    outputs = [
        adjust_color_properties(
            image_rgb=image_rgb,
            rng=rng,
            brightness_delta=brightness_delta,
            contrast_delta=contrast_delta,
            saturation_delta=saturation_delta,
            hue_delta=hue_delta,
        )
        for image_rgb, rng in zip(images_rgb, rngs)
    ]
    return torch.stack(outputs, dim=0)


def inject_noise(
    image_rgb: torch.Tensor,
    rng: np.random.Generator,
    gaussian_sigma: float = 12.0,
    salt_pepper_ratio: float = 0.002,
) -> torch.Tensor:
    gaussian_noise = rng.normal(0.0, gaussian_sigma / 255.0, size=image_rgb.shape).astype(np.float32)
    output = torch.clamp(
        image_rgb + torch.from_numpy(gaussian_noise).to(device=image_rgb.device, dtype=image_rgb.dtype),
        0.0,
        1.0,
    )

    _, height, width = image_rgb.shape
    total_pixels = height * width
    num_salt = int(total_pixels * salt_pepper_ratio)
    num_pepper = int(total_pixels * salt_pepper_ratio)

    if num_salt > 0:
        ys = torch.from_numpy(rng.integers(0, height, size=num_salt, dtype=np.int64)).to(image_rgb.device)
        xs = torch.from_numpy(rng.integers(0, width, size=num_salt, dtype=np.int64)).to(image_rgb.device)
        output[:, ys, xs] = 1.0
    if num_pepper > 0:
        ys = torch.from_numpy(rng.integers(0, height, size=num_pepper, dtype=np.int64)).to(image_rgb.device)
        xs = torch.from_numpy(rng.integers(0, width, size=num_pepper, dtype=np.int64)).to(image_rgb.device)
        output[:, ys, xs] = 0.0

    return output


def inject_noise_batch(
    images_rgb: torch.Tensor,
    rngs: List[np.random.Generator],
    gaussian_sigma: float = 12.0,
    salt_pepper_ratio: float = 0.002,
) -> torch.Tensor:
    outputs = [
        inject_noise(
            image_rgb=image_rgb,
            rng=rng,
            gaussian_sigma=gaussian_sigma,
            salt_pepper_ratio=salt_pepper_ratio,
        )
        for image_rgb, rng in zip(images_rgb, rngs)
    ]
    return torch.stack(outputs, dim=0)


def save_image_batch(
    images_rgb: Sequence[torch.Tensor],
    output_paths: Sequence[Path],
    save_workers: int,
) -> None:
    save_workers = max(1, int(save_workers))
    if save_workers == 1:
        for image_rgb, output_path in zip(images_rgb, output_paths):
            save_image_rgb_tensor(image_rgb, output_path)
        return

    with ThreadPoolExecutor(max_workers=min(save_workers, len(output_paths))) as executor:
        futures = [
            executor.submit(save_image_rgb_tensor, image_rgb, output_path)
            for image_rgb, output_path in zip(images_rgb, output_paths)
        ]
        for future in futures:
            future.result()


def build_shape_batches(image_paths: List[Path], batch_size: int) -> List[List[Path]]:
    buckets: "OrderedDict[tuple[int, int], List[Path]]" = OrderedDict()
    for image_path in image_paths:
        height, width = load_image_rgb_np(image_path).shape[:2]
        buckets.setdefault((height, width), []).append(image_path)

    batches: List[List[Path]] = []
    for bucket_paths in buckets.values():
        for start in range(0, len(bucket_paths), batch_size):
            batches.append(bucket_paths[start : start + batch_size])
    return batches


def write_augmented_batch(
    image_paths: List[Path],
    output_class_dir: Path,
    seed: int,
    device: torch.device,
    save_workers: int,
) -> int:
    image_batch = torch.stack([load_image_rgb_tensor(image_path, device=device) for image_path in image_paths], dim=0)
    rngs = [make_image_rng(seed, image_path) for image_path in image_paths]

    outputs = [
        image_batch,
        flip_batch(image_batch),
        rotate_batch(image_batch, rngs),
        scale_crop_batch(image_batch, rngs),
        adjust_color_properties_batch(image_batch, rngs),
        inject_noise_batch(image_batch, rngs),
    ]
    suffix_names = ["", "_flip", "_rotate", "_scale_crop", "_color_jitter", "_noise"]

    save_images: List[torch.Tensor] = []
    save_paths: List[Path] = []
    for batch_images, suffix_name in zip(outputs, suffix_names):
        for image_path, image_rgb in zip(image_paths, batch_images):
            stem = image_path.stem
            suffix = image_path.suffix.lower() or ".jpg"
            save_images.append(image_rgb)
            save_paths.append(output_class_dir / f"{stem}{suffix_name}{suffix}")

    save_image_batch(save_images, save_paths, save_workers=save_workers)
    return len(save_paths)


def augment_dataset(
    dataset_root: Path,
    output_root: Path,
    seed: int,
    device: torch.device,
    save_workers: int,
    batch_size: int,
) -> None:
    if not dataset_root.exists():
        raise FileNotFoundError(f"数据集目录不存在: {dataset_root}")

    class_dirs = list_class_dirs(dataset_root)
    if not class_dirs:
        raise ValueError(f"未在 {dataset_root} 中找到任何类别子目录。")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    total_written = 0
    total_inputs = 0
    with torch.inference_mode():
        for class_dir in tqdm(class_dirs, desc="增强类别", unit="class"):
            images = list_images(class_dir)
            if not images:
                continue

            output_class_dir = output_root / class_dir.name
            image_batches = build_shape_batches(images, batch_size=batch_size)
            for image_batch in tqdm(image_batches, desc=f"处理 {class_dir.name}", unit="batch", leave=False):
                total_inputs += len(image_batch)
                total_written += write_augmented_batch(
                    image_paths=image_batch,
                    output_class_dir=output_class_dir,
                    seed=seed,
                    device=device,
                    save_workers=save_workers,
                )

    print(f"增强设备: {device}")
    print(f"输入图片数: {total_inputs}")
    print(f"输出图片数: {total_written}")
    print(f"增强结果已保存到: {output_root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="对数据集图片执行常见数据增强并输出到新目录")
    parser.add_argument("--dataset-root", default="dataset_large", help="原始数据集目录")
    parser.add_argument("--output-root", default="dataset_augmented", help="增强输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机增强种子")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="增强计算设备，auto 会优先使用 CUDA",
    )
    parser.add_argument("--save-workers", type=int, default=4, help="并发保存增强结果的线程数")
    parser.add_argument("--batch-size", type=int, default=16, help="GPU 侧增强批大小（按相同尺寸分桶后处理）")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    dataset_root = (project_root / args.dataset_root).resolve()
    output_root = (project_root / args.output_root).resolve()
    device = resolve_device(args.device)

    augment_dataset(
        dataset_root=dataset_root,
        output_root=output_root,
        seed=args.seed,
        device=device,
        save_workers=args.save_workers,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
