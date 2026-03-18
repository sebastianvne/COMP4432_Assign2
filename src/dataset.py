from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SampleRecord:
    path: Path
    relative_path: str
    label: int
    time: str
    cn_name: str


def infer_project_root(csv_path: Path) -> Path:
    csv_path = Path(csv_path).resolve()
    return csv_path.parent.parent


def resolve_sample_path(project_root: Path, relative_path: str) -> Path:
    normalized = relative_path[2:] if relative_path.startswith("./") else relative_path
    return (project_root / normalized).resolve()


def load_samples(csv_path: Path | str, project_root: Path | None = None) -> List[SampleRecord]:
    csv_path = Path(csv_path).resolve()
    project_root = Path(project_root).resolve() if project_root else infer_project_root(csv_path)

    samples: List[SampleRecord] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"path", "label", "time", "cn_name"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"dataset.csv 缺少字段: {sorted(missing)}")

        for row in reader:
            image_path = resolve_sample_path(project_root, row["path"])
            samples.append(
                SampleRecord(
                    path=image_path,
                    relative_path=row["path"],
                    label=int(row["label"]),
                    time=row["time"],
                    cn_name=row["cn_name"],
                )
            )

    if not samples:
        raise ValueError("dataset.csv 中没有样本。")

    missing_paths = [str(sample.path) for sample in samples if not sample.path.exists()]
    if missing_paths:
        preview = ", ".join(missing_paths[:5])
        raise FileNotFoundError(f"以下图片不存在: {preview}")

    return samples


def build_label_to_name(samples: Iterable[SampleRecord]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for sample in samples:
        mapping.setdefault(sample.label, sample.cn_name)
    return dict(sorted(mapping.items()))


def summarize_labels(samples: Sequence[SampleRecord]) -> Dict[int, int]:
    counts = Counter(sample.label for sample in samples)
    return dict(sorted(counts.items()))


def stratified_split(
    samples: Sequence[SampleRecord],
    test_size: float = 0.3,
    random_state: int = 42,
) -> Tuple[List[SampleRecord], List[SampleRecord]]:
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size 必须在 0 和 1 之间。")

    indices = list(range(len(samples)))
    labels = [sample.label for sample in samples]

    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=labels,
    )

    train_samples = [samples[index] for index in train_indices]
    test_samples = [samples[index] for index in test_indices]
    return train_samples, test_samples
