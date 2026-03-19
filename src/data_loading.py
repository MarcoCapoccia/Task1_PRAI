from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from . import visualization as viz

try:
    from skimage.io import imread  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional import error
    imread = None  # type: ignore[assignment]


def _find_image_root(data_root: str | Path) -> Path:
    # finds the image root folder automatically
    root_path = Path(data_root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"data root not found {root_path}")

    candidates: List[Path] = []
    for child in root_path.iterdir():
        if not child.is_dir():
            continue
        if child.name.lower() == "genes":
            continue

        # checks if a child folder contains class image folders
        has_image_subdir = False
        for sub in child.iterdir():
            if not sub.is_dir():
                continue
            images = list(
                sub.glob("*.jpg")
            ) + list(sub.glob("*.jpeg")) + list(sub.glob("*.png"))
            if images:
                has_image_subdir = True
                break
        if has_image_subdir:
            candidates.append(child)

    if not candidates:
        raise RuntimeError(f"no image root found under {data_root}")
    # keeps selection deterministic when multiple roots match
    if len(candidates) > 1:
        candidates.sort()
    return candidates[0]


def load_image_dataset(
    data_root: str | Path,
    outputs_root: str | Path,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    # loads images and builds labels from folder names
    # fails early when skimage is unavailable
    if imread is None:  # pragma: no cover - import error path
        raise ImportError("scikit-image is required for image loading")

    data_root = Path(data_root)
    outputs_root = Path(outputs_root)
    figures_dir = outputs_root / "figures"

    image_root = _find_image_root(data_root)

    images: List[np.ndarray] = []
    labels: List[int] = []
    class_names: List[str] = []

    # keeps class order stable across runs
    for class_dir in sorted([p for p in image_root.iterdir() if p.is_dir()]):
        class_name = class_dir.name
        class_index = len(class_names)
        class_names.append(class_name)

        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for img_path in class_dir.glob(ext):
                img = imread(img_path)
                images.append(img)
                labels.append(class_index)

    if not images:
        raise RuntimeError(f"no images found in {image_root}")

    # saves a compact image dataset summary
    summary_rows = [
        {
            "n_images": len(images),
            "n_classes": len(class_names),
        }
    ]
    tables_dir = outputs_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(
        tables_dir / "image_data_summary.csv", index=False
    )

    labels_arr = np.array(labels, dtype=int)
    viz.plot_class_distribution(
        labels_arr,
        class_names,
        figures_dir / "image_class_distribution.png",
        title="Image dataset class distribution",
    )

    return np.array(images, dtype=object), labels_arr, class_names


def load_gene_dataset(
    data_root: str | Path,
    outputs_root: str | Path,
) -> Tuple[pd.DataFrame, np.ndarray, List[str], LabelEncoder]:
    # loads and aligns gene features and labels
    data_root = Path(data_root)
    outputs_root = Path(outputs_root)
    figures_dir = outputs_root / "figures"
    tables_dir = outputs_root / "tables"

    genes_dir = data_root / "Genes"
    data_path = genes_dir / "data.csv"
    labels_path = genes_dir / "labels.csv"

    if not data_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError(
            f"missing gene files expected {data_path} and {labels_path}"
        )

    X_df = pd.read_csv(data_path, index_col=0)
    y_df = pd.read_csv(labels_path, index_col=0)

    # validates sample id integrity before alignment
    if not X_df.index.is_unique:
        raise ValueError("gene data has duplicate sample ids")
    if not y_df.index.is_unique:
        raise ValueError("gene labels have duplicate sample ids")

    common_ids = X_df.index.intersection(y_df.index)
    if len(common_ids) == 0:
        raise ValueError("no matching sample ids between gene data and labels")

    X_df = X_df.loc[common_ids].copy()
    y_df = y_df.loc[common_ids].copy()

    if y_df.shape[1] != 1:
        raise ValueError(f"labels.csv should have one label column found {y_df.shape[1]}")

    label_col = y_df.columns[0]
    y_raw = y_df[label_col].astype(str)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    class_names = list(le.classes_)

    n_samples, n_genes = X_df.shape
    missing_summary = X_df.isna().sum()
    missing_any = missing_summary.sum()

    # writes missing value and dataset summary tables
    tables_dir.mkdir(parents=True, exist_ok=True)
    missing_summary.to_csv(
        tables_dir / "gene_missing_value_summary.csv", header=["n_missing"]
    )

    gene_summary = [
        {
            "n_samples": n_samples,
            "n_genes": n_genes,
            "n_classes": len(class_names),
            "n_missing_values": int(missing_any),
        }
    ]
    pd.DataFrame(gene_summary).to_csv(
        tables_dir / "gene_data_summary.csv", index=False
    )

    # saves the gene class distribution figure
    viz.plot_class_distribution(
        y_encoded,
        class_names,
        figures_dir / "gene_class_distribution.png",
        title="Gene dataset class distribution",
    )

    return X_df, y_encoded, class_names, le