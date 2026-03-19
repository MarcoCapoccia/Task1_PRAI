import argparse
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import data_loading
from src import gene_pipeline
from src import image_pipeline
from src.utils import ensure_dir, set_global_seed

SEED = 42

def parse_args() -> argparse.Namespace:
    # keeps the cli inputs simple
    parser = argparse.ArgumentParser(
        description="Task 1 Pattern Recognition pipelines for image and gene data."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="Data-PR-As2",
        help="Root directory containing the Data-PR-As2 datasets.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["all", "image", "genes"],
        default="all",
        help="Which dataset pipeline to run.",
    )
    return parser.parse_args()


def main() -> None:
    # runs the full task flow end to end
    args = parse_args()

    task1_root = Path(os.path.dirname(os.path.abspath(__file__)))
    repo_root = task1_root.parent
    data_root = repo_root / args.data_root

    # guards against wrong data paths
    if not data_root.is_dir():
        print(
            f"data root not found at {data_root} use --data-root to set it",
            file=sys.stderr,
        )
        sys.exit(1)

    set_global_seed(SEED)

    outputs_root = task1_root / "outputs"
    ensure_dir(outputs_root)

    # branch handles image experiments
    if args.dataset in ("all", "image"):
        images, y_img, class_names_img = data_loading.load_image_dataset(
            data_root, outputs_root
        )

        # keeps one stable split for image steps
        indices = np.arange(len(images))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, stratify=y_img, random_state=SEED
        )

        # builds the original image representation
        X_orig = image_pipeline._resize_and_flatten(images)  # type: ignore[attr-defined]
        X_train_o, X_test_o = X_orig[train_idx], X_orig[test_idx]
        y_train_o, y_test_o = y_img[train_idx], y_img[test_idx]

        images_train, images_test = images[train_idx], images[test_idx]

        feat_sel = image_pipeline.sweep_image_features(
            images, y_img, class_names_img, outputs_root, random_state=SEED
        )

        # builds reduced features reused downstream
        X_red_train, X_red_test = image_pipeline.build_reduced_features_for_split(
            images_train,
            images_test,
            feature_method=feat_sel.name,
            feature_params=feat_sel.params,
            random_state=SEED,
        )
        y_train_r, y_test_r = y_train_o, y_test_o

        # keeps readable labels in outputs
        img_repr_labels = {
            "original": "Full features",
            "reduced": "Fourier-reduced image features "
            f"({int(feat_sel.params.get('n_features', 0))} retained coefficients)"
            if feat_sel.name == "fourier"
            else "Best reduced image features",
        }
        image_pipeline.classify_images(
            X_train_o,
            X_test_o,
            X_red_train,
            X_red_test,
            y_train_o,
            y_test_o,
            class_names_img,
            outputs_root,
            random_state=SEED,
            repr_labels=img_repr_labels,
        )
        image_pipeline.cluster_images(
            X_train_o, X_red_train, y_train_o, outputs_root, random_state=SEED
        )
        # compares holdout and cv on same image split
        image_pipeline.final_validation_comparison(
            X_red_train,
            X_red_test,
            y_train_o,
            y_test_o,
            best_family="svm",
            outputs_root=outputs_root,
            random_state=SEED,
        )
        image_pipeline.evaluate_augmentation_and_ensemble(
            images_train,
            images_test,
            y_train_o,
            y_test_o,
            class_names_img,
            outputs_root,
            feature_method=feat_sel.name,
            feature_params=feat_sel.params,
            random_state=SEED,
        )
    # branch handles gene experiments
    if args.dataset in ("all", "genes"):
        X_genes, y_genes, class_names_gene, _ = data_loading.load_gene_dataset(
            data_root, outputs_root
        )
        gene_pipeline.analyze_gene_data(
            X_genes, y_genes, class_names_gene, outputs_root, random_state=SEED
        )

        # defines the canonical split for gene steps
        feat_sel_gene = gene_pipeline.sweep_gene_features(
            X_genes, y_genes, outputs_root, random_state=SEED
        )

        scaler = StandardScaler()
        X_values = X_genes.values
        train_idx = feat_sel_gene.train_idx
        test_idx = feat_sel_gene.test_idx

        X_train_o = scaler.fit_transform(X_values[train_idx])
        X_test_o = scaler.transform(X_values[test_idx])
        y_train_o, y_test_o = y_genes[train_idx], y_genes[test_idx]

        X_red_train, X_red_test = feat_sel_gene.X_train, feat_sel_gene.X_test
        y_train_r, y_test_r = y_train_o, y_test_o

        # keeps readable labels in outputs
        if feat_sel_gene.name == "mi_selection":
            k = int(feat_sel_gene.params.get("k", 0))
            red_label = f"MI-features ({k})"
        else:
            nc = int(feat_sel_gene.params.get("n_components", 0))
            red_label = f"PCA-features ({nc})"
        gene_repr_labels = {
            "original": "Full features",
            "reduced": red_label,
        }
        gene_pipeline.classify_genes(
            X_train_o,
            X_test_o,
            X_red_train,
            X_red_test,
            y_train_o,
            y_test_o,
            class_names_gene,
            outputs_root,
            random_state=SEED,
            repr_labels=gene_repr_labels,
        )
        gene_pipeline.cluster_genes(
            X_train_o, X_red_train, y_train_o, outputs_root
        )
        # compares holdout and cv on same gene split
        gene_pipeline.final_validation_comparison(
            X_red_train,
            X_red_test,
            y_train_r,
            y_test_r,
            best_family="gnb",
            outputs_root=outputs_root,
            random_state=SEED,
        )
        gene_pipeline.augmentation_and_ensemble(
            X_red_train,
            X_red_test,
            y_train_r,
            y_test_r,
            outputs_root,
            random_state=SEED,
        )
if __name__ == "__main__":
    main()