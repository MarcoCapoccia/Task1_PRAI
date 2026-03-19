from __future__ import annotations

import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from skimage.color import rgb2gray  # type: ignore[import-not-found]
from skimage.transform import resize  # type: ignore[import-not-found]
from skimage.transform import rotate  # type: ignore[import-not-found]
from skimage.exposure import rescale_intensity  # type: ignore[import-not-found]

from . import evaluation as eval_utils
from . import visualization as viz
from .utils import ensure_dir, save_json

try:
    import cv2  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency path
    cv2 = None  # type: ignore[assignment]


@dataclass
class ImageFeatureSelectionResult:
    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    params: Dict[str, float]


def _augment_image(img: np.ndarray, max_rotation: float = 15.0) -> List[np.ndarray]:
    # creates a small set of image augmentations
    aug: List[np.ndarray] = []

    # normalizes values before transforms
    x = img.astype(np.float32)
    if x.max() > 1.0:
        x = x / 255.0

    # applies a small positive rotation
    aug.append(rotate(x, angle=max_rotation, mode="edge", preserve_range=True))

    # applies a small negative rotation
    aug.append(rotate(x, angle=-max_rotation, mode="edge", preserve_range=True))

    # applies a horizontal flip
    aug.append(np.fliplr(x))

    # applies a mild intensity adjustment
    aug.append(rescale_intensity(x, in_range="image", out_range=(0.1, 0.9)))

    return [a.astype(img.dtype) for a in aug]


def _fourier_features_from_images(
    images: np.ndarray,
    patch_size: int,
) -> np.ndarray:
    # extracts low frequency fourier features
    feats = []
    for img in images:
        if img.ndim == 3 and img.shape[2] == 3:
            g = rgb2gray(img)
        else:
            g = img
        g_resized = resize(g, (128, 128), anti_aliasing=True)
        f = np.fft.fft2(g_resized)
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift)
        center = (mag.shape[0] // 2, mag.shape[1] // 2)
        h = w = patch_size
        r0 = center[0] - h // 2
        c0 = center[1] - w // 2
        patch = mag[r0 : r0 + h, c0 : c0 + w]
        feats.append(patch.ravel())
    return np.vstack(feats)


def _resize_and_flatten(images: np.ndarray, size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    # builds the original flattened pixel representation
    feats = []
    for img in images:
        if img.ndim == 3 and img.shape[2] == 3:
            img = rgb2gray(img)
        img_resized = resize(img, size, anti_aliasing=True)
        feats.append(img_resized.ravel())
    return np.vstack(feats)


def _compute_pca_embeds(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int,
    out_path: Path,
    class_names: List[str],
) -> None:
    # creates a pca visualization embedding
    pca = PCA(n_components=n_components, random_state=42)
    X_2d = pca.fit_transform(X)[:, :2]
    evr = pca.explained_variance_ratio_
    title = (
        f"Image dataset PCA projection\n"
        f"PC1 ({evr[0]*100:.1f}% variance), PC2 ({evr[1]*100:.1f}% variance)"
    )
    viz.plot_scatter_2d(X_2d, y, class_names, title, out_path)


def _sift_descriptors_for_images(images: np.ndarray) -> List[np.ndarray]:
    # extracts sift descriptors for each image
    if cv2 is None or not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("sift is not available install opencv-contrib-python")

    sift = cv2.SIFT_create()
    descriptors_list: List[np.ndarray] = []
    for img in images:
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.astype(np.uint8)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        if descriptors is None:
            descriptors = np.empty((0, 128), dtype=np.float32)
        descriptors_list.append(descriptors)
    return descriptors_list


def _build_bovw_features(
    descriptors_list: List[np.ndarray],
    vocab_size: int,
    random_state: int,
) -> Tuple[np.ndarray, MiniBatchKMeans]:
    # builds bovw histograms from descriptor vocabulary
    all_desc = np.vstack([d for d in descriptors_list if d.size > 0])
    kmeans = MiniBatchKMeans(
        n_clusters=vocab_size,
        batch_size=min(1000, len(all_desc)),
        random_state=random_state,
    )
    kmeans.fit(all_desc)

    feats = []
    for desc in descriptors_list:
        if desc.size == 0:
            hist = np.zeros(vocab_size, dtype=float)
        else:
            words = kmeans.predict(desc)
            hist, _ = np.histogram(words, bins=np.arange(vocab_size + 1))
            hist = hist.astype(float)
        # normalizes each histogram
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist /= norm
        feats.append(hist)
    return np.vstack(feats), kmeans


def _bovw_features_with_vocab(
    descriptors_list: List[np.ndarray],
    kmeans_model: MiniBatchKMeans,
) -> np.ndarray:
    # builds bovw histograms using a fixed vocabulary
    feats: List[np.ndarray] = []
    n_clusters = int(kmeans_model.n_clusters)
    for desc in descriptors_list:
        if desc.size == 0:
            hist = np.zeros(n_clusters, dtype=float)
        else:
            words = kmeans_model.predict(desc)
            hist, _ = np.histogram(words, bins=np.arange(n_clusters + 1))
            hist = hist.astype(float)
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist /= norm
        feats.append(hist)
    return np.vstack(feats)


def build_reduced_features_for_split(
    images_train: np.ndarray,
    images_test: np.ndarray,
    feature_method: str,
    feature_params: Dict[str, float],
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    # builds reduced train and test features consistently
    if feature_method == "fourier":
        patch_size = int(feature_params.get("patch_size", 8))
        X_train = _fourier_features_from_images(images_train, patch_size)
        X_test = _fourier_features_from_images(images_test, patch_size)
        return X_train, X_test

    if feature_method == "sift_bovw":
        vocab_size = int(feature_params.get("vocab_size", 50))

        descriptors_train = _sift_descriptors_for_images(images_train)
        X_train, kmeans = _build_bovw_features(
            descriptors_train, vocab_size=vocab_size, random_state=random_state
        )

        descriptors_test = _sift_descriptors_for_images(images_test)
        X_test = _bovw_features_with_vocab(descriptors_test, kmeans)
        return X_train, X_test

    raise ValueError(f"unknown feature method {feature_method} use fourier or sift_bovw")


def sweep_image_features(
    X: np.ndarray,
    y: np.ndarray,
    class_names: List[str],
    outputs_root: str | Path,
    random_state: int = 42,
) -> ImageFeatureSelectionResult:
    # compares sift bovw and fourier on cv accuracy
    outputs_root = Path(outputs_root)
    figures_dir = ensure_dir(outputs_root / "figures")
    tables_dir = ensure_dir(outputs_root / "tables")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    # saves a pca plot for the original representation
    orig_flat = _resize_and_flatten(X_train)
    _compute_pca_embeds(
        orig_flat,
        y_train,
        n_components=min(50, orig_flat.shape[1]),
        out_path=figures_dir / "image_pca_scatter.png",
        class_names=class_names,
    )

    # runs the sift bovw sweep
    vocab_sizes = [50, 100, 150, 200, 300]
    try:
        sift_descriptors_train = _sift_descriptors_for_images(X_train)
        sift_available = True
    except ImportError as exc:
        sift_available = False

    svm_params = {"svc__C": [0.1, 1.0], "svc__kernel": ["linear", "rbf"]}

    sift_scores: List[float] = []
    sift_best_by_vocab: Dict[int, float] = {}

    if sift_available:
        for k in vocab_sizes:
            # fits vocabulary on train descriptors only
            X_sift_train, _ = _build_bovw_features(
                sift_descriptors_train, vocab_size=k, random_state=random_state
            )

            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("svc", SVC(probability=False, random_state=random_state)),
                ]
            )
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
            grid = GridSearchCV(pipe, svm_params, cv=cv, n_jobs=1)
            grid.fit(X_sift_train, y_train)
            sift_scores.append(grid.best_score_)
            sift_best_by_vocab[k] = grid.best_score_

        viz.plot_line(
            vocab_sizes,
            sift_scores,
            xlabel="Vocabulary size",
            ylabel="CV accuracy",
            title="SIFT-BoVW validation",
            path=figures_dir / "image_sift_vocab_vs_acc.png",
        )

    # runs the fourier sweep
    patch_sizes = [4, 8, 16]
    fourier_scores: List[float] = []
    best_by_patch: Dict[int, float] = {}

    imgs_gray = []
    for img in X_train:
        if img.ndim == 3 and img.shape[2] == 3:
            g = rgb2gray(img)
        else:
            g = img
        g_resized = resize(g, (128, 128), anti_aliasing=True)
        imgs_gray.append(g_resized)
    imgs_gray = np.stack(imgs_gray)

    def fourier_features(imgs: np.ndarray, patch_size: int) -> np.ndarray:
        feats = []
        for g in imgs:
            f = np.fft.fft2(g)
            fshift = np.fft.fftshift(f)
            mag = np.abs(fshift)
            center = (mag.shape[0] // 2, mag.shape[1] // 2)
            h, w = patch_size, patch_size
            r0 = center[0] - h // 2
            c0 = center[1] - w // 2
            patch = mag[r0 : r0 + h, c0 : c0 + w]
            feats.append(patch.ravel())
        return np.vstack(feats)

    for ps in patch_sizes:
        X_fourier_train = fourier_features(imgs_gray, ps)
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svc", SVC(probability=False, random_state=random_state)),
            ]
        )
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        grid = GridSearchCV(pipe, svm_params, cv=cv, n_jobs=1)
        grid.fit(X_fourier_train, y_train)
        fourier_scores.append(grid.best_score_)
        best_by_patch[ps] = grid.best_score_

    viz.plot_line(
        [ps * ps for ps in patch_sizes],
        fourier_scores,
        xlabel="Number of Fourier features",
        ylabel="CV accuracy",
        title="Fourier feature validation",
        path=figures_dir / "image_fourier_features_vs_acc.png",
    )

    # picks the best reduced method by validation score
    best_fourier_ps = max(best_by_patch, key=best_by_patch.get)

    feature_rows = []
    if sift_best_by_vocab:
        best_sift_k = max(sift_best_by_vocab, key=sift_best_by_vocab.get)
        feature_rows.append(
            {
                "method": "sift_bovw",
                "param": best_sift_k,
                "cv_accuracy": sift_best_by_vocab[best_sift_k],
            }
        )
    feature_rows.append(
        {
            "method": "fourier",
            "param": best_fourier_ps * best_fourier_ps,
            "cv_accuracy": best_by_patch[best_fourier_ps],
        }
    )
    df = pd.DataFrame(feature_rows)
    df.to_csv(tables_dir / "image_feature_comparison.csv", index=False)

    if sift_best_by_vocab and sift_best_by_vocab[max(sift_best_by_vocab, key=sift_best_by_vocab.get)] >= best_by_patch[best_fourier_ps]:
        chosen = "sift_bovw"
        # rebuilds bovw features with train fitted vocabulary
        descriptors_train = _sift_descriptors_for_images(X_train)
        X_sift_train, kmeans = _build_bovw_features(
            descriptors_train, best_sift_k, random_state
        )
        descriptors_test = _sift_descriptors_for_images(X_test)
        X_sift_test = _bovw_features_with_vocab(descriptors_test, kmeans)
        params = {"vocab_size": float(best_sift_k)}
        X_red_train, X_red_test = X_sift_train, X_sift_test
    else:
        chosen = "fourier"
        ps = best_fourier_ps
        # recomputes grayscale tensors for both splits
        def prep_gray(imgs: np.ndarray) -> np.ndarray:
            arr = []
            for im in imgs:
                if im.ndim == 3 and im.shape[2] == 3:
                    g = rgb2gray(im)
                    g_resized = resize(g, (128, 128), anti_aliasing=True)
                else:
                    g_resized = im
                arr.append(g_resized)
            return np.stack(arr)

        X_train_g = prep_gray(X_train)
        X_test_g = prep_gray(X_test)

        def fourier_feats_only(imgs: np.ndarray, patch_size: int) -> np.ndarray:
            feats = []
            for g in imgs:
                f = np.fft.fft2(g)
                fshift = np.fft.fftshift(f)
                mag = np.abs(fshift)
                center = (mag.shape[0] // 2, mag.shape[1] // 2)
                h, w = patch_size, patch_size
                r0 = center[0] - h // 2
                c0 = center[1] - w // 2
                patch = mag[r0 : r0 + h, c0 : c0 + w]
                feats.append(patch.ravel())
            return np.vstack(feats)

        X_red_train = fourier_feats_only(X_train_g, ps)
        X_red_test = fourier_feats_only(X_test_g, ps)
        params = {"patch_size": float(ps), "n_features": float(ps * ps)}

    return ImageFeatureSelectionResult(
        name=chosen,
        X_train=X_red_train,
        X_test=X_red_test,
        params=params,
    )


def classify_images(
    X_orig_train: np.ndarray,
    X_orig_test: np.ndarray,
    X_red_train: np.ndarray,
    X_red_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    outputs_root: str | Path,
    random_state: int = 42,
    repr_labels: Dict[str, str] | None = None,
) -> Dict[str, Dict[str, float]]:
    # trains image classifiers on both representations
    outputs_root = Path(outputs_root)
    figures_dir = ensure_dir(outputs_root / "figures")
    tables_dir = ensure_dir(outputs_root / "tables")
    logs_dir = ensure_dir(outputs_root / "params")

    results_rows: List[Dict[str, float]] = []
    best_params: Dict[str, Dict[str, object]] = {}

    def eval_repr(X_tr: np.ndarray, X_te: np.ndarray, repr_name: str) -> None:
        nonlocal results_rows
        if repr_labels is None:
            repr_desc = repr_name
        else:
            repr_desc = repr_labels.get(repr_name, repr_name)
        models = {
            "knn": KNeighborsClassifier(),
            "svm": SVC(probability=True, random_state=random_state),
            "rf": RandomForestClassifier(random_state=random_state),
        }
        model_names = {
            "knn": "K-Nearest Neighbours",
            "svm": "Support Vector Machine",
            "rf": "Random Forest",
        }
        param_grids = {
            "knn": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
            "svm": {
                "C": [0.1, 1.0],
                "kernel": ["linear", "rbf"],
                "gamma": ["scale"],
            },
            "rf": {
                "n_estimators": [100, 200],
                "max_depth": [None, 10],
                "min_samples_split": [2, 5],
            },
        }

        for name, base_est in models.items():
            if name in ("svm", "knn", "rf"):
                pipe = Pipeline([("scaler", StandardScaler()), (name, base_est)])
            else:
                pipe = base_est
            grid = GridSearchCV(
                pipe,
                {f"{name}__{k}": v for k, v in param_grids[name].items()},
                cv=3,
                n_jobs=1,
            )
            grid.fit(X_tr, y_train)

            # stores best params per representation and model
            key = f"{repr_name}_{name}"
            best_params[key] = grid.best_params_

            y_pred = grid.best_estimator_.predict(X_te)
            try:
                y_proba = grid.best_estimator_.predict_proba(X_te)
            except Exception:
                y_proba = None
            metrics = eval_utils.evaluate_classification(y_test, y_pred, y_proba)
            cm = eval_utils.compute_confusion_matrix(y_test, y_pred)
            model_desc = model_names.get(name, name)
            title = f"{model_desc} confusion matrix: \n{repr_desc}"
            viz.plot_confusion_matrix(
                cm,
                class_names,
                title,
                figures_dir / f"image_cm_{repr_name}_{name}.png",
            )
            results_rows.append(
                {
                    "representation": repr_name,
                    "model": name,
                    "accuracy": metrics.accuracy,
                    "f1_macro": metrics.f1_macro,
                    "f1_weighted": metrics.f1_weighted or math.nan,
                    "roc_auc_ovr": metrics.roc_auc_ovr or math.nan,
                }
            )

    eval_repr(X_orig_train, X_orig_test, "original")
    eval_repr(X_red_train, X_red_test, "reduced")

    # saves all image model params to one json
    save_json(best_params, logs_dir / "image_best_params.json")

    df = pd.DataFrame(results_rows)
    df.to_csv(tables_dir / "image_classification_results.csv", index=False)

    return df.set_index(["representation", "model"]).to_dict(orient="index")


def cluster_images(
    X_orig: np.ndarray,
    X_red: np.ndarray,
    y: np.ndarray,
    outputs_root: str | Path,
    random_state: int = 42,
) -> None:
    # runs kmeans clustering for both representations
    outputs_root = Path(outputs_root)
    tables_dir = ensure_dir(outputs_root / "tables")

    n_clusters = len(np.unique(y))
    rows: List[Dict[str, float]] = []

    for repr_name, X in [("original", X_orig), ("reduced", X_red)]:
        km = KMeans(
            n_clusters=n_clusters, random_state=random_state, n_init=10
        )
        labels_pred = km.fit_predict(X)
        scores = eval_utils.clustering_scores(X, y, labels_pred)
        row = {"representation": repr_name}
        row.update(scores)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(tables_dir / "image_clustering_results.csv", index=False)


def final_validation_comparison(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    best_family: str,
    outputs_root: str | Path,
    random_state: int = 42,
) -> None:
    # compares holdout and cv for one image model family
    outputs_root = Path(outputs_root)
    tables_dir = ensure_dir(outputs_root / "tables")

    # defines estimator family and search grid
    if best_family == "svm":
        base = SVC(probability=True, random_state=random_state)
        grid_params = {
            "C": [0.1, 1.0],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale"],
        }
    elif best_family == "rf":
        base = RandomForestClassifier(random_state=random_state)
        grid_params = {
            "n_estimators": [100, 200],
            "max_depth": [None, 10],
            "min_samples_split": [2, 5],
        }
    else:
        base = KNeighborsClassifier()
        grid_params = {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]}

    # runs a plain holdout baseline
    pipe_plain = Pipeline([("scaler", StandardScaler()), ("model", base)])
    first_params = {f"model__{k}": v[0] for k, v in grid_params.items()}
    pipe_plain.set_params(**first_params)
    pipe_plain.fit(X_train, y_train)
    y_pred_plain = pipe_plain.predict(X_test)
    try:
        y_proba_plain = pipe_plain.predict_proba(X_test)
    except Exception:
        y_proba_plain = None
    metrics_plain = eval_utils.evaluate_classification(
        y_test, y_pred_plain, y_proba_plain
    )

    # runs the cross validation path
    pipe_cv = Pipeline([("scaler", StandardScaler()), ("model", base)])
    grid = GridSearchCV(
        pipe_cv,
        {f"model__{k}": v for k, v in grid_params.items()},
        cv=5,
        n_jobs=1,
    )
    grid.fit(X_train, y_train)
    y_pred_cv = grid.best_estimator_.predict(X_test)
    try:
        y_proba_cv = grid.best_estimator_.predict_proba(X_test)
    except Exception:
        y_proba_cv = None
    metrics_cv = eval_utils.evaluate_classification(y_test, y_pred_cv, y_proba_cv)

    df = pd.DataFrame(
        [
            {
                "strategy": "holdout_no_cv",
                "accuracy": metrics_plain.accuracy,
                "f1_macro": metrics_plain.f1_macro,
                "roc_auc_ovr": metrics_plain.roc_auc_ovr or math.nan,
            },
            {
                "strategy": "cv_5fold",
                "accuracy": metrics_cv.accuracy,
                "f1_macro": metrics_cv.f1_macro,
                "roc_auc_ovr": metrics_cv.roc_auc_ovr or math.nan,
            },
        ]
    )
    df.to_csv(tables_dir / "image_validation_comparison.csv", index=False)


def evaluate_augmentation_and_ensemble(
    images_train: np.ndarray,
    images_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    outputs_root: str | Path,
    feature_method: str,
    feature_params: Dict[str, float],
    random_state: int = 42,
) -> None:
    # runs augmentation and ensemble comparisons
    outputs_root = Path(outputs_root)
    tables_dir = ensure_dir(outputs_root / "tables")

    # loads tuned svm params when available
    best_svm_params: Dict[str, object] = {}
    params_path = outputs_root / "params" / "image_best_params.json"
    if params_path.is_file():
        try:
            with params_path.open("r", encoding="utf-8") as f:
                best_all = json.load(f)
            best_svm_params = best_all.get("reduced_svm", {})
        except Exception:
            best_svm_params = {}

    # computes baseline features without augmentation
    if feature_method == "fourier":
        patch_size = int(feature_params.get("patch_size", 8))
        X_train_noaug, X_test = build_reduced_features_for_split(
            images_train,
            images_test,
            feature_method="fourier",
            feature_params={"patch_size": float(patch_size)},
            random_state=random_state,
        )
    elif feature_method == "sift_bovw":
        vocab_size = int(feature_params.get("vocab_size", 50))

        X_train_noaug, X_test = build_reduced_features_for_split(
            images_train,
            images_test,
            feature_method="sift_bovw",
            feature_params={"vocab_size": float(vocab_size)},
            random_state=random_state,
        )
    else:
        raise ValueError(
            f"unknown feature method {feature_method} for augmentation use fourier or sift_bovw"
        )

    # fits the baseline svm
    base_svm = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(probability=True, random_state=random_state)),
        ]
    )
    if best_svm_params:
        base_svm.set_params(**best_svm_params)
    base_svm.fit(X_train_noaug, y_train)
    y_pred = base_svm.predict(X_test)
    try:
        y_proba = base_svm.predict_proba(X_test)
    except Exception:
        y_proba = None
    base_metrics = eval_utils.evaluate_classification(y_test, y_pred, y_proba)

    # creates augmented training images only
    augmented_images: List[np.ndarray] = []
    augmented_labels: List[np.ndarray] = []
    for img, label in zip(images_train, y_train):
        for aug_img in _augment_image(img):
            augmented_images.append(aug_img)
            augmented_labels.append(label)

    images_train_aug = list(images_train) + augmented_images
    y_train_aug = np.concatenate([y_train, np.array(augmented_labels)], axis=0)

    # recomputes features for augmented training data
    if feature_method == "fourier":
        X_train_aug = _fourier_features_from_images(images_train_aug, patch_size)
    else:
        descriptors_train_aug = _sift_descriptors_for_images(images_train_aug)
        vocab_size = int(feature_params.get("vocab_size", 50))
        X_train_aug, kmeans_aug = _build_bovw_features(
            descriptors_train_aug, vocab_size=vocab_size, random_state=random_state
        )
        descriptors_test = _sift_descriptors_for_images(images_test)
        X_test = _bovw_features_with_vocab(descriptors_test, kmeans_aug)

    aug_svm = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(probability=True, random_state=random_state)),
        ]
    )
    if best_svm_params:
        aug_svm.set_params(**best_svm_params)
    aug_svm.fit(X_train_aug, y_train_aug)
    y_pred_aug = aug_svm.predict(X_test)
    try:
        y_proba_aug = aug_svm.predict_proba(X_test)
    except Exception:
        y_proba_aug = None
    aug_metrics = eval_utils.evaluate_classification(y_test, y_pred_aug, y_proba_aug)

    df_aug = pd.DataFrame(
        [
            {
                "setting": "no_augmentation",
                "accuracy": base_metrics.accuracy,
                "f1_macro": base_metrics.f1_macro,
                "roc_auc_ovr": base_metrics.roc_auc_ovr or math.nan,
            },
            {
                "setting": "with_augmentation",
                "accuracy": aug_metrics.accuracy,
                "f1_macro": aug_metrics.f1_macro,
                "roc_auc_ovr": aug_metrics.roc_auc_ovr or math.nan,
            },
        ]
    )
    df_aug.to_csv(tables_dir / "image_augmentation_results.csv", index=False)

    # evaluates a soft voting ensemble
    knn = KNeighborsClassifier()
    svm_clf = SVC(probability=True, random_state=random_state)
    if best_svm_params:
        # maps pipeline params to standalone svc params
        svm_kwargs = {}
        for key, val in best_svm_params.items():
            if key.startswith("svm__"):
                svm_kwargs[key.split("svm__", 1)[1]] = val
        svm_clf.set_params(**svm_kwargs)
    rf = RandomForestClassifier(random_state=random_state)
    ensemble = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "voting",
                VotingClassifier(
                    estimators=[("knn", knn), ("svm", svm_clf), ("rf", rf)],
                    voting="soft",
                ),
            ),
        ]
    )
    ensemble.fit(X_train_noaug, y_train)
    y_pred_ens = ensemble.predict(X_test)
    try:
        y_proba_ens = ensemble.predict_proba(X_test)
    except Exception:
        y_proba_ens = None
    ens_metrics = eval_utils.evaluate_classification(
        y_test, y_pred_ens, y_proba_ens
    )

    df_ens = pd.DataFrame(
        [
            {
                "model": "svm_single",
                "accuracy": base_metrics.accuracy,
                "f1_macro": base_metrics.f1_macro,
                "roc_auc_ovr": base_metrics.roc_auc_ovr or math.nan,
            },
            {
                "model": "voting_ensemble",
                "accuracy": ens_metrics.accuracy,
                "f1_macro": ens_metrics.f1_macro,
                "roc_auc_ovr": ens_metrics.roc_auc_ovr or math.nan,
            },
        ]
    )
    df_ens.to_csv(tables_dir / "image_ensemble_results.csv", index=False)