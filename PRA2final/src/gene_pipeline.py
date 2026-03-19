from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier

from . import evaluation as eval_utils
from . import visualization as viz
from .utils import ensure_dir, save_json


@dataclass
class GeneFeatureSelectionResult:
    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    params: Dict[str, float]
    train_idx: np.ndarray
    test_idx: np.ndarray


def analyze_gene_data(
    X: pd.DataFrame,
    y: np.ndarray,
    class_names: List[str],
    outputs_root: str | Path,
    random_state: int = 42,
) -> None:
    # creates a simple pca view of gene data
    outputs_root = Path(outputs_root)
    figures_dir = ensure_dir(outputs_root / "figures")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    pca = PCA(n_components=2, random_state=random_state)
    X_2d = pca.fit_transform(X_scaled)
    evr = pca.explained_variance_ratio_
    title = (
        f"Gene dataset PCA projection\n"
        f"PC1 ({evr[0]*100:.1f}% variance), PC2 ({evr[1]*100:.1f}% variance)"
    )
    viz.plot_scatter_2d(
        X_2d,
        y,
        class_names,
        title,
        figures_dir / "gene_pca_scatter.png",
    )


def sweep_gene_features(
    X: pd.DataFrame,
    y: np.ndarray,
    outputs_root: str | Path,
    random_state: int = 42,
) -> GeneFeatureSelectionResult:
    # compares mi and pca on one canonical split
    outputs_root = Path(outputs_root)
    figures_dir = ensure_dir(outputs_root / "figures")
    tables_dir = ensure_dir(outputs_root / "tables")

    X_values = X.values
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=y, random_state=random_state
    )
    X_train = X_values[train_idx]
    X_test = X_values[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # sweeps top k genes from mutual information
    ks = [10, 25, 50, 100, 250]
    mi_scores = mutual_info_classif(X_train_scaled, y_train, random_state=random_state)
    gene_indices_sorted = np.argsort(mi_scores)[::-1]

    gnb = GaussianNB()
    mi_val_scores: List[float] = []
    best_k = ks[0]
    best_k_score = -np.inf

    for k in ks:
        k = min(k, X_train_scaled.shape[1])
        sel_idx = gene_indices_sorted[:k]
        X_tr_sel = X_train_scaled[:, sel_idx]
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        scores = []
        for tr_idx, val_idx in cv.split(X_tr_sel, y_train):
            gnb.fit(X_tr_sel[tr_idx], y_train[tr_idx])
            y_val_pred = gnb.predict(X_tr_sel[val_idx])
            metrics = eval_utils.evaluate_classification(
                y_train[val_idx], y_val_pred
            )
            scores.append(metrics.accuracy)
        mean_score = float(np.mean(scores))
        mi_val_scores.append(mean_score)
        if mean_score > best_k_score:
            best_k_score = mean_score
            best_k = k

    viz.plot_line(
        ks[: len(mi_val_scores)],
        mi_val_scores,
        xlabel="Number of selected genes (mutual information)",
        ylabel="Validation accuracy",
        title="Mutual information feature selection sweep (GNB)",
        path=figures_dir / "gene_mi_k_vs_acc.png",
    )

    # sweeps pca component counts
    max_components = min(50, X_train_scaled.shape[1], X_train_scaled.shape[0])
    n_components_list = [5, 10, 20, max_components]
    n_components_list = sorted(set([nc for nc in n_components_list if nc > 0]))

    pca = PCA(random_state=random_state)
    pca.fit(X_train_scaled)

    pca_val_scores: List[float] = []
    best_nc = n_components_list[0]
    best_nc_score = -np.inf

    for nc in n_components_list:
        pca_nc = PCA(n_components=nc, random_state=random_state)
        X_tr_pca = pca_nc.fit_transform(X_train_scaled)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        scores = []
        for tr_idx, val_idx in cv.split(X_tr_pca, y_train):
            gnb.fit(X_tr_pca[tr_idx], y_train[tr_idx])
            y_val_pred = gnb.predict(X_tr_pca[val_idx])
            metrics = eval_utils.evaluate_classification(
                y_train[val_idx], y_val_pred
            )
            scores.append(metrics.accuracy)
        mean_score = float(np.mean(scores))
        pca_val_scores.append(mean_score)
        if mean_score > best_nc_score:
            best_nc_score = mean_score
            best_nc = nc

    viz.plot_line(
        n_components_list,
        pca_val_scores,
        xlabel="Number of PCA components",
        ylabel="Validation accuracy",
        title="PCA component sweep (Gaussian Naive Bayes)",
        path=figures_dir / "gene_pca_components_vs_acc.png",
    )

    # writes the feature comparison table
    comp_rows = [
        {"method": "mi_selection", "param": best_k, "cv_accuracy": best_k_score},
        {"method": "pca", "param": best_nc, "cv_accuracy": best_nc_score},
    ]
    pd.DataFrame(comp_rows).to_csv(
        tables_dir / "gene_feature_comparison.csv", index=False
    )

    if best_k_score >= best_nc_score:
        chosen = "mi_selection"
        sel_idx = gene_indices_sorted[:best_k]
        X_tr_red = X_train_scaled[:, sel_idx]
        X_te_red = X_test_scaled[:, sel_idx]
        params = {"k": float(best_k)}
    else:
        chosen = "pca"
        pca_final = PCA(n_components=best_nc, random_state=random_state)
        X_tr_red = pca_final.fit_transform(X_train_scaled)
        X_te_red = pca_final.transform(X_test_scaled)
        params = {"n_components": float(best_nc)}

    return GeneFeatureSelectionResult(
        name=chosen,
        X_train=X_tr_red,
        X_test=X_te_red,
        params=params,
        train_idx=train_idx,
        test_idx=test_idx,
    )


def classify_genes(
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
    # trains gene classifiers on both representations
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
            "dt": DecisionTreeClassifier(random_state=random_state),
            "gnb": GaussianNB(),
        }
        model_names = {
            "knn": "K-Nearest Neighbours",
            "dt": "Decision Tree",
            "gnb": "Gaussian Naive Bayes",
        }
        param_grids = {
            "knn": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
            "dt": {
                "criterion": ["gini", "entropy"],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            },
            "gnb": {"var_smoothing": [1e-9, 1e-8, 1e-7]},
        }

        for name, base_est in models.items():
            if name in ("knn", "dt", "gnb"):
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
                figures_dir / f"gene_cm_{repr_name}_{name}.png",
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

    # saves all gene model params to one json
    save_json(best_params, logs_dir / "gene_best_params.json")

    df = pd.DataFrame(results_rows)
    df.to_csv(tables_dir / "gene_classification_results.csv", index=False)

    return df.set_index(["representation", "model"]).to_dict(orient="index")


def cluster_genes(
    X_orig: np.ndarray,
    X_red: np.ndarray,
    y: np.ndarray,
    outputs_root: str | Path,
) -> None:
    # runs agglomerative clustering for both representations
    outputs_root = Path(outputs_root)
    tables_dir = ensure_dir(outputs_root / "tables")

    n_clusters = len(np.unique(y))
    rows: List[Dict[str, float]] = []

    for repr_name, X in [("original", X_orig), ("reduced", X_red)]:
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        labels_pred = clustering.fit_predict(X)
        scores = eval_utils.clustering_scores(X, y, labels_pred)
        row = {"representation": repr_name}
        row.update(scores)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(tables_dir / "gene_clustering_results.csv", index=False)


def final_validation_comparison(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    best_family: str,
    outputs_root: str | Path,
    random_state: int = 42,
) -> None:
    # compares holdout and cv for one gene model family
    outputs_root = Path(outputs_root)
    tables_dir = ensure_dir(outputs_root / "tables")

    if best_family == "gnb":
        base = GaussianNB()
        grid_params = {"var_smoothing": [1e-9, 1e-8, 1e-7]}
    elif best_family == "dt":
        base = DecisionTreeClassifier(random_state=random_state)
        grid_params = {
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        }
    else:
        base = KNeighborsClassifier()
        grid_params = {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]}

    # runs the plain holdout path
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
    df.to_csv(tables_dir / "gene_validation_comparison.csv", index=False)


def augmentation_and_ensemble(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    outputs_root: str | Path,
    random_state: int = 42,
) -> None:
    # evaluates the gene ensemble against a single baseline
    outputs_root = Path(outputs_root)
    tables_dir = ensure_dir(outputs_root / "tables")

    # loads tuned gnb params when available
    best_gnb_params: Dict[str, object] = {}
    params_path = outputs_root / "params" / "gene_best_params.json"
    if params_path.is_file():
        try:
            with params_path.open("r", encoding="utf-8") as f:
                best_all = json.load(f)
            best_gnb_params = best_all.get("reduced_gnb", {})
        except Exception:
            best_gnb_params = {}

    # builds a hard voting ensemble
    knn = KNeighborsClassifier()
    dt = DecisionTreeClassifier(random_state=random_state)
    gnb = GaussianNB()
    if best_gnb_params:
        # maps pipeline params to the standalone gnb model
        gnb_kwargs: Dict[str, object] = {}
        for key, val in best_gnb_params.items():
            if key.startswith("gnb__"):
                gnb_kwargs[key.split("gnb__", 1)[1]] = val
        gnb.set_params(**gnb_kwargs)
    ensemble = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "voting",
                VotingClassifier(
                    estimators=[("knn", knn), ("dt", dt), ("gnb", gnb)],
                    voting="hard",
                ),
            ),
        ]
    )
    ensemble.fit(X_train, y_train)
    y_pred_ens = ensemble.predict(X_test)
    ens_metrics = eval_utils.evaluate_classification(y_test, y_pred_ens)

    # compares ensemble results to single gnb
    base_model = Pipeline(
        [("scaler", StandardScaler()), ("gnb", GaussianNB())]
    )
    if best_gnb_params:
        base_model.set_params(**best_gnb_params)
    base_model.fit(X_train, y_train)
    y_pred_base = base_model.predict(X_test)
    base_metrics = eval_utils.evaluate_classification(y_test, y_pred_base)

    df = pd.DataFrame(
        [
            {
                "model": "gaussian_nb_single",
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
    df.to_csv(tables_dir / "gene_ensemble_results.csv", index=False)