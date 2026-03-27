"""Task 2 — Narrative Tone Classification.

Trains and evaluates five classifiers (Logistic Regression, SVM, Random
Forest, MLP, and a voting ensemble) on the feature dataset produced by
Task 1. Saves the best model and full evaluation artefacts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from xgboost import XGBClassifier

from src.config import (
    CV_FOLDS,
    FEATURES_DIR,
    MODELS_DIR,
    NARRATIVE_TONES,
    RANDOM_STATE,
    RESULTS_DIR,
    TEST_SIZE,
    VAL_SIZE,
)

logger = logging.getLogger(__name__)

# Metadata columns that are NOT features
_META_COLS = {"file_path", "segment_idx", "emotion_label", "narrative_tone"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_features(features_csv: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the feature CSV and return (X, y).

    Parameters
    ----------
    features_csv:
        Path to ``feature_dataset.csv`` produced by Task 1.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Feature matrix X and label series y.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist.
    """
    if not features_csv.exists():
        raise FileNotFoundError(
            f"Feature dataset not found at {features_csv}. "
            "Run the audio pipeline (Task 1) first."
        )
    df = pd.read_csv(features_csv)
    feature_cols = [c for c in df.columns if c not in _META_COLS]
    X = df[feature_cols]
    y = df["narrative_tone"]
    logger.info(
        "Loaded %d samples, %d features, %d classes.",
        len(df), len(feature_cols), y.nunique(),
    )
    return X, y


def _build_models(random_state: int = RANDOM_STATE) -> dict[str, Pipeline]:
    """Instantiate all classifier pipelines (scaler + model).

    Parameters
    ----------
    random_state:
        Seed for reproducibility.

    Returns
    -------
    dict[str, Pipeline]
        Model name → fitted-ready Pipeline.
    """
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            random_state=random_state,
            class_weight="balanced",
            C=1.0,
        )),
    ])

    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,
            random_state=random_state,
            class_weight="balanced",
        )),
    ])

    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])

    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            max_iter=500,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
            learning_rate_init=1e-3,
        )),
    ])

    ensemble = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(
                max_iter=2000,
                random_state=random_state,
                class_weight="balanced",
                C=1.0,
            )),
            ("svm", SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                probability=True,
                random_state=random_state,
                class_weight="balanced",
            )),
            ("rf", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                random_state=random_state,
                class_weight="balanced",
                n_jobs=-1,
            )),
        ],
        voting="soft",
    )
    ensemble_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", ensemble),
    ])

    xgb = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            random_state=random_state,
            eval_metric="mlogloss",
            verbosity=0,
            n_jobs=-1,
        )),
    ])

    return {
        "logistic_regression": lr,
        "svm_rbf": svm,
        "random_forest": rf,
        "mlp": mlp,
        "voting_ensemble": ensemble_pipeline,
        "xgboost": xgb,
    }


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    model_name: str,
    output_path: Path,
) -> None:
    """Save a seaborn confusion matrix heatmap.

    Parameters
    ----------
    y_true:
        Ground-truth labels (integer-encoded).
    y_pred:
        Predicted labels (integer-encoded).
    labels:
        Human-readable class names.
    model_name:
        Used in the plot title.
    output_path:
        Where to save the PNG.
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, pad=12)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", output_path)


def _plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    labels: list[str],
    model_name: str,
    output_path: Path,
) -> None:
    """Save one-vs-rest ROC curves for each class.

    Parameters
    ----------
    y_true:
        Ground-truth labels (integer-encoded).
    y_proba:
        Predicted probability matrix of shape (n_samples, n_classes).
    labels:
        Human-readable class names.
    model_name:
        Used in the plot title.
    output_path:
        Where to save the PNG.
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    classes = list(range(len(labels)))
    y_bin = label_binarize(y_true, classes=classes)

    fig, ax = plt.subplots(figsize=(9, 7))

    for i, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f"{label} (AUC = {roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curves (One-vs-Rest) — {model_name}", fontsize=14, pad=12)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("ROC curves saved to %s", output_path)


# ---------------------------------------------------------------------------
# Classifier class
# ---------------------------------------------------------------------------


class ToneClassifier:
    """Train and evaluate narrative tone classifiers.

    Parameters
    ----------
    features_csv:
        Path to the feature dataset CSV (output of Task 1).
    results_dir:
        Directory where evaluation artefacts are saved.
    models_dir:
        Directory where trained model files are saved.
    random_state:
        Global random seed.
    """

    def __init__(
        self,
        features_csv: Optional[Path] = None,
        results_dir: Optional[Path] = None,
        models_dir: Optional[Path] = None,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.features_csv = features_csv or (FEATURES_DIR / "feature_dataset.csv")
        self.results_dir = results_dir or RESULTS_DIR
        self.models_dir = models_dir or MODELS_DIR
        self.random_state = random_state

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self._le = LabelEncoder()
        self._best_model: Optional[Pipeline] = None
        self._best_model_name: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Execute the full classification workflow.

        Returns
        -------
        dict[str, Any]
            Summary of results: best model name, test accuracy, F1-macro.
        """
        X, y = _load_features(self.features_csv)
        y_enc = self._le.fit_transform(y)
        class_names = list(self._le.classes_)

        # --- Split ---
        X_train, X_val, X_test, y_train, y_val, y_test = self._split(X, y_enc)
        logger.info(
            "Split sizes — train: %d, val: %d, test: %d",
            len(X_train), len(X_val), len(X_test),
        )

        # --- Train and cross-validate ---
        models = _build_models(self.random_state)
        comparison_rows: list[dict] = []

        for name, pipeline in models.items():
            logger.info("Training %s ...", name)
            pipeline.fit(X_train, y_train)

            # Cross-validation on training data
            cv_scores = cross_val_score(
                pipeline, X_train, y_train,
                cv=CV_FOLDS, scoring="f1_macro", n_jobs=-1,
            )
            val_pred = pipeline.predict(X_val)
            val_f1 = f1_score(y_val, val_pred, average="macro")
            val_acc = accuracy_score(y_val, val_pred)

            logger.info(
                "%s — CV F1-macro: %.4f ± %.4f | Val F1: %.4f | Val Acc: %.4f",
                name, cv_scores.mean(), cv_scores.std(), val_f1, val_acc,
            )
            comparison_rows.append({
                "model": name,
                "cv_f1_macro_mean": cv_scores.mean(),
                "cv_f1_macro_std": cv_scores.std(),
                "val_f1_macro": val_f1,
                "val_accuracy": val_acc,
            })

        # --- Select best model (by val F1-macro) ---
        comparison_df = pd.DataFrame(comparison_rows)
        best_idx = comparison_df["val_f1_macro"].idxmax()
        self._best_model_name = comparison_df.loc[best_idx, "model"]
        self._best_model = models[self._best_model_name]

        # Expose internals for downstream analysis
        self._all_models = models
        self._comparison_df = comparison_df
        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test
        self._y_test = y_test
        self._class_names = class_names
        self._feature_cols = [c for c in X.columns if c not in _META_COLS]

        logger.info("Best model: %s", self._best_model_name)

        # --- Final evaluation on held-out test set ---
        test_pred = self._best_model.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)
        test_f1 = f1_score(y_test, test_pred, average="macro")

        # ROC-AUC (one-vs-rest)
        try:
            test_proba = self._best_model.predict_proba(X_test)
            roc_auc = roc_auc_score(
                y_test, test_proba, multi_class="ovr", average="macro"
            )
        except AttributeError:
            roc_auc = float("nan")

        report_str = classification_report(
            y_test, test_pred, target_names=class_names
        )
        logger.info("Test set results:\n%s", report_str)

        # --- Save artefacts ---
        self._save_classification_report(report_str, test_acc, test_f1, roc_auc)
        comparison_df.to_csv(
            self.results_dir / "model_comparison.csv", index=False
        )
        _plot_confusion_matrix(
            y_test, test_pred, class_names,
            self._best_model_name,
            self.results_dir / "confusion_matrix.png",
        )
        if not np.isnan(roc_auc):
            _plot_roc_curves(
                y_test, test_proba, class_names,
                self._best_model_name,
                self.results_dir / "roc_curves.png",
            )
        self._save_model()

        return {
            "best_model": self._best_model_name,
            "test_accuracy": test_acc,
            "test_f1_macro": test_f1,
            "test_roc_auc": roc_auc,
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict narrative tones for new feature vectors.

        Parameters
        ----------
        X:
            Feature matrix with the same columns as the training data.

        Returns
        -------
        np.ndarray
            Human-readable predicted class labels.
        """
        if self._best_model is None:
            raise RuntimeError("Model has not been trained. Call run() first.")
        y_enc = self._best_model.predict(X)
        return self._le.inverse_transform(y_enc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
               np.ndarray, np.ndarray, np.ndarray]:
        """Stratified train / val / test split.

        Parameters
        ----------
        X:
            Feature matrix.
        y:
            Integer-encoded labels.

        Returns
        -------
        tuple
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First: carve out test set
        sss_test = StratifiedShuffleSplit(
            n_splits=1, test_size=TEST_SIZE, random_state=self.random_state
        )
        train_val_idx, test_idx = next(sss_test.split(X, y))

        X_tv, y_tv = X.iloc[train_val_idx], y[train_val_idx]

        # Second: carve out val set from remaining data
        val_frac = VAL_SIZE / (1.0 - TEST_SIZE)
        sss_val = StratifiedShuffleSplit(
            n_splits=1, test_size=val_frac, random_state=self.random_state
        )
        train_idx, val_idx = next(sss_val.split(X_tv, y_tv))

        return (
            X_tv.iloc[train_idx],
            X_tv.iloc[val_idx],
            X.iloc[test_idx],
            y_tv[train_idx],
            y_tv[val_idx],
            y[test_idx],
        )

    def _save_classification_report(
        self,
        report_str: str,
        accuracy: float,
        f1_macro: float,
        roc_auc: float,
    ) -> None:
        """Write the classification report to a text file.

        Parameters
        ----------
        report_str:
            sklearn classification_report string.
        accuracy:
            Overall accuracy on the test set.
        f1_macro:
            Macro F1 on the test set.
        roc_auc:
            Macro ROC-AUC on the test set.
        """
        out_path = self.results_dir / "classification_report.txt"
        with out_path.open("w") as fh:
            fh.write(f"Best model : {self._best_model_name}\n")
            fh.write(f"Test accuracy : {accuracy:.4f}\n")
            fh.write(f"Test F1-macro : {f1_macro:.4f}\n")
            fh.write(f"Test ROC-AUC  : {roc_auc:.4f}\n\n")
            fh.write(report_str)
        logger.info("Classification report saved to %s", out_path)

    def _save_model(self) -> None:
        """Persist the best pipeline and label encoder with joblib."""
        model_path = self.models_dir / "best_model.pkl"
        le_path = self.models_dir / "label_encoder.pkl"
        joblib.dump(self._best_model, model_path)
        joblib.dump(self._le, le_path)
        logger.info(
            "Model saved to %s | Label encoder saved to %s",
            model_path, le_path,
        )
