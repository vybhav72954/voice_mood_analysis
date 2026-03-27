"""Extended Model Analysis -- Feature Importance, Learning Curves, SHAP.

Generates publication-quality analysis plots from a trained
``ToneClassifier`` instance. Designed to run after ``ToneClassifier.run()``.
"""

from __future__ import annotations

import contextlib
import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve

from src.config import CV_FOLDS, RANDOM_STATE

if TYPE_CHECKING:
    from src.classification.tone_classifier import ToneClassifier

logger = logging.getLogger(__name__)

# Maximum test samples to feed into KernelExplainer (slow path).
# TreeExplainer is fast and has no cap.
_KERNEL_SHAP_MAX_SAMPLES: int = 100


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------


def _plot_feature_importance(
    classifier: ToneClassifier,
    top_n: int = 20,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Compute and plot permutation-based feature importance.

    Permutation importance is model-agnostic and works on any pipeline,
    unlike impurity-based importance which is Random-Forest-specific.

    Parameters
    ----------
    classifier:
        A ``ToneClassifier`` that has already been ``.run()``-ed.
    top_n:
        Number of top features to display.
    output_path:
        Where to save the PNG. Defaults to ``RESULTS_DIR``.

    Returns
    -------
    pd.DataFrame
        Feature importance table sorted descending.
    """
    output_path = output_path or (classifier.results_dir / "feature_importance.png")

    result = permutation_importance(
        classifier._best_model,
        classifier._X_test,
        classifier._y_test,
        n_repeats=15,
        random_state=RANDOM_STATE,
        scoring="f1_macro",
        n_jobs=-1,
    )

    importance_df = pd.DataFrame({
        "feature": classifier._feature_cols,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)

    # Save full table
    csv_path = output_path.with_suffix(".csv")
    importance_df.to_csv(csv_path, index=False)
    logger.info("Feature importance table saved to %s", csv_path)

    # Plot top N
    top = importance_df.head(top_n).copy()
    top = top.sort_values("importance_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#2166ac" if v > 0 else "#b2182b" for v in top["importance_mean"]]
    ax.barh(
        top["feature"],
        top["importance_mean"],
        xerr=top["importance_std"],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        capsize=3,
    )
    ax.set_xlabel("Mean F1-Macro Decrease (Permutation Importance)", fontsize=11)
    ax.set_title(
        f"Top {top_n} Feature Importance -- {classifier._best_model_name}",
        fontsize=13,
        pad=12,
    )
    ax.axvline(x=0, color="black", linewidth=0.8, linestyle="-")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Feature importance plot saved to %s", output_path)

    return importance_df


# ---------------------------------------------------------------------------
# Learning curves
# ---------------------------------------------------------------------------


def _plot_learning_curves(
    classifier: ToneClassifier,
    output_path: Optional[Path] = None,
) -> None:
    """Plot learning curves for the best model.

    Shows training and cross-validation F1-macro as a function of
    training set size, indicating whether the model would benefit
    from more data.

    Parameters
    ----------
    classifier:
        A ``ToneClassifier`` that has already been ``.run()``-ed.
    output_path:
        Where to save the PNG.
    """
    output_path = output_path or (classifier.results_dir / "learning_curves.png")

    train_sizes, train_scores, val_scores = learning_curve(
        classifier._best_model,
        classifier._X_train,
        classifier._y_train,
        cv=CV_FOLDS,
        scoring="f1_macro",
        train_sizes=np.linspace(0.2, 1.0, 8),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        shuffle=True,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.fill_between(
        train_sizes, train_mean - train_std, train_mean + train_std,
        alpha=0.15, color="#2166ac",
    )
    ax.fill_between(
        train_sizes, val_mean - val_std, val_mean + val_std,
        alpha=0.15, color="#b2182b",
    )
    ax.plot(
        train_sizes, train_mean, "o-",
        color="#2166ac", linewidth=2, label="Training score",
    )
    ax.plot(
        train_sizes, val_mean, "o-",
        color="#b2182b", linewidth=2, label="Cross-validation score",
    )

    ax.set_xlabel("Training Set Size", fontsize=12)
    ax.set_ylabel("F1-Macro Score", fontsize=12)
    ax.set_title(
        f"Learning Curves -- {classifier._best_model_name}",
        fontsize=13,
        pad=12,
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Learning curves saved to %s", output_path)


# ---------------------------------------------------------------------------
# Model comparison bar chart
# ---------------------------------------------------------------------------


def _plot_model_comparison(
    classifier: ToneClassifier,
    output_path: Optional[Path] = None,
) -> None:
    """Plot a grouped bar chart comparing all models on CV and Val F1.

    Parameters
    ----------
    classifier:
        A ``ToneClassifier`` that has already been ``.run()``-ed.
    output_path:
        Where to save the PNG.
    """
    output_path = output_path or (classifier.results_dir / "model_comparison.png")
    df = classifier._comparison_df.copy()

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(df))
    width = 0.35

    bars_cv = ax.bar(
        x - width / 2, df["cv_f1_macro_mean"], width,
        yerr=df["cv_f1_macro_std"], capsize=4,
        label="CV F1-Macro", color="#4C72B0", edgecolor="white",
    )
    bars_val = ax.bar(
        x + width / 2, df["val_f1_macro"], width,
        label="Val F1-Macro", color="#DD8452", edgecolor="white",
    )

    for bar in bars_cv:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.02,
            f"{h:.3f}", ha="center", va="bottom", fontsize=8,
        )
    for bar in bars_val:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.02,
            f"{h:.3f}", ha="center", va="bottom", fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.replace("_", "\n") for m in df["model"]],
        fontsize=9,
    )
    ax.set_ylabel("F1-Macro Score", fontsize=12)
    ax.set_title("Model Comparison", fontsize=13, pad=12)
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Model comparison chart saved to %s", output_path)


# ---------------------------------------------------------------------------
# SHAP helpers
# ---------------------------------------------------------------------------


def _suppress_shap_logging() -> None:
    """Silence all SHAP-internal loggers.

    SHAP creates loggers lazily and dumps full numpy arrays at INFO
    level. This kills every shap-related logger at ERROR level.
    """
    logging.getLogger("shap").setLevel(logging.ERROR)
    for name in list(logging.Logger.manager.loggerDict):
        if "shap" in name.lower():
            logging.getLogger(name).setLevel(logging.ERROR)


def _normalise_shap_values(shap_values: object) -> list[np.ndarray]:
    """Convert SHAP output to a consistent list-of-2D-arrays format.

    Newer SHAP versions return a 3D ndarray ``(samples, features, classes)``
    instead of a list of 2D arrays. This function normalises both formats.

    Parameters
    ----------
    shap_values:
        Raw output from ``explainer.shap_values()``.

    Returns
    -------
    list[np.ndarray]
        One ``(samples, features)`` array per class.
    """
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        return [shap_values[:, :, i] for i in range(shap_values.shape[2])]
    if isinstance(shap_values, list):
        return shap_values
    return [shap_values]


# ---------------------------------------------------------------------------
# SHAP analysis
# ---------------------------------------------------------------------------


def _plot_shap_analysis(
    classifier: ToneClassifier,
    output_path: Optional[Path] = None,
) -> None:
    """Generate SHAP summary and per-class bar plots.

    Uses ``TreeExplainer`` for tree-based models (fast, exact) and falls
    back to ``KernelExplainer`` with a background sample for others.
    KernelExplainer is capped at ``_KERNEL_SHAP_MAX_SAMPLES`` test
    samples and wrapped in a stdout redirect to suppress verbose output.

    Parameters
    ----------
    classifier:
        A ``ToneClassifier`` that has already been ``.run()``-ed.
    output_path:
        Where to save the PNG.
    """
    try:
        import shap
    except ImportError:
        logger.warning(
            "shap is not installed. Skipping SHAP analysis. "
            "Install with: pip install shap"
        )
        return

    _suppress_shap_logging()

    output_path = output_path or (classifier.results_dir / "shap_summary.png")

    # Scale data using the pipeline's scaler
    pipeline = classifier._best_model
    scaler = pipeline.named_steps["scaler"]
    clf = pipeline.named_steps["clf"]

    X_train_scaled = scaler.transform(classifier._X_train)
    X_test_scaled = scaler.transform(classifier._X_test)

    feature_names = classifier._feature_cols
    class_names = classifier._class_names

    # --- Choose explainer based on model type ---
    inner_clf = clf
    use_tree = False

    if isinstance(clf, VotingClassifier):
        for name, est in clf.named_estimators_.items():
            if isinstance(est, RandomForestClassifier):
                inner_clf = est
                use_tree = True
                logger.info(
                    "SHAP: using TreeExplainer on RF sub-estimator of ensemble."
                )
                break

    if not use_tree and isinstance(clf, RandomForestClassifier):
        inner_clf = clf
        use_tree = True

    # --- Compute SHAP values ---
    if use_tree:
        explainer = shap.TreeExplainer(inner_clf)
        shap_values_raw = explainer.shap_values(X_test_scaled)
    else:
        background = shap.kmeans(X_train_scaled, 50)
        explainer = shap.KernelExplainer(clf.predict_proba, background)
        X_subset = X_test_scaled[:_KERNEL_SHAP_MAX_SAMPLES]
        # Redirect stdout to suppress KernelExplainer's verbose array dumps
        with contextlib.redirect_stdout(io.StringIO()):
            shap_values_raw = explainer.shap_values(X_subset, nsamples=100)

    shap_list = _normalise_shap_values(shap_values_raw)

    # --- Summary bar plot (top 20, averaged across classes) ---
    shap_abs = np.mean([np.abs(sv) for sv in shap_list], axis=0)

    mean_shap = pd.DataFrame(
        shap_abs, columns=feature_names
    ).mean().sort_values(ascending=True)

    top_features = mean_shap.tail(20)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(
        top_features.index,
        top_features.values,
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_xlabel("Mean |SHAP Value| (Across All Classes)", fontsize=11)
    ax.set_title("SHAP Feature Importance (Top 20)", fontsize=13, pad=12)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("SHAP summary plot saved to %s", output_path)

    # --- Per-class SHAP bar plot (top 10 per class) ---
    if len(shap_list) == len(class_names):
        per_class_path = output_path.parent / "shap_per_class.png"
        n_classes = len(class_names)
        n_cols = min(3, n_classes)
        n_rows = (n_classes + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
        axes_flat = np.array(axes).flatten() if n_classes > 1 else [axes]

        for idx, (class_name, sv) in enumerate(zip(class_names, shap_list)):
            ax = axes_flat[idx]
            class_mean = pd.Series(
                np.abs(sv).mean(axis=0), index=feature_names
            ).sort_values(ascending=True).tail(10)

            ax.barh(
                class_mean.index, class_mean.values,
                color=f"C{idx}", edgecolor="white", linewidth=0.5,
            )
            ax.set_title(f"{class_name}", fontsize=11)
            ax.set_xlabel("Mean |SHAP|", fontsize=9)

        for idx in range(n_classes, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.suptitle(
            "Per-Class SHAP Feature Importance (Top 10)",
            fontsize=14, y=1.01,
        )
        plt.tight_layout()
        fig.savefig(per_class_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Per-class SHAP plot saved to %s", per_class_path)

    # --- Save raw SHAP values to CSV ---
    for i, class_name in enumerate(class_names):
        if i < len(shap_list):
            sv_df = pd.DataFrame(shap_list[i], columns=feature_names)
            sv_df.to_csv(
                output_path.parent / f"shap_values_{class_name}.csv",
                index=False,
            )
    logger.info("SHAP value CSVs saved to %s", output_path.parent)


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------


class ModelAnalyser:
    """Run extended analysis on a trained ToneClassifier.

    Parameters
    ----------
    classifier:
        A ``ToneClassifier`` instance that has already completed ``.run()``.
    """

    def __init__(self, classifier: ToneClassifier) -> None:
        if classifier._best_model is None:
            raise RuntimeError(
                "Classifier has not been trained. Call classifier.run() first."
            )
        self.classifier = classifier

    def run(self) -> None:
        """Execute all analysis steps.

        Runs in order: permutation feature importance, learning curves,
        model comparison chart, and SHAP analysis.
        """
        logger.info("Running extended model analysis ...")

        logger.info("Computing permutation feature importance ...")
        _plot_feature_importance(self.classifier)

        logger.info("Generating learning curves ...")
        _plot_learning_curves(self.classifier)

        logger.info("Generating model comparison chart ...")
        _plot_model_comparison(self.classifier)

        logger.info("Running SHAP analysis ...")
        _plot_shap_analysis(self.classifier)

        logger.info("Extended model analysis complete.")
