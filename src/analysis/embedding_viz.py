"""Feature space visualization using t-SNE and UMAP embeddings.

Projects the 90-dimensional feature space into 2D to reveal
class separability and explain classification confusion patterns.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE, RESULTS_DIR

if TYPE_CHECKING:
    from src.classification.tone_classifier import ToneClassifier

logger = logging.getLogger(__name__)

# Consistent palette across all plots
TONE_COLORS = {
    "calm_description": "#2166ac",
    "character_dialogue": "#d6604d",
    "dramatic_emphasis": "#4daf4a",
    "suspense": "#ff7f00",
    "urgency": "#984ea3",
}

TONE_MARKERS = {
    "calm_description": "o",
    "character_dialogue": "s",
    "dramatic_emphasis": "D",
    "suspense": "^",
    "urgency": "v",
}


def _run_embeddings(X_scaled: np.ndarray) -> dict[str, np.ndarray]:
    """Compute t-SNE and UMAP 2D embeddings.

    Parameters
    ----------
    X_scaled:
        Standardised feature matrix (n_samples, n_features).

    Returns
    -------
    dict[str, np.ndarray]
        Mapping of method name to (n_samples, 2) embedding.
    """
    embeddings = {}

    # t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, len(X_scaled) - 1),
        random_state=RANDOM_STATE,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
    )
    embeddings["t-SNE"] = tsne.fit_transform(X_scaled)

    # UMAP
    try:
        import umap
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            random_state=RANDOM_STATE,
            metric="euclidean",
        )
        embeddings["UMAP"] = reducer.fit_transform(X_scaled)
    except ImportError:
        logger.warning(
            "umap-learn not installed. Skipping UMAP. "
            "Install with: pip install umap-learn"
        )

    return embeddings


# ---------------------------------------------------------------------------
# Valence-Arousal circumplex projection (Chaturvedi et al., 2022)
# ---------------------------------------------------------------------------

# Anchors derived from Russell's circumplex model and the RAVDESS
# emotion-to-tone mapping used in this pipeline.
# Coordinates: (valence, arousal) in [-1, 1] range.
TONE_VA_ANCHORS: dict[str, tuple[float, float]] = {
    "calm_description":    ( 0.3,  -0.6),   # neutral/calm: moderate V, low A
    "character_dialogue":  ( 0.7,   0.5),   # happy/surprised: high V, high A
    "dramatic_emphasis":   (-0.5,  -0.2),   # sad/disgust: low V, low-mid A
    "suspense":            (-0.6,   0.4),   # fearful: low V, high A
    "urgency":             (-0.3,   0.8),   # angry: low-mid V, high A
}


def plot_valence_arousal(
    classifier: "ToneClassifier",
    output_dir: Optional[Path] = None,
) -> None:
    """Project audio features onto the Russell valence-arousal circumplex.

    Uses acoustic proxies for valence and arousal following the approach
    described in Chaturvedi et al. (2022), where:
      - **Arousal** correlates with RMS energy, spectral centroid, and ZCR
      - **Valence** correlates with mode (major/minor), spectral contrast,
        and pitch variation

    For speech (rather than music), we approximate:
      - Arousal = standardised mean of [rms_mean, spectral_centroid_mean, zcr_mean]
      - Valence = standardised mean of [pitch_mean, chroma_1_mean, spectral_bandwidth_mean]

    Parameters
    ----------
    classifier:
        A ``ToneClassifier`` that has completed ``.run()``.
    output_dir:
        Where to save the output PNG.

    References
    ----------
    Chaturvedi, V., Garg, A., Kaur, A.B., Varshney, V., & Parashar, A. (2022).
    Machine learning model for mapping of music mood and human emotion based
    on physiological signals. *Multimedia Tools and Applications*, 81,
    https://doi.org/10.1007/s11042-021-11650-0
    """
    output_dir = output_dir or classifier.results_dir

    features_csv = classifier.features_csv
    df = pd.read_csv(features_csv)

    # --- Compute arousal and valence proxies ---
    from sklearn.preprocessing import StandardScaler

    arousal_cols = ["rms_mean", "spectral_centroid_mean", "zcr_mean"]
    valence_cols = ["pitch_mean", "chroma_1_mean", "spectral_bandwidth_mean"]

    # Verify columns exist
    missing = [c for c in arousal_cols + valence_cols if c not in df.columns]
    if missing:
        logger.warning(
            "Cannot compute V-A projection: missing columns %s", missing
        )
        return

    scaler = StandardScaler()
    all_cols = arousal_cols + valence_cols
    scaled = pd.DataFrame(
        scaler.fit_transform(df[all_cols]),
        columns=all_cols,
    )

    df["arousal"] = scaled[arousal_cols].mean(axis=1)
    df["valence"] = scaled[valence_cols].mean(axis=1)

    # --- Plot circumplex ---
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw quadrant lines and labels
    ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="-", alpha=0.4)
    ax.axvline(x=0, color="gray", linewidth=0.8, linestyle="-", alpha=0.4)

    # Quadrant annotations
    ax.text( 1.8,  1.8, "EXCITED\n(high V, high A)", ha="center", fontsize=9, alpha=0.4)
    ax.text(-1.8,  1.8, "TENSE\n(low V, high A)", ha="center", fontsize=9, alpha=0.4)
    ax.text(-1.8, -1.8, "SAD\n(low V, low A)", ha="center", fontsize=9, alpha=0.4)
    ax.text( 1.8, -1.8, "CALM\n(high V, low A)", ha="center", fontsize=9, alpha=0.4)

    # Draw unit circle (Russell's circumplex reference)
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_r = 2.0
    ax.plot(
        circle_r * np.cos(theta),
        circle_r * np.sin(theta),
        color="gray", linewidth=0.8, linestyle="--", alpha=0.3,
    )

    # Scatter by tone
    unique_tones = sorted(df["narrative_tone"].unique())
    for tone in unique_tones:
        mask = df["narrative_tone"] == tone
        ax.scatter(
            df.loc[mask, "valence"],
            df.loc[mask, "arousal"],
            c=TONE_COLORS.get(tone, "gray"),
            marker=TONE_MARKERS.get(tone, "o"),
            s=70, alpha=0.65, edgecolors="white", linewidth=0.5,
            label=tone.replace("_", " "),
        )

    # Plot theoretical anchors
    for tone, (v, a) in TONE_VA_ANCHORS.items():
        # Scale anchors to match data range
        ax.scatter(
            v * 2, a * 2,
            marker="X", s=200, c="black", zorder=10, edgecolors="white",
            linewidth=1.5,
        )
        ax.annotate(
            tone.replace("_", "\n"),
            (v * 2, a * 2),
            textcoords="offset points",
            xytext=(12, 8),
            fontsize=8,
            fontweight="bold",
            alpha=0.7,
        )

    ax.set_xlabel("Valence (negative = unpleasant, positive = pleasant)", fontsize=12)
    ax.set_ylabel("Arousal (low = calm, high = excited)", fontsize=12)
    ax.set_title(
        "Russell Valence-Arousal Circumplex Projection\n"
        "Acoustic proxies: Arousal ~ [RMS, Centroid, ZCR], "
        "Valence ~ [Pitch, Chroma, Bandwidth]\n"
        "(Chaturvedi et al., 2022)",
        fontsize=12, pad=14,
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_aspect("equal")
    lim = max(abs(df["valence"].max()), abs(df["arousal"].max()), 2.5) + 0.3
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid(alpha=0.15)

    plt.tight_layout()
    out = output_dir / "valence_arousal_circumplex.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Valence-arousal circumplex saved to %s", out)


def plot_feature_embeddings(
    classifier: ToneClassifier,
    output_dir: Optional[Path] = None,
) -> None:
    """Generate t-SNE and UMAP scatter plots of the full dataset.

    Each point is colored by its true narrative tone label.
    Misclassified test samples are highlighted with red borders.

    Parameters
    ----------
    classifier:
        A ``ToneClassifier`` that has completed ``.run()``.
    output_dir:
        Directory for output PNGs. Defaults to ``RESULTS_DIR``.
    """
    if classifier._best_model is None:
        raise RuntimeError("Classifier not trained. Call run() first.")

    output_dir = output_dir or classifier.results_dir

    # Reload full dataset for embedding (not just train/test)
    features_csv = classifier.features_csv
    df = pd.read_csv(features_csv)
    meta_cols = {"file_path", "segment_idx", "emotion_label", "narrative_tone"}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X_all = df[feature_cols].values
    y_labels = df["narrative_tone"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # Get predictions on full dataset for misclassification overlay
    y_pred_enc = classifier._best_model.predict(df[feature_cols])
    y_pred = classifier._le.inverse_transform(y_pred_enc)
    is_wrong = y_labels != y_pred

    embeddings = _run_embeddings(X_scaled)

    for method_name, coords in embeddings.items():
        _plot_single_embedding(
            coords, y_labels, is_wrong,
            method_name=method_name,
            output_path=output_dir / f"embedding_{method_name.lower().replace('-', '')}.png",
        )

    # Combined side-by-side if both exist
    if len(embeddings) == 2:
        _plot_combined(
            embeddings, y_labels, is_wrong,
            output_path=output_dir / "embedding_combined.png",
        )
    
    plot_valence_arousal(classifier, output_dir)


    logger.info("Embedding visualizations saved to %s", output_dir)


def _plot_single_embedding(
    coords: np.ndarray,
    y_labels: np.ndarray,
    is_wrong: np.ndarray,
    method_name: str,
    output_path: Path,
) -> None:
    """Plot a single 2D embedding scatter.

    Parameters
    ----------
    coords:
        (n_samples, 2) embedding coordinates.
    y_labels:
        True narrative tone labels.
    is_wrong:
        Boolean mask of misclassified samples.
    method_name:
        Name for the plot title.
    output_path:
        Where to save the PNG.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    unique_tones = sorted(set(y_labels))
    for tone in unique_tones:
        mask = y_labels == tone
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=TONE_COLORS.get(tone, "gray"),
            marker=TONE_MARKERS.get(tone, "o"),
            s=60, alpha=0.7, edgecolors="white", linewidth=0.5,
            label=tone.replace("_", " "),
        )

    # Highlight misclassified with red border
    if is_wrong.any():
        ax.scatter(
            coords[is_wrong, 0], coords[is_wrong, 1],
            facecolors="none", edgecolors="#e31a1c", linewidths=2,
            s=120, zorder=5, label="misclassified",
        )

    ax.set_xlabel(f"{method_name} Dimension 1", fontsize=11)
    ax.set_ylabel(f"{method_name} Dimension 2", fontsize=11)
    ax.set_title(
        f"Narrative Tone Feature Space ({method_name})\n"
        f"{len(y_labels)} RAVDESS recordings, 90-dim features projected to 2D",
        fontsize=13, pad=12,
    )
    ax.legend(
        loc="best", fontsize=9, framealpha=0.9,
        markerscale=1.2,
    )
    ax.grid(alpha=0.2)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("%s embedding saved to %s", method_name, output_path)


def _plot_combined(
    embeddings: dict[str, np.ndarray],
    y_labels: np.ndarray,
    is_wrong: np.ndarray,
    output_path: Path,
) -> None:
    """Plot t-SNE and UMAP side by side.

    Parameters
    ----------
    embeddings:
        Dict with "t-SNE" and "UMAP" keys.
    y_labels:
        True narrative tone labels.
    is_wrong:
        Boolean mask of misclassified samples.
    output_path:
        Where to save the PNG.
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    unique_tones = sorted(set(y_labels))

    for ax, (method_name, coords) in zip(axes, embeddings.items()):
        for tone in unique_tones:
            mask = y_labels == tone
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=TONE_COLORS.get(tone, "gray"),
                marker=TONE_MARKERS.get(tone, "o"),
                s=60, alpha=0.7, edgecolors="white", linewidth=0.5,
                label=tone.replace("_", " "),
            )

        if is_wrong.any():
            ax.scatter(
                coords[is_wrong, 0], coords[is_wrong, 1],
                facecolors="none", edgecolors="#e31a1c", linewidths=2,
                s=120, zorder=5,
            )

        ax.set_xlabel(f"{method_name} Dim 1", fontsize=11)
        ax.set_ylabel(f"{method_name} Dim 2", fontsize=11)
        ax.set_title(method_name, fontsize=13)
        ax.grid(alpha=0.2)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    # Add misclassified marker to legend
    handles.append(Line2D(
        [0], [0], marker="o", color="w",
        markerfacecolor="none", markeredgecolor="#e31a1c",
        markeredgewidth=2, markersize=10,
    ))
    labels.append("misclassified")
    fig.legend(
        handles, labels, loc="lower center",
        ncol=len(labels), fontsize=10, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "Narrative Tone Feature Space -- 90-dim Acoustic Features Projected to 2D",
        fontsize=14, y=1.02,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Combined embedding saved to %s", output_path)