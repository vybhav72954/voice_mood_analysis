"""Bonus Task — Storytelling vs Conversational Speech Analysis.

Identifies acoustic features that distinguish deliberate storytelling
narration from conversational speech using RAVDESS clips as proxies.
Applies non-parametric Mann-Whitney U tests and produces comparison plots.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.config import FEATURES_DIR, RESULTS_DIR

logger = logging.getLogger(__name__)

# Features most indicative of storytelling vs conversational style
STORYTELLING_FEATURES = [
    "pitch_std",           # dramatic pitch variation
    "rms_std",             # energy dynamics
    "spectral_centroid_std",  # tonal variation
    "zcr_std",             # articulation variation
    "spectral_rolloff_std",
    "tempo",               # pacing
    "duration",            # clip length
]

# Emotions treated as proxies for each speech style
STORYTELLING_TONES = {"calm_description", "dramatic_emphasis", "suspense"}
CONVERSATIONAL_TONES = {"character_dialogue", "urgency"}


class StorytellingAnalyser:
    """Analyse acoustic features differentiating storytelling from conversation.

    Parameters
    ----------
    features_csv:
        Path to ``feature_dataset.csv`` produced by Task 1.
    results_dir:
        Where output plots are saved.
    """

    def __init__(
        self,
        features_csv: Optional[Path] = None,
        results_dir: Optional[Path] = None,
    ) -> None:
        self.features_csv = features_csv or (FEATURES_DIR / "feature_dataset.csv")
        self.results_dir = results_dir or RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """Execute the full analysis and return a stats summary DataFrame.

        Returns
        -------
        pd.DataFrame
            One row per feature with Mann-Whitney U statistic and p-value.
        """
        df = self._load_and_label()
        stats_df = self._mann_whitney_tests(df)
        self._plot_feature_distributions(df, stats_df)
        self._plot_correlation_heatmap(df)
        self._log_summary(stats_df)
        return stats_df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_and_label(self) -> pd.DataFrame:
        """Load features and add a binary ``speech_style`` column.

        Returns
        -------
        pd.DataFrame
            Feature DataFrame with ``speech_style`` ∈ {storytelling, conversational}.

        Raises
        ------
        FileNotFoundError
            If the feature CSV does not exist.
        """
        if not self.features_csv.exists():
            raise FileNotFoundError(
                f"Feature dataset not found at {self.features_csv}."
            )
        df = pd.read_csv(self.features_csv)

        storytelling_mask = df["narrative_tone"].isin(STORYTELLING_TONES)
        conversational_mask = df["narrative_tone"].isin(CONVERSATIONAL_TONES)

        df = df[storytelling_mask | conversational_mask].copy()
        df["speech_style"] = np.where(
            df["narrative_tone"].isin(STORYTELLING_TONES),
            "storytelling",
            "conversational",
        )

        logger.info(
            "Storytelling samples: %d | Conversational samples: %d",
            storytelling_mask.sum(),
            conversational_mask.sum(),
        )
        return df

    def _mann_whitney_tests(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run Mann-Whitney U tests for each feature.

        Non-parametric test chosen because normality of audio features
        cannot be assumed.

        Parameters
        ----------
        df:
            DataFrame with ``speech_style`` and feature columns.

        Returns
        -------
        pd.DataFrame
            Columns: ``feature``, ``u_statistic``, ``p_value``,
            ``significant``, ``effect_size_r``.
        """
        story = df[df["speech_style"] == "storytelling"]
        convo = df[df["speech_style"] == "conversational"]

        available = [f for f in STORYTELLING_FEATURES if f in df.columns]
        rows = []
        for feat in available:
            story_vals = story[feat].dropna()
            convo_vals = convo[feat].dropna()
            u_stat, p_val = stats.mannwhitneyu(
                story_vals,
                convo_vals,
                alternative="two-sided",
            )
            # Rank-biserial correlation: r = 1 - 2U/(n1*n2)
            n1, n2 = len(story_vals), len(convo_vals)
            effect_r = 1.0 - (2.0 * u_stat) / (n1 * n2) if n1 * n2 > 0 else 0.0

            rows.append({
                "feature": feat,
                "u_statistic": round(u_stat, 2),
                "p_value": round(p_val, 6),
                "significant": p_val < 0.05,
                "effect_size_r": round(abs(effect_r), 4),
            })

        stats_df = pd.DataFrame(rows).sort_values("p_value")
        stats_path = self.results_dir / "storytelling_stats.csv"
        stats_df.to_csv(stats_path, index=False)
        logger.info("Statistical results saved to %s", stats_path)
        return stats_df

    def _plot_feature_distributions(
        self, df: pd.DataFrame, stats_df: pd.DataFrame
    ) -> None:
        """Generate side-by-side box plots for each analysed feature.

        Parameters
        ----------
        df:
            Feature DataFrame with ``speech_style`` column.
        stats_df:
            Mann-Whitney test results used to annotate significance.
        """
        available = [f for f in STORYTELLING_FEATURES if f in df.columns]
        n_features = len(available)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes_flat = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

        sig_map = dict(zip(stats_df["feature"], stats_df["significant"]))
        p_map = dict(zip(stats_df["feature"], stats_df["p_value"]))

        for idx, feat in enumerate(available):
            ax = axes_flat[idx]
            sns.boxplot(
                data=df,
                x="speech_style",
                y=feat,
                hue="speech_style",
                palette={"storytelling": "#4C72B0", "conversational": "#DD8452"},
                ax=ax,
                width=0.5,
                legend=False,
            )
            title_suffix = (
                f"p={p_map[feat]:.4f} *" if sig_map.get(feat) else f"p={p_map.get(feat, '?'):.4f}"
            )
            ax.set_title(f"{feat}\n({title_suffix})", fontsize=10)
            ax.set_xlabel("")
            ax.set_ylabel(feat.replace("_", " ").title(), fontsize=9)

        # Hide unused subplots
        for idx in range(n_features, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.suptitle(
            "Acoustic Features: Storytelling vs Conversational Speech",
            fontsize=14, y=1.02,
        )
        plt.tight_layout()
        out = self.results_dir / "storytelling_analysis.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Storytelling analysis plot saved to %s", out)

    def _plot_correlation_heatmap(self, df: pd.DataFrame) -> None:
        """Save a correlation heatmap of the storytelling features.

        Parameters
        ----------
        df:
            Feature DataFrame.
        """
        available = [f for f in STORYTELLING_FEATURES if f in df.columns]
        corr = df[available].corr()

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            ax=ax,
        )
        ax.set_title("Feature Correlation Heatmap (Storytelling Features)", pad=12)
        plt.tight_layout()
        out = self.results_dir / "feature_correlation.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("Correlation heatmap saved to %s", out)

    @staticmethod
    def _log_summary(stats_df: pd.DataFrame) -> None:
        """Log a human-readable summary of significant features.

        Parameters
        ----------
        stats_df:
            Mann-Whitney test results.
        """
        sig = stats_df[stats_df["significant"]]
        if sig.empty:
            logger.info("No features showed statistically significant differences (α=0.05).")
        else:
            logger.info(
                "Significant features (Mann-Whitney U, α=0.05):\n%s",
                sig[["feature", "p_value", "effect_size_r"]].to_string(index=False),
            )
