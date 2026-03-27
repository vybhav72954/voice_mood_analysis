"""Task 4 — Narrative Audio Retrieval System.

Filter-based and cosine-similarity-based retrieval over the feature
dataset produced by Task 1.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

from src.config import (
    ENERGY_HIGH_PERCENTILE,
    ENERGY_LOW_PERCENTILE,
    FEATURES_DIR,
    NARRATIVE_TONES,
    TOP_K_SIMILAR,
)
from src.preprocessing.audio_pipeline import AudioFeatureExtractor, extract_features

logger = logging.getLogger(__name__)

_META_COLS = {"file_path", "segment_idx", "emotion_label", "narrative_tone"}
EnergyLevel = Literal["low", "medium", "high"]


class NarrativeAudioRetrieval:
    """Filter and similarity-based retrieval over narrative audio recordings.

    Parameters
    ----------
    features_csv:
        Path to ``feature_dataset.csv`` produced by Task 1.
    """

    def __init__(self, features_csv: Optional[Path] = None) -> None:
        self.features_csv = features_csv or (FEATURES_DIR / "feature_dataset.csv")
        self._df: Optional[pd.DataFrame] = None
        self._feature_cols: list[str] = []
        self._scaler: Optional[StandardScaler] = None
        self._X_scaled: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> "NarrativeAudioRetrieval":
        """Load the feature dataset and pre-compute energy thresholds.

        Returns
        -------
        NarrativeAudioRetrieval
            Self, for method chaining.
        """
        if not self.features_csv.exists():
            raise FileNotFoundError(
                f"Feature dataset not found at {self.features_csv}. "
                "Run the audio pipeline (Task 1) first."
            )
        self._df = pd.read_csv(self.features_csv)
        self._feature_cols = [c for c in self._df.columns if c not in _META_COLS]

        # Pre-compute energy buckets
        low_thresh = np.percentile(self._df["rms_mean"], ENERGY_LOW_PERCENTILE)
        high_thresh = np.percentile(self._df["rms_mean"], ENERGY_HIGH_PERCENTILE)
        self._df["energy_level"] = pd.cut(
            self._df["rms_mean"],
            bins=[-np.inf, low_thresh, high_thresh, np.inf],
            labels=["low", "medium", "high"],
        )

        # Fit scaler for similarity search
        self._scaler = StandardScaler()
        self._X_scaled = self._scaler.fit_transform(
            self._df[self._feature_cols].values
        )

        logger.info(
            "Retrieval index loaded: %d recordings, %d features.",
            len(self._df), len(self._feature_cols),
        )
        return self

    def search(
        self,
        tone: Optional[str] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        energy_level: Optional[EnergyLevel] = None,
        max_results: int = 10,
    ) -> pd.DataFrame:
        """Filter-based retrieval.

        Parameters
        ----------
        tone:
            Narrative tone to filter on. Must be one of the
            ``NARRATIVE_TONES`` values, or ``None`` to skip this filter.
        min_duration:
            Minimum audio duration in seconds, or ``None``.
        max_duration:
            Maximum audio duration in seconds, or ``None``.
        energy_level:
            One of ``"low"``, ``"medium"``, ``"high"``, or ``None``.
        max_results:
            Maximum number of results to return.

        Returns
        -------
        pd.DataFrame
            Matching recordings with key metadata columns.

        Raises
        ------
        RuntimeError
            If ``load()`` has not been called.
        ValueError
            If *tone* is not a recognised narrative tone.
        """
        self._check_loaded()

        if tone is not None and tone not in NARRATIVE_TONES:
            raise ValueError(
                f"Unknown tone '{tone}'. Choose from: {NARRATIVE_TONES}"
            )

        mask = pd.Series([True] * len(self._df), index=self._df.index)

        if tone is not None:
            mask &= self._df["narrative_tone"] == tone

        if min_duration is not None:
            mask &= self._df["duration"] >= min_duration

        if max_duration is not None:
            mask &= self._df["duration"] <= max_duration

        if energy_level is not None:
            mask &= self._df["energy_level"] == energy_level

        results = self._df[mask].head(max_results).copy()
        results["similarity_score"] = None  # not applicable for filter search

        logger.info(
            "Filter search returned %d / %d recordings.",
            len(results), len(self._df),
        )
        return self._format_results(results)

    def search_similar(
        self,
        query_file: Path,
        top_k: int = TOP_K_SIMILAR,
    ) -> pd.DataFrame:
        """Cosine-similarity retrieval given a query audio file.

        Parameters
        ----------
        query_file:
            Path to a ``.wav`` file to use as the query.
        top_k:
            Number of most-similar recordings to return.

        Returns
        -------
        pd.DataFrame
            Top-*k* most similar recordings with similarity scores.
        """
        self._check_loaded()

        import librosa
        from src.preprocessing.audio_pipeline import _normalize_audio

        from src.config import SAMPLE_RATE
        y, sr = librosa.load(str(query_file), sr=SAMPLE_RATE, mono=True)

        y = _normalize_audio(y)
        query_features = extract_features(y, sr)

        # Build query vector aligned with training feature columns
        query_vec = np.array(
            [query_features.get(col, 0.0) for col in self._feature_cols]
        ).reshape(1, -1)
        query_scaled = self._scaler.transform(query_vec)

        sims = cosine_similarity(query_scaled, self._X_scaled).flatten()
        top_indices = np.argsort(sims)[::-1][:top_k]

        results = self._df.iloc[top_indices].copy()
        results["similarity_score"] = sims[top_indices].round(4)

        logger.info(
            "Similarity search for '%s' returned %d results.",
            query_file.name, len(results),
        )
        return self._format_results(results)

    def print_results(self, df: pd.DataFrame, query_description: str = "") -> None:
        """Pretty-print retrieval results to stdout.

        Parameters
        ----------
        df:
            DataFrame returned by ``search()`` or ``search_similar()``.
        query_description:
            Human-readable description of the query (for the header line).
        """
        if query_description:
            print(f"\n{'='*60}")
            print(f"Query: {query_description}")
            print(f"{'='*60}")

        if df.empty:
            print("  No results found.\n")
            return

        display = df[
            [c for c in ["file_path", "narrative_tone", "duration",
                          "energy_level", "similarity_score"] if c in df.columns]
        ].copy()
        # Shorten file paths for readability
        display["file_path"] = display["file_path"].apply(
            lambda p: Path(p).name
        )
        print(tabulate(display, headers="keys", tablefmt="github", showindex=False))
        print()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_loaded(self) -> None:
        """Raise RuntimeError if load() has not been called."""
        if self._df is None:
            raise RuntimeError("Call load() before querying the retrieval system.")

    @staticmethod
    def _format_results(df: pd.DataFrame) -> pd.DataFrame:
        """Select and reorder output columns.

        Parameters
        ----------
        df:
            Raw results DataFrame.

        Returns
        -------
        pd.DataFrame
            Cleaned output with essential columns only.
        """
        keep = [
            "file_path", "narrative_tone", "duration", "energy_level",
            "rms_mean", "pitch_mean", "similarity_score",
        ]
        available = [c for c in keep if c in df.columns]
        return df[available].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Demo queries (run directly for a quick showcase)
# ---------------------------------------------------------------------------


def run_demo_queries(features_csv: Optional[Path] = None) -> None:
    """Execute five example queries (four filter-based, one similarity) and print results.

    Parameters
    ----------
    features_csv:
        Optional override for the feature dataset path.
    """
    retrieval = NarrativeAudioRetrieval(features_csv=features_csv).load()

    # --- Filter-based queries ---
    filter_queries = [
        {
            "description": 'tone="suspense", min_duration=2.0s',
            "kwargs": {"tone": "suspense", "min_duration": 2.0},
        },
        {
            "description": 'energy_level="high"',
            "kwargs": {"energy_level": "high"},
        },
        {
            "description": 'tone="calm_description", min_duration=4.0s',
            "kwargs": {"tone": "calm_description", "min_duration": 4.0},
        },
        {
            "description": 'tone="urgency", energy_level="high"',
            "kwargs": {"tone": "urgency", "energy_level": "high"},
        },
    ]

    for q in filter_queries:
        results = retrieval.search(**q["kwargs"])
        retrieval.print_results(results, query_description=q["description"])

    # --- Similarity search using the first file in the dataset ---
    first_file = Path(retrieval._df.iloc[0]["file_path"])
    if first_file.exists():
        results = retrieval.search_similar(query_file=first_file, top_k=TOP_K_SIMILAR)
        retrieval.print_results(
            results,
            query_description=f"similarity search (query={first_file.name}, top_k={TOP_K_SIMILAR})",
        )
    else:
        logger.warning(
            "Skipping similarity demo: source file %s not found on disk.",
            first_file,
        )
