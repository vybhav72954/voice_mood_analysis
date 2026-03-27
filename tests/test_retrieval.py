"""Tests for the narrative audio retrieval module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import NARRATIVE_TONES, RANDOM_STATE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def retrieval_features_csv(tmp_path: Path) -> Path:
    """Synthetic feature CSV for retrieval tests."""
    rng = np.random.default_rng(RANDOM_STATE)
    n_per_class = 20
    feature_cols = [f"feat_{i}" for i in range(10)]
    # include rms_mean and duration so the retrieval system can use them
    feature_cols += ["rms_mean", "duration"]

    rows = []
    for tone in NARRATIVE_TONES:
        for i in range(n_per_class):
            rows.append({
                "file_path": f"/fake/{tone}/file_{i}.wav",
                "segment_idx": 0,
                "emotion_label": "neutral",
                "narrative_tone": tone,
                **{f: float(rng.normal()) for f in feature_cols[:10]},
                "rms_mean": abs(float(rng.normal(0.05, 0.02))),
                "duration": float(rng.uniform(2.0, 8.0)),
            })

    csv_path = tmp_path / "retrieval_features.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNarrativeAudioRetrieval:
    def test_load_succeeds(self, retrieval_features_csv: Path) -> None:
        from src.retrieval.audio_retrieval import NarrativeAudioRetrieval

        retrieval = NarrativeAudioRetrieval(features_csv=retrieval_features_csv)
        retrieval.load()  # should not raise

    def test_missing_csv_raises(self, tmp_path: Path) -> None:
        from src.retrieval.audio_retrieval import NarrativeAudioRetrieval

        retrieval = NarrativeAudioRetrieval(
            features_csv=tmp_path / "nonexistent.csv"
        )
        with pytest.raises(FileNotFoundError):
            retrieval.load()

    def test_search_before_load_raises(self, retrieval_features_csv: Path) -> None:
        from src.retrieval.audio_retrieval import NarrativeAudioRetrieval

        retrieval = NarrativeAudioRetrieval(features_csv=retrieval_features_csv)
        with pytest.raises(RuntimeError, match="Call load\\(\\)"):
            retrieval.search(tone="suspense")

    def test_tone_filter_returns_correct_tone(
        self, retrieval_features_csv: Path
    ) -> None:
        from src.retrieval.audio_retrieval import NarrativeAudioRetrieval

        retrieval = NarrativeAudioRetrieval(features_csv=retrieval_features_csv).load()
        results = retrieval.search(tone="suspense")
        assert not results.empty
        assert all(results["narrative_tone"] == "suspense")

    def test_invalid_tone_raises(self, retrieval_features_csv: Path) -> None:
        from src.retrieval.audio_retrieval import NarrativeAudioRetrieval

        retrieval = NarrativeAudioRetrieval(features_csv=retrieval_features_csv).load()
        with pytest.raises(ValueError, match="Unknown tone"):
            retrieval.search(tone="nonexistent_tone")

    def test_duration_filter(self, retrieval_features_csv: Path) -> None:
        from src.retrieval.audio_retrieval import NarrativeAudioRetrieval

        retrieval = NarrativeAudioRetrieval(features_csv=retrieval_features_csv).load()
        results = retrieval.search(min_duration=4.0, max_duration=6.0)
        if not results.empty:
            assert all(results["duration"] >= 4.0)
            assert all(results["duration"] <= 6.0)

    def test_energy_filter(self, retrieval_features_csv: Path) -> None:
        from src.retrieval.audio_retrieval import NarrativeAudioRetrieval

        retrieval = NarrativeAudioRetrieval(features_csv=retrieval_features_csv).load()
        results = retrieval.search(energy_level="high")
        assert not results.empty

    def test_combined_filter(self, retrieval_features_csv: Path) -> None:
        from src.retrieval.audio_retrieval import NarrativeAudioRetrieval

        retrieval = NarrativeAudioRetrieval(features_csv=retrieval_features_csv).load()
        results = retrieval.search(
            tone="calm_description", min_duration=3.0
        )
        if not results.empty:
            assert all(results["narrative_tone"] == "calm_description")
            assert all(results["duration"] >= 3.0)

    def test_max_results_respected(self, retrieval_features_csv: Path) -> None:
        from src.retrieval.audio_retrieval import NarrativeAudioRetrieval

        retrieval = NarrativeAudioRetrieval(features_csv=retrieval_features_csv).load()
        results = retrieval.search(max_results=3)
        assert len(results) <= 3
