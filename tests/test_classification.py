"""Tests for the tone classification module."""

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
def synthetic_features_csv(tmp_path: Path) -> Path:
    """Create a synthetic feature CSV for testing without real audio."""
    rng = np.random.default_rng(RANDOM_STATE)
    n_samples_per_class = 30
    feature_cols = [f"feat_{i}" for i in range(20)]

    rows = []
    for tone in NARRATIVE_TONES:
        for _ in range(n_samples_per_class):
            feat_vals = rng.normal(size=len(feature_cols)).tolist()
            rows.append({
                "file_path": f"/fake/{tone}/{rng.integers(9999)}.wav",
                "segment_idx": 0,
                "emotion_label": "neutral",
                "narrative_tone": tone,
                **dict(zip(feature_cols, feat_vals)),
            })

    df = pd.DataFrame(rows)
    csv_path = tmp_path / "features.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestToneClassifier:
    def test_run_returns_metrics(
        self, synthetic_features_csv: Path, tmp_path: Path
    ) -> None:
        from src.classification.tone_classifier import ToneClassifier

        classifier = ToneClassifier(
            features_csv=synthetic_features_csv,
            results_dir=tmp_path / "results",
            models_dir=tmp_path / "models",
        )
        results = classifier.run()

        assert "best_model" in results
        assert "test_accuracy" in results
        assert 0.0 <= results["test_accuracy"] <= 1.0
        assert 0.0 <= results["test_f1_macro"] <= 1.0

    def test_model_saved(
        self, synthetic_features_csv: Path, tmp_path: Path
    ) -> None:
        from src.classification.tone_classifier import ToneClassifier

        models_dir = tmp_path / "models"
        classifier = ToneClassifier(
            features_csv=synthetic_features_csv,
            results_dir=tmp_path / "results",
            models_dir=models_dir,
        )
        classifier.run()

        assert (models_dir / "best_model.pkl").exists()
        assert (models_dir / "label_encoder.pkl").exists()

    def test_classification_report_saved(
        self, synthetic_features_csv: Path, tmp_path: Path
    ) -> None:
        from src.classification.tone_classifier import ToneClassifier

        results_dir = tmp_path / "results"
        classifier = ToneClassifier(
            features_csv=synthetic_features_csv,
            results_dir=results_dir,
            models_dir=tmp_path / "models",
        )
        classifier.run()

        assert (results_dir / "classification_report.txt").exists()
        assert (results_dir / "confusion_matrix.png").exists()
        assert (results_dir / "model_comparison.csv").exists()

    def test_predict_after_run(
        self, synthetic_features_csv: Path, tmp_path: Path
    ) -> None:
        import joblib
        from src.classification.tone_classifier import ToneClassifier

        classifier = ToneClassifier(
            features_csv=synthetic_features_csv,
            results_dir=tmp_path / "results",
            models_dir=tmp_path / "models",
        )
        classifier.run()

        # Load a few rows from the CSV to use as prediction input
        df = pd.read_csv(synthetic_features_csv)
        feature_cols = [c for c in df.columns
                        if c not in {"file_path", "segment_idx", "emotion_label", "narrative_tone"}]
        X_new = df[feature_cols].head(5)

        preds = classifier.predict(X_new)
        assert len(preds) == 5
        assert all(p in NARRATIVE_TONES for p in preds)

    def test_predict_before_run_raises(
        self, synthetic_features_csv: Path, tmp_path: Path
    ) -> None:
        from src.classification.tone_classifier import ToneClassifier

        classifier = ToneClassifier(
            features_csv=synthetic_features_csv,
            results_dir=tmp_path / "results",
            models_dir=tmp_path / "models",
        )
        with pytest.raises(RuntimeError, match="Call run\\(\\) first"):
            classifier.predict(pd.DataFrame())
