"""Tests for the audio preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest
import soundfile as sf


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sine_wave_file(tmp_path: Path) -> Path:
    """Write a 3-second 440 Hz sine wave to a RAVDESS-named temp file."""
    sr = 22050
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    # Valid RAVDESS filename: mod-voc-emo-int-stmt-rep-actor
    wav_path = tmp_path / "03-01-03-01-01-01-01.wav"
    sf.write(str(wav_path), y, sr)
    return wav_path


@pytest.fixture
def sine_wave_dir(sine_wave_file: Path) -> Path:
    """Return the directory containing the sine wave fixture."""
    return sine_wave_file.parent


# ---------------------------------------------------------------------------
# Tests: normalisation
# ---------------------------------------------------------------------------


class TestNormalizeAudio:
    def test_peak_at_target(self) -> None:
        from src.preprocessing.audio_pipeline import _normalize_audio

        y = np.array([0.1, -0.5, 0.3, -0.2], dtype=np.float64)
        y_norm = _normalize_audio(y, target_db=-3.0)
        target_amp = 10 ** (-3.0 / 20.0)
        assert np.isclose(np.max(np.abs(y_norm)), target_amp, atol=1e-6)

    def test_silence_unchanged(self) -> None:
        from src.preprocessing.audio_pipeline import _normalize_audio

        y = np.zeros(100, dtype=np.float64)
        y_norm = _normalize_audio(y, target_db=-3.0)
        np.testing.assert_array_equal(y, y_norm)


# ---------------------------------------------------------------------------
# Tests: feature extraction
# ---------------------------------------------------------------------------


class TestExtractFeatures:
    def test_output_is_dict(self, sine_wave_file: Path) -> None:
        import librosa
        from src.preprocessing.audio_pipeline import extract_features

        y, sr = librosa.load(str(sine_wave_file), sr=22050)
        feats = extract_features(y, sr)
        assert isinstance(feats, dict)

    def test_feature_count(self, sine_wave_file: Path) -> None:
        import librosa
        from src.preprocessing.audio_pipeline import extract_features

        y, sr = librosa.load(str(sine_wave_file), sr=22050)
        feats = extract_features(y, sr)
        # 13*2 MFCC + 13*2 delta + 12*2 chroma + 2+2+2+2+2+2 + 1+1 = 90
        assert len(feats) == 90, f"Expected 90 features, got {len(feats)}"

    def test_no_nan_features(self, sine_wave_file: Path) -> None:
        import librosa
        from src.preprocessing.audio_pipeline import extract_features

        y, sr = librosa.load(str(sine_wave_file), sr=22050)
        feats = extract_features(y, sr)
        for name, val in feats.items():
            assert not np.isnan(val), f"Feature '{name}' is NaN"


# ---------------------------------------------------------------------------
# Tests: RAVDESS filename parsing
# ---------------------------------------------------------------------------


class TestParseRavdessFilename:
    def test_valid_filename(self, tmp_path: Path) -> None:
        from src.preprocessing.audio_pipeline import _parse_ravdess_filename

        p = tmp_path / "03-01-06-01-02-01-12.wav"
        emotion, tone = _parse_ravdess_filename(p)
        assert emotion == "fearful"
        assert tone == "suspense"

    def test_invalid_filename_raises(self, tmp_path: Path) -> None:
        from src.preprocessing.audio_pipeline import _parse_ravdess_filename

        p = tmp_path / "not_ravdess.wav"
        with pytest.raises(ValueError, match="7 dash-separated"):
            _parse_ravdess_filename(p)


# ---------------------------------------------------------------------------
# Tests: AudioFeatureExtractor integration
# ---------------------------------------------------------------------------


class TestAudioFeatureExtractor:
    def test_process_directory(
        self, sine_wave_dir: Path, tmp_path: Path
    ) -> None:
        from src.preprocessing.audio_pipeline import AudioFeatureExtractor

        out_csv = tmp_path / "features.csv"
        extractor = AudioFeatureExtractor(output_path=out_csv)
        df = extractor.process_directory(sine_wave_dir, subset=5)

        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 1
        assert "narrative_tone" in df.columns
        assert "emotion_label" in df.columns
        assert out_csv.exists()

    def test_csv_has_correct_columns(
        self, sine_wave_dir: Path, tmp_path: Path
    ) -> None:
        from src.preprocessing.audio_pipeline import AudioFeatureExtractor

        out_csv = tmp_path / "features2.csv"
        extractor = AudioFeatureExtractor(output_path=out_csv)
        df = extractor.process_directory(sine_wave_dir)

        # Metadata cols + 90 feature cols
        assert df.shape[1] == 90 + 4, (
            f"Expected 94 columns (90 features + 4 metadata), got {df.shape[1]}"
        )
