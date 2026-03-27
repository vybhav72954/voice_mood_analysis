"""Task 1 — Audio Feature Extraction Pipeline.

Loads RAVDESS audio files, normalises levels, optionally segments long clips,
and extracts a 90-dimensional feature vector per file. Results are written to
``outputs/features/feature_dataset.csv``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from src.config import (
    EMOTION_TO_TONE,
    FEATURES_DIR,
    HOP_LENGTH,
    MAX_DURATION,
    MIN_DURATION,
    N_FFT,
    N_MFCC,
    PEAK_NORMALIZATION_DB,
    RAVDESS_EMOTIONS,
    SAMPLE_RATE,
    SEGMENT_DURATION,
    SEGMENT_OVERLAP,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_ravdess_filename(path: Path) -> tuple[str, str]:
    """Parse emotion label and narrative tone from a RAVDESS filename.

    Parameters
    ----------
    path:
        Path to a RAVDESS ``.wav`` file.

    Returns
    -------
    tuple[str, str]
        ``(emotion_label, narrative_tone)`` derived from the filename.

    Raises
    ------
    ValueError
        If the filename does not conform to the RAVDESS naming convention.
    """
    parts = path.stem.split("-")
    if len(parts) != 7:
        raise ValueError(
            f"Expected 7 dash-separated parts in RAVDESS filename, "
            f"got {len(parts)}: {path.name}"
        )
    emotion_code = parts[2]
    emotion_label = RAVDESS_EMOTIONS.get(emotion_code)
    if emotion_label is None:
        raise ValueError(
            f"Unknown RAVDESS emotion code '{emotion_code}' in {path.name}"
        )
    narrative_tone = EMOTION_TO_TONE[emotion_label]
    return emotion_label, narrative_tone


def _normalize_audio(y: np.ndarray, target_db: float = PEAK_NORMALIZATION_DB) -> np.ndarray:
    """Apply peak normalization so the loudest sample hits ``target_db`` dBFS.

    Parameters
    ----------
    y:
        Raw audio time-series array.
    target_db:
        Target peak level in decibels full-scale (negative value).

    Returns
    -------
    np.ndarray
        Normalised audio array.
    """
    peak = np.max(np.abs(y))
    if peak < 1e-8:
        return y  # silence / near-silence – nothing to normalise
    target_amplitude = 10 ** (target_db / 20.0)
    return y * (target_amplitude / peak)


def _segment_audio(
    y: np.ndarray,
    sr: int,
    segment_duration: float = SEGMENT_DURATION,
    overlap: float = SEGMENT_OVERLAP,
) -> list[np.ndarray]:
    """Split a long audio clip into overlapping segments.

    Parameters
    ----------
    y:
        Audio time-series.
    sr:
        Sample rate.
    segment_duration:
        Length of each segment in seconds.
    overlap:
        Overlap between consecutive segments in seconds.

    Returns
    -------
    list[np.ndarray]
        List of audio segments. Returns ``[y]`` for short clips.
    """
    total_samples = len(y)
    seg_samples = int(segment_duration * sr)
    step_samples = int((segment_duration - overlap) * sr)

    if total_samples <= seg_samples:
        return [y]

    segments: list[np.ndarray] = []
    start = 0
    while start < total_samples:
        end = min(start + seg_samples, total_samples)
        segments.append(y[start:end])
        start += step_samples

    return segments


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_features(y: np.ndarray, sr: int) -> dict[str, float]:
    """Extract a 90-dimensional feature vector from an audio segment.

    Features extracted
    ------------------
    - MFCC (13 coefficients) → mean & std = 26 values
    - Delta MFCC             → mean & std = 26 values
    - Chroma STFT (12)       → mean & std = 24 values
    - Spectral centroid       → mean & std = 2 values
    - Spectral rolloff        → mean & std = 2 values
    - Spectral bandwidth      → mean & std = 2 values
    - Zero-crossing rate      → mean & std = 2 values
    - RMS energy              → mean & std = 2 values
    - Pitch (YIN F0)          → mean & std = 2 values
    - Duration                → 1 value
    - Tempo                   → 1 value

    Total: 90 dimensions.

    Parameters
    ----------
    y:
        Audio time-series (mono).
    sr:
        Sample rate.

    Returns
    -------
    dict[str, float]
        Feature name → scalar value mapping.
    """
    features: dict[str, float] = {}

    # --- MFCC ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT)
    for i in range(N_MFCC):
        features[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        features[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))

    # --- Delta MFCC ---
    delta_mfcc = librosa.feature.delta(mfcc)
    for i in range(N_MFCC):
        features[f"delta_mfcc_{i+1}_mean"] = float(np.mean(delta_mfcc[i]))
        features[f"delta_mfcc_{i+1}_std"] = float(np.std(delta_mfcc[i]))

    # --- Chroma ---
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT)
    for i in range(12):
        features[f"chroma_{i+1}_mean"] = float(np.mean(chroma[i]))
        features[f"chroma_{i+1}_std"] = float(np.std(chroma[i]))

    # --- Spectral centroid ---
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT)
    features["spectral_centroid_mean"] = float(np.mean(centroid))
    features["spectral_centroid_std"] = float(np.std(centroid))

    # --- Spectral rolloff ---
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT)
    features["spectral_rolloff_mean"] = float(np.mean(rolloff))
    features["spectral_rolloff_std"] = float(np.std(rolloff))

    # --- Spectral bandwidth ---
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT)
    features["spectral_bandwidth_mean"] = float(np.mean(bandwidth))
    features["spectral_bandwidth_std"] = float(np.std(bandwidth))

    # --- Zero-crossing rate ---
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=HOP_LENGTH)
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))

    # --- RMS energy ---
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))

    # --- Pitch (YIN) ---
    f0 = librosa.yin(y=y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr)
    # Replace unvoiced frames (where f0 ≈ fmin) with NaN before statistics
    f0_voiced = f0[f0 > librosa.note_to_hz("C2") * 1.01]
    features["pitch_mean"] = float(np.mean(f0_voiced)) if len(f0_voiced) > 0 else 0.0
    features["pitch_std"] = float(np.std(f0_voiced)) if len(f0_voiced) > 0 else 0.0

    # --- Duration ---
    features["duration"] = float(librosa.get_duration(y=y, sr=sr))

    # --- Tempo ---
    tempo_arr, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=HOP_LENGTH)
    features["tempo"] = float(np.asarray(tempo_arr).flat[0]) if np.asarray(tempo_arr).size > 0 else 0.0

    return features


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------


class AudioFeatureExtractor:
    """Extract and persist audio features for an entire directory of files.

    Parameters
    ----------
    output_path:
        Path where the resulting ``feature_dataset.csv`` will be saved.
        Defaults to ``FEATURES_DIR / "feature_dataset.csv"``.
    """

    def __init__(self, output_path: Optional[Path] = None) -> None:
        self.output_path = output_path or (FEATURES_DIR / "feature_dataset.csv")
        self._records: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_directory(
        self,
        data_dir: Path,
        subset: Optional[int] = None,
        resume: bool = True,
    ) -> pd.DataFrame:
        """Process all ``.wav`` files in *data_dir* and return a feature DataFrame.

        Parameters
        ----------
        data_dir:
            Root directory containing RAVDESS ``.wav`` files
            (may include subdirectories).
        subset:
            If provided, only process the first *subset* files found.
        resume:
            If ``True`` and the output CSV already exists, skip files
            that have already been processed.

        Returns
        -------
        pd.DataFrame
            Feature matrix with one row per (file, segment).
        """
        audio_files = sorted(data_dir.rglob("*.wav"))
        if not audio_files:
            raise FileNotFoundError(f"No .wav files found under {data_dir}")

        if subset is not None:
            audio_files = audio_files[:subset]

        logger.info("Found %d audio files to process.", len(audio_files))

        already_processed: set[str] = set()
        if resume and self.output_path.exists():
            existing = pd.read_csv(self.output_path)
            already_processed = set(existing["file_path"].tolist())
            self._records = existing.to_dict("records")
            logger.info(
                "Resuming: %d files already in CSV.", len(already_processed)
            )

        skipped, failed = 0, 0
        for path in tqdm(audio_files, desc="Extracting features", unit="file"):
            if str(path) in already_processed:
                skipped += 1
                continue
            try:
                new_records = self._process_file(path)
                self._records.extend(new_records)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping %s: %s", path.name, exc)
                failed += 1

        logger.info(
            "Processing complete. Skipped=%d, Failed=%d, Processed=%d",
            skipped,
            failed,
            len(audio_files) - skipped - failed,
        )

        df = pd.DataFrame(self._records)
        self._save(df)
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_file(self, path: Path) -> list[dict]:
        """Load, normalise, segment (if needed), and extract features.

        Parameters
        ----------
        path:
            Path to a single RAVDESS ``.wav`` file.

        Returns
        -------
        list[dict]
            One record per segment (usually just one).
        """
        emotion_label, narrative_tone = _parse_ravdess_filename(path)

        y, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        if duration < MIN_DURATION:
            raise ValueError(f"Duration {duration:.2f}s below minimum {MIN_DURATION}s")

        y = _normalize_audio(y)

        segments = (
            _segment_audio(y, sr) if duration > MAX_DURATION else [y]
        )

        records = []
        for seg_idx, segment in enumerate(segments):
            feats = extract_features(segment, sr)
            record = {
                "file_path": str(path),
                "segment_idx": seg_idx,
                "emotion_label": emotion_label,
                "narrative_tone": narrative_tone,
                **feats,
            }
            records.append(record)

        return records

    def _save(self, df: pd.DataFrame) -> None:
        """Persist the feature DataFrame to CSV.

        Parameters
        ----------
        df:
            DataFrame to save.
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.output_path, index=False)
        logger.info("Feature dataset saved to %s (%d rows).", self.output_path, len(df))
