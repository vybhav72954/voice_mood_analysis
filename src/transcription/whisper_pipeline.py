"""Task 3 — Automatic Speech Transcription with OpenAI Whisper.

Transcribes a directory of audio files and measures Word Error Rate (WER)
on the RAVDESS standard sentences where ground truth is available.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from tqdm import tqdm

from src.config import (
    RAVDESS_STATEMENTS,
    TRANSCRIPTS_DIR,
    TRANSCRIPTION_SUBSET,
    WHISPER_MODEL,
)

logger = logging.getLogger(__name__)


def _get_ravdess_ground_truth(path: Path) -> Optional[str]:
    """Return the ground-truth transcript for a RAVDESS file, or None.

    RAVDESS actors read one of two fixed statements identified by the 5th
    dash-separated field of the filename:
      - ``01`` → "Kids are talking by the door"
      - ``02`` → "Dogs are sitting by the door"

    Parameters
    ----------
    path:
        Path to a RAVDESS audio file.

    Returns
    -------
    Optional[str]
        Ground-truth sentence, or ``None`` if not parseable.
    """
    parts = path.stem.split("-")
    if len(parts) != 7:
        return None
    statement_code = parts[4]
    return RAVDESS_STATEMENTS.get(statement_code)


class WhisperTranscriber:
    """Transcribe audio files using OpenAI Whisper and measure WER.

    Parameters
    ----------
    model_name:
        Whisper model size: ``"base"``, ``"small"``, ``"medium"``, ``"large"``.
    output_dir:
        Directory where individual transcript ``.txt`` files and the
        summary CSV are saved.
    """

    def __init__(
        self,
        model_name: str = WHISPER_MODEL,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.model_name = model_name
        self.output_dir = output_dir or TRANSCRIPTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._model = None  # lazy load

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe_directory(
        self,
        audio_dir: Path,
        subset: int = TRANSCRIPTION_SUBSET,
    ) -> pd.DataFrame:
        """Transcribe up to *subset* audio files in *audio_dir*.

        Parameters
        ----------
        audio_dir:
            Root directory containing ``.wav`` files.
        subset:
            Maximum number of files to transcribe.

        Returns
        -------
        pd.DataFrame
            Summary DataFrame with one row per file:
            ``file_path``, ``transcript``, ``ground_truth``,
            ``duration_s``, ``processing_time_s``.
        """
        self._load_model()

        audio_files = sorted(audio_dir.rglob("*.wav"))[:subset]
        if not audio_files:
            raise FileNotFoundError(f"No .wav files found under {audio_dir}")

        logger.info(
            "Transcribing %d files with Whisper '%s'.",
            len(audio_files), self.model_name,
        )

        records: list[dict] = []
        for path in tqdm(audio_files, desc="Transcribing", unit="file"):
            record = self._transcribe_file(path)
            records.append(record)

        df = pd.DataFrame(records)

        # --- Compute WER on files with ground truth ---
        wer_df = df[df["ground_truth"].notna()].copy()
        if not wer_df.empty:
            avg_wer = self._compute_wer(
                list(wer_df["ground_truth"]),
                list(wer_df["transcript"]),
            )
            logger.info(
                "WER on %d RAVDESS files with ground truth: %.4f",
                len(wer_df), avg_wer,
            )
            df["wer"] = None
            for idx in wer_df.index:
                df.at[idx, "wer"] = self._compute_wer(
                    [df.at[idx, "ground_truth"]],
                    [df.at[idx, "transcript"]],
                )
        else:
            logger.warning("No RAVDESS ground-truth sentences found for WER calculation.")
            df["wer"] = None

        summary_path = self.output_dir / "transcription_summary.csv"
        df.to_csv(summary_path, index=False)
        logger.info("Transcription summary saved to %s", summary_path)
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Lazily load the Whisper model onto the best available device."""
        if self._model is not None:
            return
        try:
            import whisper  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "openai-whisper is not installed. "
                "Run: pip install openai-whisper"
            ) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "Loading Whisper model '%s' on %s ...", self.model_name, device
        )
        self._model = whisper.load_model(self.model_name, device=device)
        logger.info("Whisper model loaded.")

    def _transcribe_file(self, path: Path) -> dict:
        """Transcribe a single audio file and return a result record.

        Parameters
        ----------
        path:
            Path to a ``.wav`` file.

        Returns
        -------
        dict
            Record containing file metadata and the transcript.
        """
        import whisper  # type: ignore[import]

        ground_truth = _get_ravdess_ground_truth(path)

        start = time.perf_counter()
        try:
            result = self._model.transcribe(
                str(path),
                language="en",
                fp16=torch.cuda.is_available(),
            )
            transcript = result["text"].strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Transcription failed for %s: %s", path.name, exc)
            transcript = ""

        elapsed = time.perf_counter() - start

        # Save individual transcript
        txt_path = self.output_dir / f"{path.stem}.txt"
        txt_path.write_text(transcript, encoding="utf-8")

        # Estimate audio duration
        try:
            import librosa
            duration = librosa.get_duration(path=str(path))
        except Exception:  # noqa: BLE001
            duration = float("nan")

        return {
            "file_path": str(path),
            "transcript": transcript,
            "ground_truth": ground_truth,
            "duration_s": round(duration, 3),
            "processing_time_s": round(elapsed, 3),
        }

    @staticmethod
    def _compute_wer(references: list[str], hypotheses: list[str]) -> float:
        """Compute Word Error Rate using the ``jiwer`` library.

        Parameters
        ----------
        references:
            List of ground-truth transcript strings.
        hypotheses:
            List of predicted transcript strings.

        Returns
        -------
        float
            WER value in [0, 1].
        """
        try:
            import jiwer  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "jiwer is not installed. Run: pip install jiwer"
            ) from exc

        transform = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfWords(),
        ])
        return jiwer.wer(
            references,
            hypotheses,
            reference_transform=transform,
            hypothesis_transform=transform,
        )
