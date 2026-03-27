"""TableTalk GSoC 2026 — End-to-End Pipeline Entrypoint.

Runs all four tasks (+ bonus) in sequence:
  1. Audio feature extraction
  2. Narrative tone classification
  3. Whisper transcription
  4. Retrieval system demo
  (5. Storytelling analysis — bonus)

Usage
-----
    python run_pipeline.py --data_dir data/raw --subset 200
    python run_pipeline.py --data_dir data/raw --subset 50 --skip_transcription
    python run_pipeline.py --help
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging setup (file + console)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("tabletalk.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("run_pipeline")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _banner(title: str) -> None:
    width = 60
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}\n")


def _elapsed(start: float) -> str:
    seconds = time.perf_counter() - start
    mins, secs = divmod(int(seconds), 60)
    return f"{mins}m {secs}s"


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def stage_features(data_dir: Path, subset: int) -> Path:
    """Run Task 1: feature extraction.

    Parameters
    ----------
    data_dir:
        Directory containing raw RAVDESS ``.wav`` files.
    subset:
        Maximum number of files to process.

    Returns
    -------
    Path
        Path to the saved ``feature_dataset.csv``.
    """
    from src.preprocessing.audio_pipeline import AudioFeatureExtractor

    _banner("TASK 1 — Audio Feature Extraction")
    t0 = time.perf_counter()
    extractor = AudioFeatureExtractor()
    df = extractor.process_directory(data_dir, subset=subset)
    print(f"  ✓ {len(df)} samples extracted → {extractor.output_path}")
    print(f"  ✓ Feature dimensions: {df.shape[1]} columns")
    print(f"  ✓ Tone distribution:\n{df['narrative_tone'].value_counts().to_string()}")
    print(f"\n  Elapsed: {_elapsed(t0)}")
    return extractor.output_path


def stage_classification(features_csv: Path) -> tuple[dict, "ToneClassifier"]:
    """Run Task 2: tone classification.

    Parameters
    ----------
    features_csv:
        Path to the feature dataset produced by Task 1.

    Returns
    -------
    tuple[dict, ToneClassifier]
        Summary metrics and the trained classifier instance.
    """
    from src.classification.tone_classifier import ToneClassifier

    _banner("TASK 2 -- Narrative Tone Classification")
    t0 = time.perf_counter()
    classifier = ToneClassifier(features_csv=features_csv)
    results = classifier.run()
    print(f"  ✓ Best model      : {results['best_model']}")
    print(f"  ✓ Test accuracy   : {results['test_accuracy']:.4f}")
    print(f"  ✓ Test F1-macro   : {results['test_f1_macro']:.4f}")
    print(f"  ✓ Test ROC-AUC    : {results['test_roc_auc']:.4f}")
    print(f"\n  Elapsed: {_elapsed(t0)}")
    return results, classifier


def stage_transcription(data_dir: Path, transcription_subset: int) -> None:
    """Run Task 3: Whisper transcription.

    Parameters
    ----------
    data_dir:
        Directory containing raw audio files.
    transcription_subset:
        Number of files to transcribe.
    """
    from src.transcription.whisper_pipeline import WhisperTranscriber

    _banner("TASK 3 — Automatic Speech Transcription (Whisper)")
    t0 = time.perf_counter()
    transcriber = WhisperTranscriber()
    df = transcriber.transcribe_directory(data_dir, subset=transcription_subset)

    wer_df = df[df["wer"].notna()]
    if not wer_df.empty:
        avg_wer = wer_df["wer"].mean()
        print(f"  ✓ Average WER on {len(wer_df)} files: {avg_wer:.4f}")
    else:
        print("  ✓ Transcription complete (no WER ground truth available)")

    print(f"  ✓ {len(df)} files transcribed → data/transcripts/")
    print(f"  Sample transcript: '{df.iloc[0]['transcript'][:80]}...'")
    print(f"\n  Elapsed: {_elapsed(t0)}")


def stage_retrieval(features_csv: Path) -> None:
    """Run Task 4: retrieval demo queries.

    Parameters
    ----------
    features_csv:
        Path to the feature dataset.
    """
    from src.retrieval.audio_retrieval import NarrativeAudioRetrieval, run_demo_queries

    _banner("TASK 4 — Narrative Audio Retrieval System")
    t0 = time.perf_counter()
    run_demo_queries(features_csv=features_csv)
    print(f"  Elapsed: {_elapsed(t0)}")



def stage_storytelling(features_csv: Path) -> None:
    """Run Bonus: storytelling vs conversational analysis.

    Parameters
    ----------
    features_csv:
        Path to the feature dataset.
    """
    from src.analysis.storytelling_analysis import StorytellingAnalyser

    _banner("BONUS — Storytelling vs Conversational Analysis")
    t0 = time.perf_counter()
    analyser = StorytellingAnalyser(features_csv=features_csv)
    stats_df = analyser.run()
    sig = stats_df[stats_df["significant"]]
    print(f"  ✓ {len(sig)}/{len(stats_df)} features significantly different (α=0.05)")
    if not sig.empty:
        top = sig.nsmallest(3, "p_value")[["feature", "p_value", "effect_size_r"]]
        print("  Top discriminating features:")
        for _, row in top.iterrows():
            print(
                f"    {row['feature']:<30} p={row['p_value']:.5f}  "
                f"r={row['effect_size_r']:.3f}"
            )
    print(f"\n  Elapsed: {_elapsed(t0)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed argument object.
    """
    parser = argparse.ArgumentParser(
        description="TableTalk GSoC 2026 — end-to-end audio ML pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/raw"),
        help="Path to directory containing raw RAVDESS .wav files.",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=200,
        help="Maximum number of audio files to process in Task 1.",
    )
    parser.add_argument(
        "--transcription_subset",
        type=int,
        default=20,
        help="Number of files to transcribe in Task 3.",
    )
    parser.add_argument(
        "--skip_transcription",
        action="store_true",
        help="Skip Task 3 (Whisper transcription) to save time.",
    )
    parser.add_argument(
        "--skip_storytelling",
        action="store_true",
        help="Skip the bonus storytelling analysis.",
    )
    return parser.parse_args()

def stage_model_analysis(classifier: "ToneClassifier") -> None:
    from src.analysis.model_analysis import ModelAnalyser
    from src.analysis.embedding_viz import plot_feature_embeddings

    _banner("TASK 2b -- Extended Model Analysis")
    t0 = time.perf_counter()
    analyser = ModelAnalyser(classifier)
    analyser.run()
    print("  ✓ Feature importance plot + CSV")
    print("  ✓ Learning curves")
    print("  ✓ Model comparison chart")
    print("  ✓ SHAP summary + per-class plots")

    plot_feature_embeddings(classifier)
    print("  ✓ t-SNE + UMAP embedding visualizations")
    print(f"\n  Elapsed: {_elapsed(t0)}")


def main() -> None:
    """Execute the full TableTalk pipeline."""
    args = parse_args()
    pipeline_start = time.perf_counter()

    print("\n" + "█" * 60)
    print("  TableTalk — Narrative Voice Processing Pipeline")
    print("  GSoC 2026 | HumanAI Foundation")
    print("█" * 60)
    print(f"\n  Data directory   : {args.data_dir}")
    print(f"  Subset size      : {args.subset} files")
    print(f"  Transcribe files : {args.transcription_subset}")
    print(f"  Skip transcription: {args.skip_transcription}")

    if not args.data_dir.exists():
        logger.error(
            "Data directory '%s' does not exist. "
            "Download RAVDESS from https://zenodo.org/record/1188976 "
            "and place the .wav files under this directory.",
            args.data_dir,
        )
        sys.exit(1)

    # --- Task 1 ---
    features_csv = stage_features(args.data_dir, args.subset)

    # --- Task 2 ---
    results, classifier = stage_classification(features_csv)

    # --- Task 2b: Extended Analysis ---
    try:
        stage_model_analysis(classifier)
    except Exception as exc:
        logger.warning("Extended analysis failed: %s", exc)

    # --- Task 3 ---
    if not args.skip_transcription:
        try:
            stage_transcription(args.data_dir, args.transcription_subset)
        except ImportError as exc:
            logger.warning(
                "Skipping transcription: %s. "
                "Install with: pip install openai-whisper",
                exc,
            )
    else:
        print("\n  [Skipping Task 3 — transcription]\n")

    # --- Task 4 ---
    stage_retrieval(features_csv)

    # --- Bonus ---
    if not args.skip_storytelling:
        stage_storytelling(features_csv)

    total = _elapsed(pipeline_start)
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {total}")
    print(f"  All outputs saved to: outputs/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
