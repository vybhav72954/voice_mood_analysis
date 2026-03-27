# Narrative Voice Processing and Classification

A machine learning pipeline for classifying and retrieving narrative voice recordings in interactive storytelling environments. Built for tabletop RPG systems where pre-recorded narration (suspense, calm description, dramatic dialogue, urgency) is triggered dynamically during gameplay.

The system extracts 90-dimensional acoustic feature vectors from speech recordings, trains multiple classifiers to predict narrative tone, provides filter-based and cosine-similarity retrieval, and includes automatic speech transcription via OpenAI Whisper.

The valence-arousal circumplex projection in the analysis module applies acoustic proxy mappings derived from Chaturvedi et al. (2022), where RMS energy, spectral centroid, and zero-crossing rate serve as arousal proxies, while pitch variation, chroma, and spectral bandwidth approximate valence. This framework, originally developed for music mood and human emotion mapping based on physiological signals, generalises naturally to narrative speech classification.

**References:**
- Chaturvedi, V., Garg, A., Kaur, A.B., Varshney, V., & Parashar, A. (2022). Machine learning model for mapping of music mood and human emotion based on physiological signals. *Multimedia Tools and Applications*, 81. https://doi.org/10.1007/s11042-021-11650-0
- Chaturvedi, V., Kaur, A.B., Varshney, V. et al. (2022). Music mood and human emotion recognition based on physiological signals: a systematic review. *Multimedia Systems*. https://doi.org/10.1007/s00530-021-00786-6

Refer to the GitHub - https://github.com/vybhav72954/Music-Mood-Analysis
---

## Dataset

This project uses **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) as a proxy corpus for narrative recordings.

**Download:** https://zenodo.org/record/1188976

Place extracted `.wav` files under `data/raw/`. The pipeline searches recursively, so nested directory structures (e.g. `Actor_01/`, `Actor_02/`) work without modification.

RAVDESS emotions are mapped to narrative tones as follows:

| RAVDESS Emotion | Narrative Tone |
|---|---|
| neutral, calm | calm_description |
| happy, surprised | character_dialogue |
| sad, disgust | dramatic_emphasis |
| fearful | suspense |
| angry | urgency |

---

## Installation

### Prerequisites

- Python 3.10+
- `ffmpeg` on PATH (required for Whisper transcription)
  - macOS: `brew install ffmpeg`
  - Ubuntu: `sudo apt install ffmpeg`
  - Windows: https://ffmpeg.org/download.html

### Setup

```bash
git clone https://github.com/vybhav72954/tabletalk.git
cd tabletalk

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

If Whisper transcription is not needed:

```bash
pip install -r requirements.txt
pip install -e .
# Run with --skip_transcription flag
```

---

## Usage

```bash
# Full pipeline (all 1440 RAVDESS files)
python run_pipeline.py --data_dir data/raw --subset 1440

# Quick run (200 files, no transcription)
python run_pipeline.py --data_dir data/raw --subset 200 --skip_transcription
```

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `data/raw` | Path to raw `.wav` files |
| `--subset` | `200` | Number of files to process |
| `--transcription_subset` | `20` | Files to transcribe |
| `--skip_transcription` | `False` | Skip Whisper transcription |
| `--skip_storytelling` | `False` | Skip storytelling analysis |

The pipeline runs in order: feature extraction, classification (with extended analysis including SHAP, learning curves, and valence-arousal projection), transcription, retrieval demo, and storytelling analysis. All outputs are written to `outputs/`.

---

## Project Structure

```
tabletalk/
├── run_pipeline.py              # Single entrypoint
├── requirements.txt
├── pyproject.toml               # pytest config + editable install
├── conftest.py
│
├── src/
│   ├── config.py                # Constants, paths, hyperparameters
│   ├── preprocessing/
│   │   └── audio_pipeline.py    # 90-dim feature extraction
│   ├── classification/
│   │   └── tone_classifier.py   # SVM, RF, MLP, LR, ensemble, XGBoost
│   ├── transcription/
│   │   └── whisper_pipeline.py  # Whisper ASR + WER evaluation
│   ├── retrieval/
│   │   └── audio_retrieval.py   # Filter + cosine similarity search
│   └── analysis/
│       ├── model_analysis.py    # SHAP, feature importance, learning curves
│       ├── embedding_viz.py     # t-SNE, UMAP, valence-arousal circumplex
│       └── storytelling_analysis.py
│
├── tests/                       # Synthetic data tests (no RAVDESS needed)
├── data/
│   ├── raw/                     # RAVDESS .wav files (user-provided)
│   └── transcripts/             # Whisper output
├── outputs/                     # All generated artefacts
│   ├── features/                # feature_dataset.csv
│   ├── models/                  # Saved classifiers
│   └── results/                 # Plots, reports, SHAP values
└── report/
    └── technical_report.md
```

---

## Tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=src --cov-report=term-missing
```

Tests use synthetically generated audio and feature data. No RAVDESS download required.

---

## Technical Report

See [`report/technical_report.md`](report/technical_report.md) for methodology, results, and discussion.
