"""Global configuration constants for the TableTalk pipeline."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FEATURES_DIR = OUTPUTS_DIR / "features"
MODELS_DIR = OUTPUTS_DIR / "models"
RESULTS_DIR = OUTPUTS_DIR / "results"

# ---------------------------------------------------------------------------
# Audio processing
# ---------------------------------------------------------------------------
SAMPLE_RATE: int = 22050
N_MFCC: int = 13
HOP_LENGTH: int = 512
N_FFT: int = 2048
MAX_DURATION: float = 10.0       # seconds; files longer than this are segmented
SEGMENT_DURATION: float = 5.0   # seconds per segment for long files
SEGMENT_OVERLAP: float = 1.0    # seconds of overlap between segments
MIN_DURATION: float = 0.5       # seconds; files shorter than this are skipped
PEAK_NORMALIZATION_DB: float = -3.0  # target peak level in dBFS

# ---------------------------------------------------------------------------
# RAVDESS metadata
# Filename format: 03-01-06-01-02-01-12.wav
# Positions:       Mod-VocCh-Emo-Int-Stmt-Rep-Actor
# ---------------------------------------------------------------------------
RAVDESS_EMOTIONS: dict[str, str] = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# RAVDESS standard sentences (used as ground truth for WER)
RAVDESS_STATEMENTS: dict[str, str] = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door",
}

# ---------------------------------------------------------------------------
# RAVDESS emotion → TableTalk narrative tone mapping
# ---------------------------------------------------------------------------
EMOTION_TO_TONE: dict[str, str] = {
    "neutral":   "calm_description",
    "calm":      "calm_description",
    "happy":     "character_dialogue",
    "sad":       "dramatic_emphasis",
    "angry":     "urgency",
    "fearful":   "suspense",
    "disgust":   "dramatic_emphasis",
    "surprised": "character_dialogue",
}

NARRATIVE_TONES: list[str] = [
    "calm_description",
    "character_dialogue",
    "dramatic_emphasis",
    "urgency",
    "suspense",
]

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
TEST_SIZE: float = 0.15
VAL_SIZE: float = 0.15
RANDOM_STATE: int = 42
CV_FOLDS: int = 5

# ---------------------------------------------------------------------------
# Whisper transcription
# ---------------------------------------------------------------------------
WHISPER_MODEL: str = "base"       # base | small | medium | large
TRANSCRIPTION_SUBSET: int = 20   # number of files to transcribe

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
ENERGY_LOW_PERCENTILE: int = 33
ENERGY_HIGH_PERCENTILE: int = 66
TOP_K_SIMILAR: int = 5

# ---------------------------------------------------------------------------
# Feature column name prefix conventions (used across modules)
# ---------------------------------------------------------------------------
FEATURE_COLUMNS_PREFIX: list[str] = [
    "mfcc", "delta_mfcc", "chroma",
    "spectral_centroid", "spectral_rolloff", "spectral_bandwidth",
    "zcr", "rms", "pitch",
]
