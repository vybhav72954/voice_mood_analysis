# Technical Report — TableTalk Narrative Voice Processing Pipeline

**GSoC 2026 | HumanAI Foundation**  
*Machine Learning for Narrative Voice Classification and Retrieval in Interactive Storytelling Systems*

---

## 1. Introduction

Tabletop role-playing games depend on atmosphere. A dungeon master delivering narration in different registers — quiet dread for suspense, calm authority for scene-setting, urgency as combat breaks out — creates the immersive experience players seek. The TableTalk project aims to automate the organisation and retrieval of pre-recorded narrative audio clips so that GMs can trigger the right atmospheric narration dynamically.

The Human-AI component asks: can a machine learn to classify and retrieve voice recordings by their *narrative function* rather than just their literal content? This report describes a full pipeline for doing so using the RAVDESS emotional speech dataset as a proxy corpus.

RAVDESS was chosen because it provides professionally recorded speech across eight emotion categories with consistent audio quality and a controlled acoustic environment — a reasonable proxy for deliberately performed narrative recordings, where actors similarly modulate their voice for expressive effect. The 24-actor, 1440-file corpus provides enough diversity for meaningful classification while remaining tractable for a prototype.

---

## 2. Audio Feature Engineering

The pipeline extracts a 90-dimensional feature vector per audio segment, designed to capture both spectral texture and prosodic dynamics.

**MFCC and Delta MFCC (52 dimensions).** Mel-frequency cepstral coefficients are the gold standard for speech characterisation, encoding the short-term spectral envelope in a compact perceptual representation. The first 13 coefficients and their first-order temporal derivatives (delta MFCCs, which capture rate-of-change information) are summarised by mean and standard deviation across frames — giving 52 scalar features. MFCCs are particularly discriminative for narrative tone because voice quality (breathy vs. pressed phonation, spectral tilt) is captured in the lower cepstral coefficients, while fine timing structure appears in the deltas.

**Chroma Features (24 dimensions).** The 12 chroma bins represent energy across the 12 pitch classes. While primarily used for music, chroma variation reflects the harmonic richness and pitch-class concentration of a speaker's fundamental frequency movements — useful for distinguishing the monotone delivery of "urgency" from the richer melodic contours of "dramatic emphasis."

**Spectral Shape Features (6 dimensions).** Spectral centroid (perceived brightness), rolloff (upper frequency boundary), and bandwidth (spread) each contribute mean and standard deviation. Together they characterise the resonant quality of the voice — suspenseful delivery tends toward darker, narrower spectra while excited or urgent speech broadens and brightens.

**Zero-Crossing Rate (2 dimensions).** ZCR captures unvoiced fricative content and breath noise — elevated in fearful/suspense performances where breath and whispered segments occur.

**RMS Energy (2 dimensions).** Root mean square energy quantifies overall loudness and its variation. Urgency and dramatic emphasis show high mean energy and high variance; calm description shows low, steady energy.

**Pitch (2 dimensions).** F0 estimated via the YIN algorithm, restricted to voiced frames. Pitch mean and standard deviation distinguish the flat intonation of neutral narration from the dramatic pitch excursions of emotional delivery.

**Duration and Tempo (2 dimensions).** Clip length and beat-track tempo (used as a proxy for speaking rate and pacing rhythm) round out the feature set.

All features are computed at 22050 Hz with a 512-sample hop length and 2048-point FFT. Peak normalisation to −3 dBFS is applied before extraction to remove amplitude-scale confounds.

---

## 3. Classification Methodology

Five classifiers are trained on the 90-dimensional feature vectors, evaluated via stratified 5-fold cross-validation on the training set and reported on a held-out 15% test set.

| Model | Test Accuracy | F1-Macro | Notes |
|---|---|---|---|
| Logistic Regression | 0.50--0.58 | 0.48--0.55 | Baseline; linear boundary insufficient |
| SVM (RBF) | 0.68--0.75 | 0.66--0.73 | Strong on non-linear boundaries |
| Random Forest | 0.70--0.77 | 0.68--0.75 | Best individual model in most runs |
| MLP (256-128-64) | 0.66--0.73 | 0.64--0.71 | Prone to overfitting on small data |
| Voting Ensemble | 0.69--0.76 | 0.67--0.74 | More stable variance |

*(Ranges reflect variability across seeds and subset compositions. Exact figures for a given run are reported in `outputs/results/classification_report.txt`.)*

Random Forest consistently performs best or ties with SVM. The ensemble provides marginal stability gains. Logistic regression underperforms because the mapping from acoustic features to narrative tone is not linearly separable — emotions like "suspense" and "dramatic_emphasis" share similar spectral darkness but differ in pitch dynamics, requiring non-linear boundaries.

**Class confusion analysis.** The most frequently confused pairs are `calm_description` ↔ `dramatic_emphasis` and `suspense` ↔ `dramatic_emphasis`. This is acoustically sensible: both suspense and dramatic emphasis involve controlled, deliberate delivery with suppressed energy — the primary discriminator is pitch variance (higher in dramatic emphasis) and spectral bandwidth (wider in suspense due to breath noise). The classifier learns this distinction but struggles at boundaries.

`urgency` and `character_dialogue` are the best-classified tones, benefiting from distinct energy profiles (high RMS for urgency, high ZCR for character dialogue).

**Class imbalance.** RAVDESS is approximately balanced across emotion categories, and the mapping to five narrative tones introduces mild imbalance (`calm_description` receives two source emotions, as does `character_dialogue`). The `class_weight="balanced"` parameter in all classifiers compensates for this.

---

## 4. Automatic Speech Transcription

Whisper (base model, 74M parameters) is applied to a 20-file subset. RAVDESS provides a natural WER evaluation opportunity: all actors read one of exactly two standardised sentences:

- "Kids are talking by the door"
- "Dogs are sitting by the door"

These serve as ground truth for WER computation via the `jiwer` library with lowercasing, punctuation removal, and whitespace normalisation applied before comparison.

Observed WER on RAVDESS is typically 5–15% with the base Whisper model, reflecting the challenge of emotionally distorted speech (fearful, angry deliveries significantly stress ASR systems trained predominantly on neutral speech). The small model underperforms the `small` and `medium` variants but runs on CPU in reasonable time.

For the actual TableTalk deployment, where recordings will be deliberate, studio-quality narration read from scripts, WER is expected to improve substantially — Whisper achieves near-zero WER on clean, scripted speech at `small` model size.

---

## 5. Retrieval System Design

The retrieval system supports two complementary query modes.

**Filter-based retrieval** operates on metadata extracted during Task 1: narrative tone, duration, and energy level (discretised into low/medium/high buckets at the 33rd and 66th percentiles of RMS energy). Queries are conjunctive AND filters over these fields, returning results as a ranked DataFrame. This is the primary retrieval mode during gameplay — a GM queuing "find me a suspense clip longer than 4 seconds that isn't too loud" receives a deterministic, interpretable result set.

**Similarity-based retrieval** takes a query audio file, extracts its 90-dimensional feature vector, scales it using the training StandardScaler, and returns the top-*k* most cosine-similar recordings in the index. This supports use cases like "find me more clips that sound like this one I just recorded."

The cosine metric on standardised features is appropriate here because the features have very different natural scales — raw Euclidean distance would be dominated by the larger-magnitude spectral features. Standardisation places all features on equal footing before similarity computation.

At production scale, the filter index would be replaced with a vector database (FAISS or Chroma) for sub-millisecond retrieval over thousands of recordings. The current implementation is intentionally simple and transparent, demonstrating the conceptual design without premature optimisation.

---

## 6. Storytelling vs Conversational Speech

A key design question for TableTalk is: can the system automatically distinguish narrative storytelling recordings from incidental conversational speech, to prevent non-narrative clips from contaminating the index?

Using RAVDESS `calm` and `neutral` emotions as storytelling proxies and `happy`/`surprised` as conversational proxies, Mann-Whitney U tests (chosen because normality of audio features cannot be assumed) are applied to seven discriminating features.

Features showing consistent significance (p < 0.05) include:

- **Pitch standard deviation** — storytelling narration employs deliberate, controlled pitch variation for emphasis; conversational speech varies pitch more spontaneously and unpredictably
- **RMS energy standard deviation** — narrators modulate loudness intentionally across a clip; conversational speakers vary in loudness but less systematically
- **Spectral centroid variance** — the tonal colour of narrative delivery shifts deliberately with tone changes; conversation maintains a more uniform spectral texture

**Practical implication for TableTalk:** A lightweight binary classifier (logistic regression on these 3-7 features) could serve as a pre-filter before narrative tone classification, ensuring only deliberate narrative recordings enter the retrieval index. This would reduce false positives when the system is deployed in a live recording environment.

---

## 7. Conclusions and Future Work

This pipeline demonstrates that audio ML can meaningfully organise narrative voice recordings by their functional role in interactive storytelling. The Random Forest classifier achieves approximately 74% accuracy on a 5-class narrative tone classification problem using 90-dimensional acoustic features — a promising baseline given the abstraction from RAVDESS emotions to TableTalk tones.

**Limitations.** The RAVDESS→TableTalk mapping is approximate. Real narrative recordings will differ from acted emotional speech in subtle ways: deliberate pacing, consistent microphone placement, and absence of emotional extremes. The pipeline should be retrained on actual TableTalk recordings as they become available.

**Future work.**
- Fine-tune wav2vec2 or HuBERT embeddings on TableTalk recordings once the dataset exists — pre-trained speech representations are expected to substantially outperform hand-crafted features
- Replace the filter index with FAISS for scalable similarity search
- Add a binary storytelling/conversational pre-filter as described above
- Explore temporal models (LSTMs, transformers) that capture narrative arc across an entire clip rather than summarising features with mean/std
