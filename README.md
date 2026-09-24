# Voogle

The ultimate voice search engine  
**Team Name:** yes, this_is_fft

## Description

Voogle is an intelligent, multi-purpose audio analysis and recognition platform designed to process and extract meaningful information from complex audio clips. The system operates purely on **classical signal processing** and **spatial metric tree search structures (KD-Trees)**. By extracting acoustic feature representations without deep neural network dependencies, Voogle performs audio retrieval and classification through mathematical signal transformations and metric-space nearest-neighbor queries.

---

## Datasets Used by the Authors

Voogle uses benchmark datasets to construct reference KD-Tree indices:

1. **Emotion Dataset: [RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)**
   - **Source:** Kaggle / Zenodo (24 professional actors, 2,880 speech audio files).
   - **Emotions covered:** `neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`.
   - **Usage:** Classical non-linear dynamics and spectral features (TKEO, HNR, MFCC moments, spectral roll-off, zero-crossing rate, onset novelty) are extracted to form a 32-dimensional spatial KD-Tree (`indexes/emotions_*`).

2. **Language Dataset (Reference Acoustic Baseline)**
   - **Formulation:** Shifted Delta Cepstrals (SDC) tensors ($N=7, d=1, P=3, k=7$, 49 dimensions) indexed into a language spatial KD-Tree (`indexes/languages_*`).
   - **Custom Data:** Easily expandable with multi-lingual speech corpora (e.g. Common Voice, VoxForge).

3. **Voice & Speaker Dataset (Custom Multi-Recording Pool)**
   - **Formulation:** 28-dimensional acoustic descriptors (MFCC moments + spectral centroid statistics) indexed into `indexes/voices_*`.
   - **Multi-Sample Grouping:** Supports multiple recordings per person to construct aggregated acoustic profiles.

4. **Music / Song Dataset**
   - **Formulation:** 274-dimensional vectors combining log-mel spectrogram moments, chromagram pitch classes (12-D), and tonnetz harmonic vectors (6-D) indexed into `indexes/songs_*`.

---

## Setting Up Your Own Datasets

Voogle is designed to make adding your own local datasets seamless. All datasets and generated tree models are ignored by Git (`.gitignore`) so you can download large datasets locally without polluting your repository.

### Dataset Directory Structure

```text
datasets/
├── emotions/     # Raw emotion audio files (e.g. RAVDESS actor folders)
├── languages/    # Custom language speech recordings
├── songs/        # Track samples and music recordings
└── voices/       # Multi-sample speaker voice recordings
```

### Option A: Using the Graphical Interface (Easiest)

1. Launch the web UI:
   ```bash
   python gui.py
   ```
2. Navigate to the **"🌲 Record & Add to Tree Datasets"** tab.
3. Choose your audio input:
   - **Record live from your microphone** (e.g., say a sentence or sing a phrase).
   - **Upload an audio file** (`.wav`, `.mp3`, `.flac`, etc.).
4. Enter an identifier/label (e.g., person's name like `"Alice"`, a song title, or a language name).
5. Select the target dataset (**Voice**, **Song**, **Language**, or **Emotion**).
6. Click **"Save & Index into KD-Tree"**.
   - The file is archived into `datasets/<target>/` and indexed into the corresponding KD-Tree.
   - You can record **multiple audio clips under the same person or class**; the KD-Tree groups them together automatically and calculates both best and average similarities during queries.

### Option B: Batch Indexing from a Downloaded Dataset (e.g., RAVDESS)

If you have downloaded a dataset as a zip file (e.g., `emotions.zip`):

1. **Extract into the dataset folder:**
   ```bash
   unzip emotions.zip -d datasets/emotions/
   ```
2. **Build / Rebuild the KD-Tree model:**
   Run the dedicated builder script:
   ```bash
   python build_emotion_tree.py
   ```
   This will process the `.wav` files, extract 32-D classical acoustic signatures, and write the metric tree directly to `indexes/emotions_meta.json` and `indexes/emotions_vectors.npy`.

3. **Building other datasets programmatically:**
   You can index any directory of audio clips using Python:
   ```python
   from src.tasks.retrieval import add_voice_sample, add_song_sample
   from src.tasks.language import add_language_sample

   # Add speaker samples
   add_voice_sample(label="Alice", audio_file_path="path/to/alice_sample1.wav")
   add_voice_sample(label="Alice", audio_file_path="path/to/alice_sample2.wav")

   # Add songs or languages
   add_song_sample(label="Track 1", audio_file_path="path/to/song.wav")
   add_language_sample(label="French", audio_file_path="path/to/french_speech.wav")
   ```

---

## Key Features & Algorithms

* **Tree-Based Matching (Metric KD-Trees):** Nearest-neighbor queries on unit-normalized acoustic descriptors using `scipy.spatial.KDTree` (equivalent to cosine similarity ranking: $d^2 = 2 - 2\cos\theta$).
* **Speaker Detection & Diarization:** Voice Activity Detection (VAD) coupled with Spectral Graph Theory (Normalized Laplacian $L_{\text{sym}} = I - D^{-1/2}AD^{-1/2}$) and Spectral Bisection along the Fiedler vector.
* **Emotion Identification:** Non-linear Teager-Kaiser Energy Operator (TKEO), Harmonic-to-Noise Ratio (HNR) via autocorrelation, and KD-Tree template matching.
* **Speech-to-Text Transcription:** Linear Predictive Coding (LPC) via Levinson-Durbin recursion and Dynamic Time Warping (DTW) distance minimization.
* **Music & Rhythm Search:** Beat tracking and tempo estimation using onset autocorrelation.

---

## Getting Started

1. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Launch the Web Interface:**
   ```bash
   python gui.py
   ```