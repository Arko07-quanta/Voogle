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

2. **Language Dataset: [VoxForge Multilingual Speech Corpus](http://www.voxforge.org/)**
   - **Source:** VoxForge open-source multilingual speech collection (`language.zip`).
   - **Languages covered:** `German (Deutsch)`, `English`, `Spanish (Español)` — named by their two-letter ISO code prefix in filenames (`de_`, `en_`, `es_`).
   - **Structure inside zip:** `train/` and `test/` splits containing `.flac` fragments named `<lang>_<gender>_<hash>.fragment<n>.flac`.
   - **Usage:** Shifted Delta Cepstral (SDC) tensors ($N=7, d=1, P=3, k=7$, 49 dimensions) extracted and indexed as a spatial KD-Tree (`indexes/languages_*`). 60 representative samples per language are indexed by default.

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
├── emotions/           # Raw emotion audio files (e.g. RAVDESS Actor_01/ ... Actor_24/)
│   └── Actor_01/
│       └── 03-01-01-...wav
├── languages/          # Language speech recordings, organized by language code subfolder
│   ├── de/             # German samples
│   ├── en/             # English samples
│   └── es/             # Spanish samples
├── songs/              # Track samples and music recordings
└── voices/             # Multi-sample speaker voice recordings
```

> **Note:** `datasets/` is in `.gitignore`. All audio files, zip archives, and downloaded datasets stay local only.

---

### Emotion Dataset Setup (RAVDESS)

1. **Download** from Kaggle: [RAVDESS Emotional Speech Audio](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) and save as `emotions.zip`.
2. **Extract:**
   ```bash
   unzip emotions.zip -d datasets/emotions/
   ```
3. **Build the KD-Tree model:**
   ```bash
   python build_emotion_tree.py
   ```
   Scans all `Actor_*/` folders, parses the RAVDESS filename convention to identify emotion labels, extracts 32-D classical acoustic signatures (TKEO, HNR, MFCC moments, spectral flux), and writes `indexes/emotions_meta.json` and `indexes/emotions_vectors.npy`.

---

### Language Dataset Setup (VoxForge)

1. **Download** VoxForge multilingual data and save as `language.zip`.
2. **Extract** representative samples per language (fast — does not extract all 73k files):
   ```bash
   python3 -c "
   import zipfile, os
   langs = ['de', 'en', 'es']
   for l in langs:
       os.makedirs(f'datasets/languages/{l}', exist_ok=True)
   counts = {l: 0 for l in langs}
   MAX = 150
   with zipfile.ZipFile('language.zip') as z:
       for name in z.namelist():
           base = os.path.basename(name)
           if not base or '_' not in base:
               continue
           prefix = base.split('_')[0]
           if prefix in counts and counts[prefix] < MAX:
               target = f'datasets/languages/{prefix}/{base}'
               if not os.path.exists(target):
                   with z.open(name) as src, open(target, 'wb') as dst:
                       dst.write(src.read())
               counts[prefix] += 1
           if all(c >= MAX for c in counts.values()):
               break
   print('Done:', counts)
   "
   ```
3. **Build the KD-Tree model:**
   ```bash
   python build_language_tree.py
   ```
   Processes `.flac`/`.wav` files in each language subfolder, extracts 49-D Shifted Delta Cepstral (SDC) feature vectors, and writes `indexes/languages_meta.json` and `indexes/languages_vectors.npy`.

---

### Adding Audio via the GUI (Any Dataset Type)

For adding individual audio clips to any tree at runtime:

1. Launch the web UI:
   ```bash
   python gui.py
   ```
2. Navigate to the **"🌲 Record & Add to Tree Datasets"** tab.
3. Record from your microphone or upload a file, enter a label, pick a target (**Voice**, **Song**, **Language**, or **Emotion**), and click **"Save & Index into KD-Tree"**.
   - Files are archived into `datasets/<target>/` and immediately indexed.
   - **Multiple recordings per label are supported** — the tree aggregates them and ranks by best/average cosine similarity.

### Programmatic API

```python
from src.tasks.retrieval import add_voice_sample, add_song_sample
from src.tasks.language import add_language_sample
from src.tasks.emotion import add_emotion_sample

# Multiple voice recordings for the same person
add_voice_sample(label="Alice", audio_file_path="alice_calm.wav")
add_voice_sample(label="Alice", audio_file_path="alice_loud.wav")

# Songs, languages, and emotions
add_song_sample(label="Beethoven - Symphony 5", audio_file_path="symphony.wav")
add_language_sample(label="French", audio_file_path="bonjour.wav")
add_emotion_sample(label="happy", audio_file_path="laughing.wav")
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