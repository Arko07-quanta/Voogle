# Voogle

The ultimate voice search engine
Team Name: yes, this_is_fft

## Description

Voogle is an intelligent, multi-purpose audio analysis and recognition application designed to process and extract meaningful information from complex audio clips. The platform serves as an end-to-end processing pipeline utilizing state-of-the-art classical signal processing and classical spatial tree search structures (KD-Trees). By extracting and analyzing acoustic features, Voogle can perform a wide array of audio-centric tasks purely through algorithmic signal manipulation and metric space nearest-neighbor queries without any heavy neural network models.

## Key Features

*   **Tree-Based Song, Voice & Language Matching:** Constructs local, persistent metric spatial KD-Trees over acoustic descriptors (MFCC statistics, spectral centroid moments, chroma/tonnetz features, and Shifted Delta Cepstral vectors). Allows adding new sample audio to expand datasets dynamically.
*   **Speaker Detection & Diarization:** Identifies and clusters multiple speakers using Voice Activity Detection (VAD), Normalized Graph Laplacians ($L_{\text{sym}} = I - D^{-1/2}AD^{-1/2}$), and Spectral Bisection along the Fiedler vector.
*   **Emotion Identification:** Analyzes vocal stress, tone, and quality using the non-linear Teager-Kaiser Energy Operator (TKEO), Harmonic-to-Noise Ratio (HNR) via autocorrelation, and micro-tremor pitch jitter.
*   **Audio Matching & Retrieval:** Queries unknown audio against your local KD-Tree index using metric cosine distances.
*   **Speech-to-Text Transcription:** Rigorous Linear Predictive Coding (LPC) via Levinson-Durbin recursion and Dynamic Time Warping (DTW) distance minimization.
*   **Music & Rhythm Search:** Identifies tempo and rhythmic beat periodicity via Librosa beat-tracking.

## Getting Started

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the Web Interface:
   ```bash
   python gui.py
   ```