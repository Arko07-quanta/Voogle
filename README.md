# Voogle

The ultimate voice search engine
Team Name: yes, this_is_fft

## Description

Voogle is an intelligent, multi-purpose audio analysis and recognition application designed to process and extract meaningful information from complex audio clips. The platform serves as an end-to-end processing pipeline utilizing state-of-the-art classical signal processing. By extracting and analyzing acoustic features, Voogle can perform a wide array of audio-centric tasks purely through algorithmic signal manipulation.

## Key Features

*   **Speaker Detection & Diarization:** Identifies and clusters multiple speakers using Voice Activity Detection and MFCC clustering.
*   **Emotion Identification:** Analyzes vocal tones, pitch variations, and speech patterns using classical signal analysis heuristics.
*   **Audio Matching & Retrieval:** Scans and matches specific target audio snippets using cosine similarity on spectral features.
*   **Speech-to-Text Transcription:** Provides heuristic phoneme-level pseudo-transcription based on onset detection and formant frequency mapping.
*   **Language & Accent Identification:** Performs heuristic language estimation analyzing rhythm, tempo, and pitch variability.
*   **Music & Rhythm Search:** Identifies musical properties and beat sequences via Librosa.

## System Architecture & Methodology

*   **Audio Preprocessing:** Window clipping and sliding windows.
*   **Feature Extraction:** Mel-frequency filter banks and MFCCs.
*   **Signal Processing Algorithms:** Spectral centroid analysis, onset detection, and statistical clustering.

## Getting Started

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the Web Interface:
   ```bash
   python gui.py
   ```