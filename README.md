# Voogle

The ultimate voice search engine
Team Name: yes, this_is_fft

## Description

Voogle is an intelligent, multi-purpose audio analysis and recognition application designed to process and extract meaningful information from complex audio clips. The platform serves as an end-to-end processing pipeline that seamlessly transitions from raw signal manipulation to advanced deep learning predictions. By utilizing general-purpose audio embeddings alongside specialized neural network architectures, Voogle can perform a wide array of audio-centric tasks.

## Key Features

*   **Speaker Detection & Diarization:** Automatically identifies and separates multiple speakers using SOTA Pyannote models.
*   **Emotion Identification:** Analyzes vocal tones, pitch variations, and speech patterns using Wav2Vec2.
*   **Audio Matching & Retrieval:** Scans and matches specific target audio snippets.
*   **Speech-to-Text Transcription:** Converts spoken language into text using OpenAI Whisper.
*   **Language & Accent Identification:** Detects spoken language and classifies accents using MMS.
*   **Music & Rhythm Search:** Identifies songs or musical patterns via Librosa.

## System Architecture & Methodology

*   **Audio Preprocessing:** Window clipping and sliding windows.
*   **Feature Extraction:** Mel-frequency filter banks.
*   **Neural Network Integration:** Deep neural networks for audio embeddings (AST Model).

## Getting Started

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Accept the terms of use for Pyannote models on HuggingFace and set your `HF_TOKEN`.
3. Launch the Web Interface:
   ```bash
   python gui.py
   ```