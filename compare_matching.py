import time
import numpy as np
import librosa
import soundfile as sf
from src.tasks.retrieval import (
    add_song_sample, process_retrieval,
    add_song_sample_window, process_retrieval_window,
    add_song_sample_fingerprint, process_retrieval_fingerprint
)

# 1. Create a 30-second fake "song" (sine waves + noise so it has features)
sr = 16000
duration = 30
t = np.linspace(0, duration, int(sr * duration), endpoint=False)
song_audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
# Add some structure so fingerprinting works (peaks in spectrogram)
song_audio[int(sr*5):int(sr*6)] += 0.8 * np.sin(2 * np.pi * 880 * t[int(sr*5):int(sr*6)])
song_audio[int(sr*15):int(sr*17)] += 0.8 * np.sin(2 * np.pi * 1200 * t[int(sr*15):int(sr*17)])

sf.write("test_full_song.wav", song_audio, sr)

print("--- INDEXING ---")
# Index using global feature (Old approach)
add_song_sample("My_Awesome_Song", "test_full_song.wav")

# Index using Sliding Window
add_song_sample_window("My_Awesome_Song", "test_full_song.wav")

# Index using Fingerprint
add_song_sample_fingerprint("My_Awesome_Song", "test_full_song.wav")

print("Indexed successfully.\n")

# 2. Extract a 5-second crop (Partial match) from the middle (seconds 14 to 19)
crop_audio = song_audio[14*sr : 19*sr]

print("--- PARTIAL MATCHING QUERY (5-second crop) ---")

# A. Global Baseline
start_time = time.time()
res_global = process_retrieval(audio_signal=crop_audio, match_type="song")
time_global = time.time() - start_time
best_global = res_global['best_match']['label'] if res_global['best_match'] else "None"
print(f"[1. Global Mean KD-Tree] Match: {best_global} | Time: {time_global:.3f}s")

# B. Sliding Window KD-Tree
start_time = time.time()
res_win = process_retrieval_window(audio_signal=crop_audio, sample_rate=sr)
time_win = time.time() - start_time
print(f"[2. Sliding Window KD-Tree] Match: {res_win['best_match']} | Score: {res_win['score']:.2f} | Time: {time_win:.3f}s")

# C. Constellation Fingerprint
start_time = time.time()
res_fp = process_retrieval_fingerprint(audio_signal=crop_audio, sample_rate=sr)
time_fp = time.time() - start_time
print(f"[3. Constellation Fingerprint] Match: {res_fp['best_match']} | Aligned Peaks: {res_fp['score']} | Time: {time_fp:.3f}s")

print("\nComparison Complete!")
