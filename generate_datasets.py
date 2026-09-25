import os
import numpy as np
import soundfile as sf
import shutil

sr = 16000
duration = 3.0
t = np.linspace(0, duration, int(sr * duration), endpoint=False)

def generate_language_dataset():
    base_dir = "datasets/languages"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    
    # English: Baseline speech-like
    os.makedirs(f"{base_dir}/en", exist_ok=True)
    for i in range(10):
        # 100 Hz to 200 Hz
        f0 = 120 + np.random.randn() * 10
        audio = 0.5 * np.sin(2 * np.pi * f0 * t) + 0.1 * np.random.randn(len(t))
        sf.write(f"{base_dir}/en/sample_{i}.wav", audio, sr)
        
    # Spanish: Fast rhythm (amplitude modulation)
    os.makedirs(f"{base_dir}/es", exist_ok=True)
    for i in range(10):
        f0 = 160 + np.random.randn() * 10
        am = 1 + 0.5 * np.sin(2 * np.pi * 5 * t) # 5 Hz rhythm
        audio = am * 0.5 * np.sin(2 * np.pi * f0 * t) + 0.1 * np.random.randn(len(t))
        sf.write(f"{base_dir}/es/sample_{i}.wav", audio, sr)
        
    # Mandarin: High pitch variation (frequency modulation)
    os.makedirs(f"{base_dir}/zh", exist_ok=True)
    for i in range(10):
        f0 = 200 + 50 * np.sin(2 * np.pi * 2 * t) # Pitch sweeps
        phase = np.cumsum(f0) / sr
        audio = 0.5 * np.sin(2 * np.pi * phase) + 0.1 * np.random.randn(len(t))
        sf.write(f"{base_dir}/zh/sample_{i}.wav", audio, sr)
        
    # German: Fricative heavy (high freq noise)
    os.makedirs(f"{base_dir}/de", exist_ok=True)
    for i in range(10):
        f0 = 100 + np.random.randn() * 5
        audio = 0.5 * np.sin(2 * np.pi * f0 * t)
        noise = 0.3 * np.random.randn(len(t))
        # high pass filter noise via simple difference
        noise = np.append(noise[0], noise[1:] - 0.9 * noise[:-1])
        sf.write(f"{base_dir}/de/sample_{i}.wav", audio + noise, sr)

def generate_emotion_dataset():
    base_dir = "datasets/emotions"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
        
    # Happy: High pitch, fast, bouncy (spiky TKEO)
    os.makedirs(f"{base_dir}/happy", exist_ok=True)
    for i in range(10):
        f0 = 300 + np.random.randn() * 20
        am = np.maximum(0, np.sin(2 * np.pi * 4 * t)) # staccato
        audio = am * 0.5 * np.sin(2 * np.pi * f0 * t) + 0.05 * np.random.randn(len(t))
        sf.write(f"{base_dir}/happy/sample_{i}.wav", audio, sr)
        
    # Sad: Low pitch, slow, smooth
    os.makedirs(f"{base_dir}/sad", exist_ok=True)
    for i in range(10):
        f0 = 100 + np.random.randn() * 5
        am = 0.8 + 0.2 * np.sin(2 * np.pi * 0.5 * t) # very slow modulation
        audio = am * 0.3 * np.sin(2 * np.pi * f0 * t) + 0.01 * np.random.randn(len(t))
        sf.write(f"{base_dir}/sad/sample_{i}.wav", audio, sr)
        
    # Angry: High energy, high noise (low HNR), bursts
    os.makedirs(f"{base_dir}/angry", exist_ok=True)
    for i in range(10):
        f0 = 250 + np.random.randn() * 10
        am = 1 + 0.5 * np.sign(np.sin(2 * np.pi * 2 * t)) # aggressive square wave AM
        audio = am * 0.8 * np.sin(2 * np.pi * f0 * t) + 0.4 * np.random.randn(len(t))
        sf.write(f"{base_dir}/angry/sample_{i}.wav", audio, sr)
        
    # Neutral: Baseline
    os.makedirs(f"{base_dir}/neutral", exist_ok=True)
    for i in range(10):
        f0 = 150 + np.random.randn() * 5
        audio = 0.4 * np.sin(2 * np.pi * f0 * t) + 0.05 * np.random.randn(len(t))
        sf.write(f"{base_dir}/neutral/sample_{i}.wav", audio, sr)

if __name__ == "__main__":
    print("Generating Language Dataset...")
    generate_language_dataset()
    print("Generating Emotion Dataset...")
    generate_emotion_dataset()
    print("Done!")
