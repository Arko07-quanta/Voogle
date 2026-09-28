import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.ndimage import maximum_filter
import os

os.makedirs('images', exist_ok=True)
plt.style.use('dark_background')

# 1. Basic Spectrogram
def gen_spectrogram():
    y, sr = librosa.load(librosa.ex('trumpet'), duration=3)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, cmap='magma')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.savefig('images/img_dsp_spectrogram.png', dpi=300)
    plt.close()

# 2. TKEO
def gen_tkeo():
    y, sr = librosa.load(librosa.ex('trumpet'), duration=0.1)
    tkeo = y[1:-1]**2 - y[:-2] * y[2:]
    plt.figure(figsize=(8, 4))
    plt.plot(y, alpha=0.5, label='Waveform')
    plt.plot(np.arange(1, len(tkeo)+1), tkeo*10, color='red', label='TKEO Energy')
    plt.title('Teager-Kaiser Energy Operator (TKEO)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('images/img_emotion_tkeo.png', dpi=300)
    plt.close()

# 3. Constellation
def gen_constellation():
    y, sr = librosa.load(librosa.ex('brahms'), duration=3)
    stft = librosa.stft(y, n_fft=2048, hop_length=512)
    magnitude = np.abs(stft)
    local_max = maximum_filter(magnitude, size=15) == magnitude
    threshold = np.mean(magnitude) * 3
    peaks_mask = local_max & (magnitude > threshold)
    
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(librosa.amplitude_to_db(magnitude, ref=np.max), y_axis='log', x_axis='time', cmap='magma', alpha=0.6)
    
    times = librosa.frames_to_time(np.arange(magnitude.shape[1]), sr=sr, hop_length=512)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    
    f_idx, t_idx = np.where(peaks_mask)
    plt.scatter(times[t_idx], freqs[f_idx], color='cyan', s=10, marker='x', label='Constellation Peaks')
    plt.title('Shazam 2D Spectral Peak Detection')
    plt.ylim(0, 8000)
    plt.tight_layout()
    plt.savefig('images/img_music_constellation.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    print("Generating LaTeX images...")
    gen_spectrogram()
    gen_tkeo()
    gen_constellation()
    print("Images saved in images/ directory!")
