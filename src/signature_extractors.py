import numpy as np
import librosa
from src.features import extract_mel_spectrogram
from src.models import get_general_purpose_embeddings

def compute_sdc(mfccs, N=7, d=1, P=3, k=7):
    """
    Computes Shifted Delta Cepstral (SDC) coefficients.
    N: number of MFCC coefficients to use
    d: distance for delta computation (frame lag)
    P: shift between consecutive blocks
    k: number of concatenated blocks
    """
    mfccs = mfccs[:N, :]
    num_frames = mfccs.shape[1]
    
    delta = np.zeros_like(mfccs)
    for t in range(num_frames):
        left = max(0, t - d)
        right = min(num_frames - 1, t + d)
        delta[:, t] = mfccs[:, right] - mfccs[:, left]
        
    sdc_features = []
    valid_frames = num_frames - (k * P)
    
    if valid_frames <= 0:
        return np.mean(delta, axis=1)
        
    for t in range(valid_frames):
        sdc_frame = []
        for i in range(k):
            sdc_frame.append(delta[:, t + i * P])
        sdc_features.append(np.concatenate(sdc_frame))
        
    sdc_features = np.array(sdc_features)
    return np.mean(sdc_features, axis=0)

def extract_voice_feature(audio_signal, sample_rate=16000):
    """
    Extracts classical speaker/voice spectral feature vector:
    MFCC means (13) + MFCC stds (13) + Spectral Centroid mean/std (2) -> 28-dim vector.
    """
    mfcc = librosa.feature.mfcc(y=audio_signal, sr=sample_rate, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    centroid = librosa.feature.spectral_centroid(y=audio_signal, sr=sample_rate)
    cent_mean = np.array([np.mean(centroid)])
    cent_std = np.array([np.std(centroid)])
    feature = np.concatenate([mfcc_mean, mfcc_std, cent_mean, cent_std])
    return feature.astype(np.float32)

def extract_song_feature(audio_signal, sample_rate=16000):
    """
    Extracts classical music/song acoustic feature vector:
    Mel-spectrogram statistics (256) + Chroma energy mean (12) + Tonnetz (6) -> 274-dim vector.
    """
    mel_spec = extract_mel_spectrogram(audio_signal, sample_rate)
    general_emb = get_general_purpose_embeddings(mel_spec) # 256-dim
    chroma = librosa.feature.chroma_stft(y=audio_signal, sr=sample_rate)
    chroma_mean = np.mean(chroma, axis=1) # 12-dim
    tonnetz = librosa.feature.tonnetz(y=audio_signal, sr=sample_rate)
    tonnetz_mean = np.mean(tonnetz, axis=1) # 6-dim
    feature = np.concatenate([general_emb, chroma_mean, tonnetz_mean])
    return feature.astype(np.float32)

def extract_language_feature(audio_signal, sample_rate=16000):
    """
    Extracts classical Shifted Delta Cepstral (SDC) vector -> 49-dim vector.
    """
    mfccs = librosa.feature.mfcc(y=audio_signal, sr=sample_rate, n_mfcc=13)
    sdc_vector = compute_sdc(mfccs, N=7, d=1, P=3, k=7)
    if sdc_vector.shape[0] != 49:
        sdc_vector = np.pad(sdc_vector, (0, max(0, 49 - sdc_vector.shape[0])))[:49]
    return sdc_vector.astype(np.float32)

def compute_tkeo(signal):
    """
    Teager-Kaiser Energy Operator: Psi[x(n)] = x^2(n) - x(n-1)x(n+1)
    """
    if len(signal) < 3:
        return 0.0
    tkeo = signal[1:-1]**2 - signal[:-2] * signal[2:]
    return float(np.mean(tkeo))

def compute_hnr(audio_signal, sample_rate=16000):
    """
    Harmonic-to-Noise Ratio via autocorrelation.
    """
    autocorr = librosa.autocorrelate(audio_signal, max_size=int(sample_rate/50))
    if len(autocorr) < 2:
        return 0.0
    zero_crossings = np.where(np.diff(np.sign(autocorr)))[0]
    if len(zero_crossings) == 0 or zero_crossings[0] >= len(autocorr) - 1:
        return 0.0
    first_zero = zero_crossings[0]
    harmonic_peak = np.max(autocorr[first_zero:])
    noise_floor = autocorr[0] - harmonic_peak
    if noise_floor <= 0:
        return 20.0
    return float(10 * np.log10((harmonic_peak + 1e-12) / (noise_floor + 1e-12)))

def extract_emotion_feature(audio_signal, sample_rate=16000):
    """
    Extracts classical emotion acoustic vector (32-dim):
    - MFCC statistics (13 means + 13 stds = 26)
    - Non-linear dynamics: TKEO energy (1)
    - Harmonic-to-Noise Ratio (HNR) (1)
    - Spectral roll-off mean (1) and zero-crossing rate mean (1)
    - Spectral flux / onset novelty mean (1) and std (1)
    """
    mfcc = librosa.feature.mfcc(y=audio_signal, sr=sample_rate, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    # Scale mfcc[0] (overall gain) to balance with timbre coefficients
    mfcc_mean_scaled = mfcc_mean.copy()
    mfcc_mean_scaled[0] /= 10.0
    
    tkeo_val = compute_tkeo(audio_signal) * 10000.0
    hnr_val = compute_hnr(audio_signal, sample_rate)
    
    # Scale rolloff from ~3000 Hz down to ~30 to match MFCC magnitude
    rolloff = librosa.feature.spectral_rolloff(y=audio_signal, sr=sample_rate)
    rolloff_scaled = float(np.mean(rolloff)) / 100.0
    
    # Scale ZCR from ~0.1 up to ~15
    zcr = librosa.feature.zero_crossing_rate(audio_signal)
    zcr_scaled = float(np.mean(zcr)) * 100.0
    
    # Scale onset envelope metrics to ~15-40
    onset_env = librosa.onset.onset_strength(y=audio_signal, sr=sample_rate)
    onset_mean_scaled = float(np.mean(onset_env)) * 10.0
    onset_std_scaled = float(np.std(onset_env)) * 10.0
    
    extra_dynamics = np.array([tkeo_val, hnr_val, rolloff_scaled, zcr_scaled, onset_mean_scaled, onset_std_scaled], dtype=np.float32)
    feature = np.concatenate([mfcc_mean_scaled, mfcc_std, extra_dynamics])
    return feature.astype(np.float32)

from scipy.ndimage import maximum_filter
import hashlib

def extract_constellation_hashes(audio_signal, sample_rate=16000):
    stft = librosa.stft(audio_signal, n_fft=2048, hop_length=512)
    magnitude = np.abs(stft)
    
    neighborhood_size = 15
    local_max = maximum_filter(magnitude, size=neighborhood_size) == magnitude
    
    threshold = np.mean(magnitude) * 3
    peaks_mask = local_max & (magnitude > threshold)
    
    freq_bins, time_frames = np.where(peaks_mask)
    
    sort_idx = np.argsort(time_frames)
    freq_bins = freq_bins[sort_idx]
    time_frames = time_frames[sort_idx]
    
    target_zone_size = 5
    hashes = []
    
    for i in range(len(time_frames)):
        t1 = time_frames[i]
        f1 = freq_bins[i]
        
        for j in range(1, target_zone_size + 1):
            if i + j < len(time_frames):
                t2 = time_frames[i + j]
                f2 = freq_bins[i + j]
                
                delta_t = t2 - t1
                
                if 0 < delta_t < 100:
                    hash_str = f"{f1}|{f2}|{delta_t}"
                    h = int(hashlib.md5(hash_str.encode('utf-8')).hexdigest()[:8], 16)
                    hashes.append((h, t1))
                    
    return hashes
