import numpy as np
import librosa

def compute_tkeo(signal):
    """
    Computes the Teager-Kaiser Energy Operator (TKEO).
    Psi[x(n)] = x^2(n) - x(n-1)x(n+1)
    """
    tkeo = np.zeros_like(signal)
    # Exclude the boundaries
    tkeo[1:-1] = signal[1:-1]**2 - signal[:-2] * signal[2:]
    return np.mean(tkeo[1:-1])

def compute_hnr(audio_signal, sample_rate):
    """
    Computes a mock Harmonic-to-Noise Ratio (HNR).
    True HNR requires complex pitch tracking and cepstral analysis.
    Here we approximate it using the autocorrelation of the signal.
    """
    autocorr = librosa.autocorrelate(audio_signal, max_size=int(sample_rate/50))
    if len(autocorr) < 2:
        return 0
    # Peak corresponding to the fundamental frequency
    # We ignore the zero-lag peak by starting from lag > 0
    zero_crossings = np.where(np.diff(np.sign(autocorr)))[0]
    if len(zero_crossings) == 0:
        return 0
        
    first_zero = zero_crossings[0]
    if first_zero >= len(autocorr) - 1:
        return 0
        
    harmonic_peak = np.max(autocorr[first_zero:])
    noise_floor = autocorr[0] - harmonic_peak
    
    if noise_floor <= 0:
        return 20.0 # High HNR limit
        
    hnr = 10 * np.log10((harmonic_peak + 1e-12) / (noise_floor + 1e-12))
    return hnr

def process_emotion(audio_signal, sample_rate=16000):
    """
    Process audio for Emotion Identification using non-linear dynamics.
    Deploys Teager-Kaiser Energy Operator (TKEO) and Harmonic-to-Noise Ratio (HNR).
    """
    print('Running Hard Math Emotion ID: TKEO Non-Linear Dynamics & HNR...')
    try:
        # 1. Non-linear Energy (Arousal indicator / Stress)
        tkeo_energy = compute_tkeo(audio_signal)
        # Normalize TKEO magnitude for interpretability
        tkeo_norm = tkeo_energy * 1000.0 
        
        # 2. Harmonic-to-Noise Ratio (Valence / Voice Quality)
        hnr = compute_hnr(audio_signal, sample_rate)
        
        # 3. Micro-tremor extraction (Jitter approximation)
        f0, _, _ = librosa.pyin(audio_signal, fmin=50, fmax=500, sr=sample_rate)
        valid_f0 = f0[~np.isnan(f0)]
        jitter = np.std(np.diff(valid_f0)) / (np.mean(valid_f0) + 1e-6) if len(valid_f0) > 1 else 0.0
        
        print(f"Non-Linear Stats -> TKEO: {tkeo_norm:.4f}, HNR: {hnr:.2f}dB, Jitter: {jitter:.4f}")
        
        # Matrix transformation into Emotion Space
        # Emotion = W * Features + Bias
        
        arousal = (tkeo_norm * 5.0) + (jitter * 100.0) - 2.0
        valence = (hnr * 0.5) - (jitter * 50.0)
        
        arousal = np.clip(arousal, -5, 5)
        valence = np.clip(valence, -5, 5)
        
        if arousal > 0 and valence > 0:
            emotion = "Happy / Excited"
        elif arousal > 0 and valence <= 0:
            emotion = "Angry / Intense"
        elif arousal <= 0 and valence < 0:
            emotion = "Sad / Calm"
        else:
            emotion = "Neutral / Relaxed"
            
        print(f"Projected -> Valence: {valence:.2f}, Arousal: {arousal:.2f}")
        print(f"Emotion Result: {emotion}")
        return {"status": "success", "emotions": emotion}
    except Exception as e:
        print(f"Emotion classification failed: {e}")
        return {"status": "error", "emotions": "Unknown"}
