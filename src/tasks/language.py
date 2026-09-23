import numpy as np
import librosa
from scipy.spatial.distance import mahalanobis

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
    
    # Standard delta across lag d
    delta = np.zeros_like(mfccs)
    for t in range(num_frames):
        left = max(0, t - d)
        right = min(num_frames - 1, t + d)
        delta[:, t] = mfccs[:, right] - mfccs[:, left]
        
    # Shift and stack
    sdc_features = []
    valid_frames = num_frames - (k * P)
    
    if valid_frames <= 0:
        # Fallback if audio is too short
        return np.mean(delta, axis=1)
        
    for t in range(valid_frames):
        sdc_frame = []
        for i in range(k):
            sdc_frame.append(delta[:, t + i * P])
        sdc_features.append(np.concatenate(sdc_frame))
        
    sdc_features = np.array(sdc_features)
    # Return the mean SDC vector for the utterance
    return np.mean(sdc_features, axis=0)

def process_language(audio_signal, sample_rate=16000):
    """
    Process audio for Language Identification using SDC and Mahalanobis distances.
    """
    print('Running Hard Math Language ID: Shifted Delta Cepstral (SDC) & Mahalanobis Distance...')
    try:
        # 1. Compute MFCCs
        mfccs = librosa.feature.mfcc(y=audio_signal, sr=sample_rate, n_mfcc=13)
        
        # 2. Compute SDC Tensor
        # Using classical parameters N=7, d=1, P=3, k=7 -> yields a 49-dimensional vector
        sdc_vector = compute_sdc(mfccs, N=7, d=1, P=3, k=7)
        
        if sdc_vector.shape[0] != 49:
            # Fallback handling
            sdc_vector = np.pad(sdc_vector, (0, max(0, 49 - sdc_vector.shape[0])))[:49]
            
        # 3. Define Mock Language Covariance and Mean Statistics
        # In a real SOTA system, a Universal Background Model (UBM) provides the covariance
        np.random.seed(123)
        inv_cov_matrix = np.eye(49) * 0.5 # Dummy Inverse Covariance
        
        language_models = {
            "Spanish (Fast, tonal dynamics)": np.random.randn(49) * 2.0,
            "Mandarin (High pitch variation, tonal)": np.random.randn(49) * 2.0 + 1.0,
            "German (Fricative heavy, steady)": np.random.randn(49) * 2.0 - 1.0,
            "English (Neutral baseline)": np.zeros(49)
        }
        
        # 4. Evaluate Likelihoods via Mahalanobis Distance
        best_lang = None
        min_dist = float('inf')
        
        for lang, mean_vec in language_models.items():
            dist = mahalanobis(sdc_vector, mean_vec, inv_cov_matrix)
            if dist < min_dist:
                min_dist = dist
                best_lang = lang
                
        print(f"Mahalanobis Distance to best match: {min_dist:.2f}")
        print(f"Language Result: {best_lang}")
        return {"status": "success", "language": best_lang}
    except Exception as e:
        print(f"Language ID failed: {e}")
        return {"status": "error", "language": ""}
