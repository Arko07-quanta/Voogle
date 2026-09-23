import numpy as np
import librosa
from scipy.spatial.distance import cdist

def levinson_durbin(r, order):
    """
    Solves the Yule-Walker equations using the Levinson-Durbin recursion.
    Returns the Linear Predictive Coding (LPC) coefficients.
    """
    a = np.zeros(order + 1)
    e = np.zeros(order + 1)
    
    a[0] = 1.0
    e[0] = r[0]
    
    for i in range(1, order + 1):
        acc = sum(a[j] * r[i - j] for j in range(1, i))
        k = (r[i] - acc) / e[i - 1]
        
        a_new = np.copy(a)
        for j in range(1, i):
            a_new[j] = a[j] - k * a[i - j]
        a_new[i] = k
        a = a_new
        e[i] = (1 - k * k) * e[i - 1]
        
    return a

def extract_lpc(signal, order=10):
    """
    Extracts LPC coefficients for a given signal frame.
    """
    # Autocorrelation
    r = np.correlate(signal, signal, mode='full')
    r = r[len(signal)-1 : len(signal)+order]
    
    if r[0] == 0:
        return np.zeros(order + 1)
        
    return levinson_durbin(r, order)

def dtw_distance(seq1, seq2):
    """
    Dynamic Time Warping (DTW) to find the optimal alignment distance between two sequences.
    Uses pure dynamic programming.
    """
    N, M = len(seq1), len(seq2)
    # Distance matrix
    dist_mat = cdist(seq1, seq2, metric='euclidean')
    
    # Cost matrix
    cost = np.full((N + 1, M + 1), np.inf)
    cost[0, 0] = 0
    
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            cost[i, j] = dist_mat[i-1, j-1] + min(cost[i-1, j],    # insertion
                                                  cost[i, j-1],    # deletion
                                                  cost[i-1, j-1])  # match
    return cost[N, M]

def process_transcription(audio_signal, sample_rate=16000):
    """
    Process audio for Speech-to-Text using rigorous LPC feature extraction and DTW matching.
    """
    print('Running Hard Math ASR: Levinson-Durbin LPC & Dynamic Time Warping (DTW)...')
    try:
        # Pre-emphasis filter to boost high frequencies (standard in classical ASR)
        pre_emphasized = np.append(audio_signal[0], audio_signal[1:] - 0.97 * audio_signal[:-1])
        
        # Frame the signal
        frame_length = int(sample_rate * 0.025)
        hop_length = int(sample_rate * 0.010)
        frames = librosa.util.frame(pre_emphasized, frame_length=frame_length, hop_length=hop_length).T
        
        # Apply Hamming window and extract LPC for each frame
        window = np.hamming(frame_length)
        lpc_features = []
        for frame in frames:
            lpc = extract_lpc(frame * window, order=12)
            lpc_features.append(lpc[1:]) # Drop the 1.0 coefficient
            
        lpc_features = np.array(lpc_features)
        
        if len(lpc_features) == 0:
            return {"status": "success", "text": ""}
            
        # Mock "Ideal" Phoneme LPC Templates for DTW matching
        # In a real classical system, these templates are pre-computed from a clean corpus
        np.random.seed(42)
        mock_templates = {
            "hello": np.random.randn(15, 12) * 0.5,
            "voogle": np.random.randn(20, 12) * 0.5,
            "search": np.random.randn(18, 12) * 0.5,
            "engine": np.random.randn(16, 12) * 0.5
        }
        
        # Sub-sequence matching via sliding DTW window
        # We will match small chunks of the incoming LPCs against the dictionary
        step = 20
        transcription = []
        for start in range(0, len(lpc_features) - step, step):
            chunk = lpc_features[start : start + step]
            
            best_match = None
            min_dist = float('inf')
            
            for word, template in mock_templates.items():
                dist = dtw_distance(chunk, template)
                # Normalize by path length approx
                norm_dist = dist / (len(chunk) + len(template))
                if norm_dist < min_dist:
                    min_dist = norm_dist
                    best_match = word
                    
            if min_dist < 2.0: # Threshold for a match
                transcription.append(best_match)
                
        # Remove consecutive duplicates
        final_transcription = [v for i, v in enumerate(transcription) if i == 0 or v != transcription[i-1]]
        
        result_text = " ".join(final_transcription) if final_transcription else "unrecognized"
        
        print(f"Transcription Result: {result_text}")
        return {"status": "success", "text": result_text}
    except Exception as e:
        print(f"Transcription failed: {e}")
        return {"status": "error", "text": ""}
