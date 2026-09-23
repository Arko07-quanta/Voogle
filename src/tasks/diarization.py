import numpy as np
import librosa
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans

def compute_normalized_laplacian(affinity_matrix):
    """
    Computes the Normalized Graph Laplacian: L = I - D^{-1/2} A D^{-1/2}
    """
    degree_matrix = np.diag(np.sum(affinity_matrix, axis=1))
    # D^{-1/2}
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(degree_matrix) + 1e-12))
    
    # L_sym = I - D^{-1/2} * A * D^{-1/2}
    identity = np.eye(affinity_matrix.shape[0])
    normalized_laplacian = identity - np.dot(d_inv_sqrt, np.dot(affinity_matrix, d_inv_sqrt))
    
    return normalized_laplacian

def process_diarization(audio_signal, sample_rate=16000):
    """
    Process audio for Speaker Detection & Diarization using Spectral Graph Theory.
    Projects speech features into a non-linear Laplacian eigen-space before clustering.
    """
    print('Running Hard Math Diarization: Spectral Graph Laplacian Clustering...')
    try:
        # 1. Voice Activity Detection (VAD)
        non_mute_intervals = librosa.effects.split(audio_signal, top_db=20)
        
        if len(non_mute_intervals) == 0:
            return {"status": "success", "speakers": []}

        # 2. Extract features
        segment_features = []
        valid_intervals = []
        for start_i, end_i in non_mute_intervals:
            segment = audio_signal[start_i:end_i]
            if len(segment) < 2048:
                continue
            mfcc = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=13)
            segment_features.append(np.mean(mfcc, axis=1))
            valid_intervals.append((start_i, end_i))
            
        if not segment_features:
            return {"status": "success", "speakers": []}
            
        X = np.array(segment_features)
        n_samples = X.shape[0]
        
        if n_samples < 2:
            labels = np.zeros(n_samples, dtype=int)
        else:
            # 3. Construct Affinity Matrix using RBF Kernel
            # A_ij = exp(-gamma * ||x_i - x_j||^2)
            pairwise_dists = squareform(pdist(X, 'euclidean'))
            gamma = 1.0 / (2.0 * np.var(pairwise_dists) + 1e-6)
            affinity_matrix = np.exp(-gamma * (pairwise_dists ** 2))
            
            # Zero out diagonal to avoid self-loops dominating
            np.fill_diagonal(affinity_matrix, 0)
            
            # 4. Compute Graph Laplacian
            laplacian = compute_normalized_laplacian(affinity_matrix)
            
            # 5. Eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
            
            # We want to separate into a known number of speakers, let's assume k=2
            k_speakers = min(2, n_samples)
            
            # Extract the eigenvectors corresponding to the k smallest eigenvalues (skip the very first which is 0)
            # Actually for Normalized Laplacian, the multiplicity of 0 is the number of connected components.
            # We take the first k eigenvectors.
            spectral_embedding = eigenvectors[:, :k_speakers]
            
            # Normalize rows of the embedding matrix
            norms = np.linalg.norm(spectral_embedding, axis=1, keepdims=True)
            spectral_embedding = spectral_embedding / (norms + 1e-12)
            
            # 6. Cluster in the Eigen-space
            kmeans = KMeans(n_clusters=k_speakers, random_state=42, n_init=10)
            labels = kmeans.fit_predict(spectral_embedding)
            
        speakers_detected = []
        for i, (start_i, end_i) in enumerate(valid_intervals):
            start_s = start_i / sample_rate
            end_s = end_i / sample_rate
            speaker = f"SPEAKER_{labels[i]:02d}"
            print(f"start={start_s:.1f}s stop={end_s:.1f}s {speaker}")
            speakers_detected.append({"start": start_s, "end": end_s, "speaker": speaker})
            
        return {"status": "success", "speakers": speakers_detected}
    except Exception as e:
        print(f"Diarization failed: {e}")
        return {"status": "error", "reason": str(e)}
