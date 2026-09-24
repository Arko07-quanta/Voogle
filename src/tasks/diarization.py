import numpy as np
import librosa
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform

def compute_normalized_laplacian(affinity_matrix):
    """
    Computes the Normalized Graph Laplacian: L = I - D^{-1/2} A D^{-1/2}
    """
    degree_matrix = np.diag(np.sum(affinity_matrix, axis=1))
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(degree_matrix) + 1e-12))
    identity = np.eye(affinity_matrix.shape[0])
    normalized_laplacian = identity - np.dot(d_inv_sqrt, np.dot(affinity_matrix, d_inv_sqrt))
    return normalized_laplacian

def spectral_bisection_cluster(spectral_embedding, k=2):
    """
    Classical spectral bisection clustering using the Fiedler vector / eigen-projection.
    Partitions segments based on sign and median splitting in the spectral embedding space.
    """
    n_samples = spectral_embedding.shape[0]
    if n_samples <= k:
        return np.arange(n_samples)
    
    if k == 2 and spectral_embedding.shape[1] >= 2:
        fiedler = spectral_embedding[:, 1]
        threshold = np.median(fiedler)
        labels = (fiedler > threshold).astype(int)
        if len(np.unique(labels)) == 1 and n_samples >= 2:
            labels[0] = 1 - labels[0]
        return labels
    
    z = linkage(spectral_embedding, method='ward')
    labels = fcluster(z, t=k, criterion='maxclust') - 1
    return labels

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
            pairwise_dists = squareform(pdist(X, 'euclidean'))
            gamma = 1.0 / (2.0 * np.var(pairwise_dists) + 1e-6)
            affinity_matrix = np.exp(-gamma * (pairwise_dists ** 2))
            np.fill_diagonal(affinity_matrix, 0)
            
            # 4. Compute Graph Laplacian
            laplacian = compute_normalized_laplacian(affinity_matrix)
            
            # 5. Eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
            k_speakers = min(2, n_samples)
            spectral_embedding = eigenvectors[:, :k_speakers]
            norms = np.linalg.norm(spectral_embedding, axis=1, keepdims=True)
            spectral_embedding = spectral_embedding / (norms + 1e-12)
            
            # 6. Cluster in the Eigen-space using classical spectral bisection / linkage
            labels = spectral_bisection_cluster(spectral_embedding, k=k_speakers)
            
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
