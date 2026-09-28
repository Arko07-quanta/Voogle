import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import cdist
import os

os.makedirs('images', exist_ok=True)
plt.style.use('dark_background')

# 1. Language SDC (MFCC Deltas)
def gen_language_sdc():
    y, sr = librosa.load(librosa.ex('brahms'), duration=1.0)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=7)
    delta = librosa.feature.delta(mfccs, width=3)
    
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(delta, x_axis='time', cmap='coolwarm')
    plt.title('Shifted Delta Cepstra (Phonotactic Rhythm)')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig('images/img_language_sdc.png', dpi=300)
    plt.close()

# 2. Diarization Spectral Graph Affinity Matrix
def gen_diarization_graph():
    # Simulate an affinity matrix for speaker clustering
    np.random.seed(42)
    # Speaker 1
    X1 = np.random.randn(20, 13) + np.array([2]*13)
    # Speaker 2
    X2 = np.random.randn(25, 13) - np.array([2]*13)
    X = np.vstack((X1, X2))
    
    pairwise_dists = squareform(pdist(X, 'euclidean'))
    gamma = 0.1
    affinity_matrix = np.exp(-gamma * (pairwise_dists ** 2))
    
    plt.figure(figsize=(4, 4))
    plt.imshow(affinity_matrix, cmap='magma', interpolation='nearest')
    plt.title('RBF Affinity Matrix $A_{ij}$')
    plt.tight_layout()
    plt.savefig('images/img_diarization_graph.png', dpi=300)
    plt.close()

# 3. Transcription DTW Cost Matrix
def gen_transcription_dtw():
    # Simulate a DTW cost matrix alignment path
    np.random.seed(42)
    seq1 = np.random.randn(30, 12)
    seq2 = np.random.randn(40, 12)
    
    dist_mat = cdist(seq1, seq2, metric='euclidean')
    N, M = len(seq1), len(seq2)
    cost = np.full((N + 1, M + 1), np.inf)
    cost[0, 0] = 0
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            cost[i, j] = dist_mat[i-1, j-1] + min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])
            
    plt.figure(figsize=(4, 4))
    plt.imshow(cost[1:, 1:], origin='lower', cmap='viridis', interpolation='nearest')
    plt.title('DTW Dynamic Programming Path')
    plt.tight_layout()
    plt.savefig('images/img_transcription_dtw.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    gen_language_sdc()
    gen_diarization_graph()
    gen_transcription_dtw()
    print("New images generated!")
