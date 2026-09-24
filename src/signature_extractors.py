import numpy as np
import librosa
from src.features import extract_mel_spectrogram
from src.models import get_general_purpose_embeddings
from src.tasks.language import compute_sdc

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
