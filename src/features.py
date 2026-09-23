import numpy as np
import librosa

def extract_mel_spectrogram(audio_signal, sample_rate):
    """
    Extracts Mel-frequency spectrograms using librosa.
    """
    mel_spec = librosa.feature.melspectrogram(y=audio_signal, sr=sample_rate, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db
