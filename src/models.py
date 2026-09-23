import numpy as np

def get_general_purpose_embeddings(mel_spec_db):
    """
    Computes statistical features (mean and std across time) from the mel spectrogram 
    to serve as a general-purpose 'embedding' vector for classical signal processing.
    """
    if mel_spec_db.ndim > 2:
        mel_spec_db = np.squeeze(mel_spec_db)
        
    mean_features = np.mean(mel_spec_db, axis=1)
    std_features = np.std(mel_spec_db, axis=1)
    
    # Concatenate mean and std to form the embedding vector
    embeddings = np.concatenate([mean_features, std_features])
    return embeddings
