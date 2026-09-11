import numpy as np
from transformers import ASTFeatureExtractor
import torch

# Load the SOTA feature extractor corresponding to our model
feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

def extract_mel_spectrogram(audio_signal, sample_rate):
    """
    Extracts Mel-frequency spectrograms using SOTA ASTFeatureExtractor.
    This maintains the 'Mel filter bank' approach but optimizes it for our SOTA model.
    """
    # The AST feature extractor handles the mel-spectrogram creation
    inputs = feature_extractor(audio_signal, sampling_rate=sample_rate, return_tensors="pt")
    # We return the input_values which are the mel-spectrograms expected by the AST model
    return inputs.input_values
