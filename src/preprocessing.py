import numpy as np

def clip_audio(audio_signal, threshold):
    """
    Clips the audio signal based on a given threshold.
    """
    return np.clip(audio_signal, -threshold, threshold)

def sliding_window(audio_signal, window_size, step_size):
    """
    Applies a sliding window to the audio signal.
    
    Args:
        audio_signal (np.ndarray): 1D array representing the audio signal.
        window_size (int): The size of each window.
        step_size (int): The step size between windows.
        
    Returns:
        np.ndarray: 2D array where each row is a window.
    """
    num_windows = (len(audio_signal) - window_size) // step_size + 1
    if num_windows <= 0:
        return np.array([])
    
    windows = np.array([audio_signal[i * step_size : i * step_size + window_size] for i in range(num_windows)])
    return windows
