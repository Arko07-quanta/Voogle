import librosa
import numpy as np

from src.preprocessing import clip_audio, sliding_window
from src.features import extract_mel_spectrogram
from src.models import get_general_purpose_embeddings

from src.tasks.diarization import process_diarization
from src.tasks.emotion import process_emotion
from src.tasks.retrieval import process_retrieval
from src.tasks.transcription import process_transcription
from src.tasks.language import process_language
from src.tasks.music import process_music

class VooglePipeline:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        print("Voogle Pipeline Initialized.")
        
    def process(self, audio_file_path, selected_tasks=None):
        """
        End-to-end processing pipeline for a given audio file.
        """
        print(f"Processing audio file: {audio_file_path}")
        
        # 1. Load Audio
        try:
            audio_signal, sr = librosa.load(audio_file_path, sr=self.sample_rate)
        except Exception as e:
            print(f"Error loading audio: {e}")
            return
        
        # 2. Audio Preprocessing
        print("Applying audio preprocessing...")
        clipped_signal = clip_audio(audio_signal, threshold=0.9)
        # 30ms window, 10ms step
        window_size = int(self.sample_rate * 0.03) 
        step_size = int(self.sample_rate * 0.01)
        windows = sliding_window(clipped_signal, window_size, step_size)
        
        # 3. Feature Extraction
        print("Extracting features...")
        mel_spec = extract_mel_spectrogram(clipped_signal, self.sample_rate)
        
        # 4. Signal Features Integration (Embeddings)
        print("Generating audio signal embeddings...")
        embeddings = get_general_purpose_embeddings(mel_spec)
        print(f"Generated embedding vector of shape: {embeddings.shape}")
        # 5. Execute Tasks
        print("\n--- Executing Voogle Tasks (using SOTA signal processing) ---")
        results = self.run_tasks(embeddings, audio_file_path, audio_signal, selected_tasks)
        return results
        
    def run_tasks(self, embeddings, audio_file_path, audio_signal, selected_tasks=None):
        results = {}
        
        # If no specific tasks provided, default to all for backward compatibility
        if selected_tasks is None:
            selected_tasks = ['Diarization', 'Emotion', 'Retrieval', 'Transcription', 'Language', 'Music']
            
        if 'Diarization' in selected_tasks:
            results['diarization'] = process_diarization(audio_signal)
        if 'Emotion' in selected_tasks:
            results['emotion'] = process_emotion(audio_signal)
        if 'Retrieval' in selected_tasks:
            results['retrieval'] = process_retrieval(embeddings) 
        if 'Transcription' in selected_tasks:
            results['transcription'] = process_transcription(audio_signal)
        if 'Language' in selected_tasks:
            results['language'] = process_language(audio_signal)
        if 'Music' in selected_tasks:
            results['music'] = process_music(audio_file_path)
            
        return results

if __name__ == "__main__":
    pipeline = VooglePipeline()
    # Provide a dummy path or a real audio path here
    pipeline.process("test_audio.wav", selected_tasks=['Diarization', 'Emotion', 'Retrieval', 'Transcription', 'Language', 'Music'])
    print("\nVoogle core engine is ready.")
