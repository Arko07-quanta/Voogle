import os
import subprocess
import numpy as np
import librosa
import soundfile as sf
from src.tasks.transcription import extract_lpc

WORDS = ["hello", "voogle", "search", "engine", "classical", "signal", "processing", "today", "is", "amazing"]

def get_lpc_for_audio(audio_path, sample_rate=16000):
    y, sr = librosa.load(audio_path, sr=sample_rate)
    # Trim silence
    y, _ = librosa.effects.trim(y, top_db=30)
    
    pre_emphasized = np.append(y[0], y[1:] - 0.97 * y[:-1])
    frame_length = int(sample_rate * 0.025)
    hop_length = int(sample_rate * 0.010)
    frames = librosa.util.frame(pre_emphasized, frame_length=frame_length, hop_length=hop_length).T
    
    window = np.hamming(frame_length)
    lpc_features = []
    for frame in frames:
        lpc = extract_lpc(frame * window, order=12)
        lpc_features.append(lpc[1:])
        
    return np.array(lpc_features)

def build_templates():
    print("Building Real LPC Templates for Transcription...")
    os.makedirs("datasets", exist_ok=True)
    os.makedirs("eval_temp", exist_ok=True)
    
    templates = {}
    for word in WORDS:
        print(f"Synthesizing template for: {word}")
        wav_path = f"eval_temp/word_{word}.wav"
        mp3_path = f"eval_temp/word_{word}.mp3"
        
        try:
            subprocess.run([
                "edge-tts", "--voice", "en-US-AriaNeural", "--text", word, "--write-media", mp3_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            y, sr = librosa.load(mp3_path, sr=16000)
            sf.write(wav_path, y, sr)
            
            lpc_seq = get_lpc_for_audio(wav_path)
            templates[word] = lpc_seq
            print(f" -> Extracted LPC shape: {lpc_seq.shape}")
        except Exception as e:
            print(f"Failed to process {word}: {e}")
            
    np.save("datasets/transcription_templates.npy", templates)
    print("Saved real templates to datasets/transcription_templates.npy")

if __name__ == "__main__":
    build_templates()
