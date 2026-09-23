import pyttsx3
import numpy as np
import soundfile as sf
import os

print("Generating test_speech.wav (Transcription/Language/Emotion)...")
engine = pyttsx3.init()
engine.save_to_file('Hello! I am a highly advanced artificial intelligence voice engine. I am feeling absolutely fantastic today!', 'test_speech.wav')
engine.runAndWait()

print("Generating test_diarization.wav (Two speakers)...")
# Speaker 1
engine.setProperty('rate', 150)
engine.save_to_file('Hello Bob, how are you doing today?', 'temp1.wav')
engine.runAndWait()

# Speaker 2 (change pitch/rate/voice if possible, or just rate)
voices = engine.getProperty('voices')
if len(voices) > 1:
    engine.setProperty('voice', voices[1].id)
engine.setProperty('rate', 120)
engine.save_to_file('I am doing great Alice, thank you for asking!', 'temp2.wav')
engine.runAndWait()

# Concatenate
try:
    data1, samplerate = sf.read('temp1.wav')
    data2, _ = sf.read('temp2.wav')
    combined = np.concatenate((data1, data2))
    sf.write('test_diarization.wav', combined, samplerate)
    os.remove('temp1.wav')
    os.remove('temp2.wav')
except Exception as e:
    print(f"Could not merge diarization audio: {e}")

print("Generating test_music.wav (Music/Tempo)...")
sample_rate = 16000
duration = 4.0 # seconds
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
# Create a synthetic "kick drum" beat every 0.5 seconds (120 BPM)
beat_freq = 2.0 # Hz (120 BPM = 2 beats per second)
envelope = np.abs(np.sin(2 * np.pi * beat_freq * t)) ** 10
# Mix with a chord
chord = (np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 554.37 * t) + np.sin(2 * np.pi * 659.25 * t)) / 3.0
audio = chord * envelope * 0.5
sf.write('test_music.wav', audio, sample_rate)

print("\\nSuccess! Created:")
print(" - test_speech.wav")
print(" - test_diarization.wav")
print(" - test_music.wav")
