import os
import shutil
import librosa
import soundfile as sf
import subprocess
import time
from src.tasks.retrieval import add_song_sample, add_voice_sample

def create_demo_songs():
    print("Populating Song Retrieval DB...")
    songs = ["brahms", "choice", "fishin", "nutcracker", "pistachio", "sweetwaltz", "trumpet", "vibeace"]
    os.makedirs("demo_audio", exist_ok=True)
    
    for s in songs:
        try:
            path = librosa.example(s)
            y, sr = librosa.load(path, sr=16000)
            
            # Save the full clean track to the database
            db_path = f"demo_audio/db_song_{s}.wav"
            sf.write(db_path, y, sr)
            add_song_sample(s, db_path, sr)
            
            # Create a cropped noisy demo file for the user to search with!
            demo_path = f"demo_audio/demo_search_song_{s}.wav"
            start = len(y) // 2
            chunk = y[start : start + 5*sr] # 5 seconds
            noise = np.random.randn(len(chunk)) * 0.02
            sf.write(demo_path, chunk + noise, sr)
            
            print(f"Added {s} to Song DB and created demo file.")
        except Exception as e:
            print(f"Failed song {s}: {e}")

import numpy as np

def create_demo_voices():
    print("\nPopulating Voice ID DB...")
    
    # 3 Real human voices from LibriSpeech (via librosa)
    voices_libri = {"Garth_Comira": "libri1", "Anders_Lankford": "libri2", "Heather_Barnett": "libri3"}
    
    for name, s in voices_libri.items():
        path = librosa.example(s)
        y, sr = librosa.load(path, sr=16000)
        
        # Save reference for DB (first 10 seconds)
        db_path = f"demo_audio/db_voice_{name}.wav"
        sf.write(db_path, y[:10*sr], sr)
        add_voice_sample(name, db_path, sr)
        
        # Save a different clip for demo search (next 5 seconds)
        demo_path = f"demo_audio/demo_search_voice_{name}.wav"
        sf.write(demo_path, y[10*sr : 15*sr], sr)
        print(f"Added {name} to Voice DB.")
        
    # Generate some TTS characters
    tts_characters = {
        "Professor_Andrew": ("en-US-AndrewNeural", "Welcome to the classical signal processing lecture!"),
        "Assistant_Ava": ("en-US-AvaNeural", "I have finished indexing the constellation fingerprints."),
        "Agent_Steffan": ("en-US-SteffanNeural", "Target acquired. Geometric alignment complete.")
    }
    
    for name, (voice_id, text) in tts_characters.items():
        mp3_path = f"demo_audio/temp_{name}.mp3"
        subprocess.run(["edge-tts", "--voice", voice_id, "--text", text, "--write-media", mp3_path], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        y, sr = librosa.load(mp3_path, sr=16000)
        
        db_path = f"demo_audio/db_voice_{name}.wav"
        sf.write(db_path, y, sr)
        add_voice_sample(name, db_path, sr)
        
        # Demo file is the exact same for TTS since it's just a demo
        demo_path = f"demo_audio/demo_search_voice_{name}.wav"
        shutil.copy2(db_path, demo_path)
        os.remove(mp3_path)
        print(f"Added {name} to Voice DB.")
        time.sleep(1)

if __name__ == "__main__":
    create_demo_songs()
    create_demo_voices()
    print("\nRetrieval Databases Populated Successfully!")
