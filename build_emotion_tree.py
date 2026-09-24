import os
import glob
import time
import numpy as np
import librosa
from src.tree_index import AudioTreeIndex
from src.signature_extractors import extract_emotion_feature

RAVDESS_EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def build_emotion_tree(emotions_dir="datasets/emotions", storage_dir="indexes", sample_rate=16000, max_per_class=40):
    print("=" * 60)
    print("Building Classical Metric KD-Tree for Emotions from RAVDESS...")
    print("=" * 60)
    
    wav_files = glob.glob(os.path.join(emotions_dir, "**/*.wav"), recursive=True)
    if not wav_files:
        print(f"No .wav files found in {emotions_dir}!")
        return None

    print(f"Found {len(wav_files)} total audio files in {emotions_dir}.")
    
    # Initialize 32-dimensional KDTree index for emotions
    emotion_tree = AudioTreeIndex("emotions", dimension=32, storage_dir=storage_dir)
    # Clear existing vectors to ensure clean rebuild from dataset
    emotion_tree.items = []
    emotion_tree.vectors = np.empty((0, 32), dtype=np.float32)
    
    counts = {}
    items_to_add = []
    
    t0 = time.time()
    for i, path in enumerate(wav_files):
        filename = os.path.basename(path)
        parts = filename.split(".")[0].split("-")
        
        # RAVDESS format: 03-01-06-01-02-01-12.wav -> 3rd element is emotion code
        if len(parts) >= 3 and parts[2] in RAVDESS_EMOTIONS:
            emotion_label = RAVDESS_EMOTIONS[parts[2]]
        else:
            emotion_label = "unknown"
            
        current_cnt = counts.get(emotion_label, 0)
        if current_cnt >= max_per_class:
            continue
            
        try:
            signal, _ = librosa.load(path, sr=sample_rate)
            feat = extract_emotion_feature(signal, sample_rate)
            items_to_add.append((emotion_label, path, feat))
            counts[emotion_label] = current_cnt + 1
        except Exception as e:
            print(f"Error processing {path}: {e}")
            
        if (len(items_to_add)) % 50 == 0:
            print(f"Processed {len(items_to_add)} emotion samples...")
            
    print(f"Adding {len(items_to_add)} samples to the KD-Tree...")
    emotion_tree.add_items_batch(items_to_add)
    t_elapsed = time.time() - t0
    
    print("\n✅ Emotion KD-Tree built successfully!")
    print(f"Time taken: {t_elapsed:.2f}s")
    print(f"Total samples indexed: {len(emotion_tree.items)}")
    print("Class distribution:")
    for em, c in counts.items():
        print(f"  - {em}: {c} samples")
        
    return emotion_tree

if __name__ == "__main__":
    build_emotion_tree()
