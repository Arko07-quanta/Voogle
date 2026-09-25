import os
import glob
import time
import numpy as np
import librosa
from src.tree_index import AudioTreeIndex
from src.signature_extractors import extract_language_feature

LANGUAGE_NAMES = {
    "de": "German (Deutsch)",
    "en": "English",
    "es": "Spanish (Español)"
}

def build_language_tree(languages_dir="datasets/languages", storage_dir="indexes", sample_rate=16000, max_per_class=60):
    print("=" * 60)
    print("Building Classical Metric KD-Tree for Languages from Dataset...")
    print("=" * 60)
    
    # Check for audio files (flac, wav, mp3)
    extensions = ("*.flac", "*.wav", "*.mp3", "*.ogg")
    audio_files = []
    for ext in extensions:
        audio_files.extend(glob.glob(os.path.join(languages_dir, "**", ext), recursive=True))

    if not audio_files:
        print(f"No audio files found in {languages_dir}!")
        return None

    print(f"Found {len(audio_files)} total audio files in {languages_dir}.")
    
    # Initialize 49-dimensional KDTree index for languages (SDC features)
    language_tree = AudioTreeIndex("languages", dimension=49, storage_dir=storage_dir)
    language_tree.items = []
    language_tree.vectors = np.empty((0, 49), dtype=np.float32)
    
    counts = {}
    items_to_add = []
    
    t0 = time.time()
    for path in audio_files:
        base = os.path.basename(path)
        parts = base.split("_")
        prefix = parts[0].lower() if parts else "unknown"
        
        # Check folder or prefix
        lang_name = LANGUAGE_NAMES.get(prefix)
        if not lang_name:
            folder_name = os.path.basename(os.path.dirname(path)).lower()
            lang_name = LANGUAGE_NAMES.get(folder_name, folder_name.capitalize())
            
        current_cnt = counts.get(lang_name, 0)
        if current_cnt >= max_per_class:
            continue
            
        try:
            signal, _ = librosa.load(path, sr=sample_rate)
            feat = extract_language_feature(signal, sample_rate)
            items_to_add.append((lang_name, path, feat))
            counts[lang_name] = current_cnt + 1
        except Exception as e:
            print(f"Error processing {path}: {e}")
            
        if len(items_to_add) % 30 == 0:
            print(f"Processed {len(items_to_add)} language samples...")
            
    print(f"Adding {len(items_to_add)} samples to the Language KD-Tree...")
    language_tree.add_items_batch(items_to_add)
    t_elapsed = time.time() - t0
    
    print("\nLanguage KD-Tree built successfully!")
    print(f"Time taken: {t_elapsed:.2f}s")
    print(f"Total samples indexed: {len(language_tree.items)}")
    print("Class distribution:")
    for lang, c in counts.items():
        print(f"  - {lang}: {c} samples")
        
    return language_tree

if __name__ == "__main__":
    build_language_tree()
