import os
import shutil
import uuid
import numpy as np
import librosa
from src.tree_index import AudioTreeIndex
from src.signature_extractors import extract_voice_feature, extract_song_feature

# Directory to permanently archive recorded/uploaded dataset files
VOICES_STORAGE_DIR = "datasets/voices"
SONGS_STORAGE_DIR = "datasets/songs"
os.makedirs(VOICES_STORAGE_DIR, exist_ok=True)
os.makedirs(SONGS_STORAGE_DIR, exist_ok=True)

# Shared tree instances
_voice_tree = None
_song_tree = None

def get_voice_tree(storage_dir="indexes"):
    global _voice_tree
    if _voice_tree is None:
        _voice_tree = AudioTreeIndex("voices", dimension=28, storage_dir=storage_dir)
    return _voice_tree

def get_song_tree(storage_dir="indexes"):
    global _song_tree
    if _song_tree is None:
        _song_tree = AudioTreeIndex("songs", dimension=274, storage_dir=storage_dir)
    return _song_tree

def _persist_audio_sample(audio_file_path, category, label):
    """
    Saves a copy of the recorded/uploaded temporary audio into a persistent local folder.
    """
    safe_label = "".join([c if c.isalnum() else "_" for c in label]).strip("_")
    unique_id = uuid.uuid4().hex[:8]
    ext = os.path.splitext(audio_file_path)[1]
    if not ext:
        ext = ".wav"
    folder = VOICES_STORAGE_DIR if category == "voice" else SONGS_STORAGE_DIR
    target_name = f"{category}_{safe_label}_{unique_id}{ext}"
    target_path = os.path.join(folder, target_name)
    shutil.copy2(audio_file_path, target_path)
    return target_path

def add_voice_sample(label, audio_file_path, sample_rate=16000):
    """
    Registers a new speaker/voice sample into the local KD-Tree.
    Can be called multiple times with the same person's name to index multiple voice recordings!
    """
    persisted_path = _persist_audio_sample(audio_file_path, "voice", label)
    audio_signal, _ = librosa.load(persisted_path, sr=sample_rate)
    feat = extract_voice_feature(audio_signal, sample_rate)
    tree = get_voice_tree()
    tree.add_item(label, persisted_path, feat)
    stats = tree.get_label_stats()
    return {
        "status": "success",
        "label": label,
        "sample_count_for_label": stats.get(label, 1),
        "total_samples": len(tree.items),
        "saved_path": persisted_path
    }

def add_song_sample(label, audio_file_path, sample_rate=16000):
    """
    Registers a new song sample into the local KD-Tree.
    """
    persisted_path = _persist_audio_sample(audio_file_path, "song", label)
    audio_signal, _ = librosa.load(persisted_path, sr=sample_rate)
    feat = extract_song_feature(audio_signal, sample_rate)
    tree = get_song_tree()
    tree.add_item(label, persisted_path, feat)
    stats = tree.get_label_stats()
    return {
        "status": "success",
        "label": label,
        "sample_count_for_label": stats.get(label, 1),
        "total_samples": len(tree.items),
        "saved_path": persisted_path
    }

def process_retrieval(audio_signal=None, audio_embeddings=None, sample_rate=16000, top_k=3, match_type="song"):
    """
    Process Audio Matching & Retrieval using local classical KD-Tree index.
    Matches against indexed songs or voices.
    """
    print(f"Running Tree-Based Audio Matching & Retrieval ({match_type})...")
    
    if match_type == "voice":
        tree = get_voice_tree()
        if audio_signal is None:
            return {"status": "error", "reason": "audio_signal required for voice matching"}
        query_vec = extract_voice_feature(audio_signal, sample_rate)
    else:
        tree = get_song_tree()
        if audio_signal is not None:
            query_vec = extract_song_feature(audio_signal, sample_rate)
        elif audio_embeddings is not None:
            query_vec = np.pad(audio_embeddings.flatten(), (0, max(0, 274 - len(audio_embeddings.flatten()))))[:274]
        else:
            return {"status": "error", "reason": "No audio signal or embeddings provided"}

    if len(tree.items) == 0:
        return {
            "status": "success",
            "matches": [],
            "best_match": None,
            "similarity": 0.0,
            "message": "Local tree index is empty. Add reference samples to index."
        }

    matches = tree.query(query_vec, top_k=top_k, aggregate_by_label=True)
    best_match = matches[0] if matches else None
    sim = best_match["similarity"] if best_match else 0.0

    print(f"Tree search complete. Best match: {best_match['label'] if best_match else 'None'} (Similarity: {sim:.4f})")
    return {
        "status": "success",
        "matches": matches,
        "best_match": best_match,
        "similarity": sim
    }
