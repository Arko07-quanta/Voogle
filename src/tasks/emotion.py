import os
import shutil
import uuid
import numpy as np
import librosa
from src.tree_index import AudioTreeIndex
from src.signature_extractors import extract_emotion_feature

SAMPLES_STORAGE_DIR = "datasets/emotions"
os.makedirs(SAMPLES_STORAGE_DIR, exist_ok=True)

_emotion_tree = None

def get_emotion_tree(storage_dir="indexes"):
    global _emotion_tree
    if _emotion_tree is None:
        _emotion_tree = AudioTreeIndex("emotions", dimension=32, storage_dir=storage_dir)
    return _emotion_tree

def add_emotion_sample(label, audio_file_path, sample_rate=16000):
    """
    Registers a new sample of an emotion into the local emotion KD-Tree.
    Saves a copy of the recorded/uploaded audio.
    """
    safe_label = "".join([c if c.isalnum() else "_" for c in label]).strip("_")
    unique_id = uuid.uuid4().hex[:8]
    ext = os.path.splitext(audio_file_path)[1] or ".wav"
    target_path = os.path.join(SAMPLES_STORAGE_DIR, f"user_emotion_{safe_label}_{unique_id}{ext}")
    shutil.copy2(audio_file_path, target_path)

    audio_signal, _ = librosa.load(target_path, sr=sample_rate)
    feat = extract_emotion_feature(audio_signal, sample_rate)
    tree = get_emotion_tree()
    tree.add_item(label, target_path, feat)
    stats = tree.get_label_stats()
    return {
        "status": "success",
        "label": label,
        "sample_count_for_label": stats.get(label, 1),
        "total_samples": len(tree.items),
        "saved_path": target_path
    }

def process_emotion(audio_signal, sample_rate=16000):
    """
    Process audio for Emotion Identification using classical non-linear features &
    metric KD-Tree matching against indexed emotion datasets (e.g. RAVDESS).
    """
    print('Running Tree-Based Emotion ID: Non-Linear Dynamics & KD-Tree Metric Matching...')
    try:
        feat = extract_emotion_feature(audio_signal, sample_rate)
        tree = get_emotion_tree()
        
        if len(tree.items) == 0:
            return {"status": "error", "emotions": "No emotion index found", "matches": []}
            
        matches = tree.query(feat, top_k=4, aggregate_by_label=True)
        if not matches:
            return {"status": "error", "emotions": "Unknown", "matches": []}
            
        best_match = matches[0]["label"]
        best_score = matches[0]["similarity"]
        
        print(f"Top KD-Tree Emotion Match: {best_match} (Similarity: {best_score:.4f})")
        return {
            "status": "success",
            "emotions": best_match,
            "similarity": best_score,
            "matches": matches
        }
    except Exception as e:
        print(f"Emotion classification failed: {e}")
        return {"status": "error", "emotions": "Unknown", "reason": str(e)}
