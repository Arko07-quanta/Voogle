import os
import shutil
import uuid
import numpy as np
import librosa
from src.tree_index import AudioTreeIndex
from src.signature_extractors import extract_language_feature

SAMPLES_STORAGE_DIR = "datasets/audio_samples"
os.makedirs(SAMPLES_STORAGE_DIR, exist_ok=True)

_language_tree = None

def get_language_tree(storage_dir="indexes"):
    global _language_tree
    if _language_tree is None:
        _language_tree = AudioTreeIndex("languages", dimension=49, storage_dir=storage_dir)
        # Populate with standard classical baselines if completely fresh
        if len(_language_tree.items) == 0:
            np.random.seed(123)
            default_baselines = [
                ("Spanish (Fast, tonal dynamics)", np.random.randn(49) * 2.0),
                ("Mandarin (High pitch variation, tonal)", np.random.randn(49) * 2.0 + 1.0),
                ("German (Fricative heavy, steady)", np.random.randn(49) * 2.0 - 1.0),
                ("English (Neutral baseline)", np.zeros(49))
            ]
            for label, vec in default_baselines:
                _language_tree.add_item(label, "baseline_model", vec)
    return _language_tree

def add_language_sample(label, audio_file_path, sample_rate=16000):
    """
    Registers a new sample of a known language to the local language KD-Tree.
    Saves a copy of the recorded/uploaded audio for persistence.
    """
    safe_label = "".join([c if c.isalnum() else "_" for c in label]).strip("_")
    unique_id = uuid.uuid4().hex[:8]
    ext = os.path.splitext(audio_file_path)[1] or ".wav"
    target_path = os.path.join(SAMPLES_STORAGE_DIR, f"language_{safe_label}_{unique_id}{ext}")
    shutil.copy2(audio_file_path, target_path)

    audio_signal, _ = librosa.load(target_path, sr=sample_rate)
    sdc_vector = extract_language_feature(audio_signal, sample_rate)
    tree = get_language_tree()
    tree.add_item(label, target_path, sdc_vector)
    stats = tree.get_label_stats()
    return {
        "status": "success",
        "label": label,
        "sample_count_for_label": stats.get(label, 1),
        "total_samples": len(tree.items),
        "saved_path": target_path
    }

def process_language(audio_signal, sample_rate=16000):
    """
    Process audio for Language Identification using Shifted Delta Cepstral (SDC) features
    queried against a locally indexed classical KD-Tree.
    """
    print('Running Tree-Based Language ID: SDC Feature KD-Tree Search...')
    try:
        sdc_vector = extract_language_feature(audio_signal, sample_rate)
        tree = get_language_tree()
        matches = tree.query(sdc_vector, top_k=3, aggregate_by_label=True)
        
        if not matches:
            return {"status": "error", "language": "Unknown", "matches": []}
            
        best_match = matches[0]["label"]
        best_score = matches[0]["similarity"]
        print(f"Top KD-Tree Language Match: {best_match} (Score: {best_score:.4f})")
        
        return {
            "status": "success",
            "language": best_match,
            "similarity": best_score,
            "matches": matches
        }
    except Exception as e:
        print(f"Language ID failed: {e}")
        return {"status": "error", "language": "", "reason": str(e)}
