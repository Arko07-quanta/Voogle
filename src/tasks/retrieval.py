import numpy as np
import librosa
from src.tree_index import AudioTreeIndex
from src.signature_extractors import extract_voice_feature, extract_song_feature

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

def add_voice_sample(label, audio_file_path, sample_rate=16000):
    """
    Registers a new speaker/voice sample into the local KD-Tree.
    """
    audio_signal, _ = librosa.load(audio_file_path, sr=sample_rate)
    feat = extract_voice_feature(audio_signal, sample_rate)
    tree = get_voice_tree()
    tree.add_item(label, audio_file_path, feat)
    return {"status": "success", "label": label, "total_samples": len(tree.items)}

def add_song_sample(label, audio_file_path, sample_rate=16000):
    """
    Registers a new song sample into the local KD-Tree.
    """
    audio_signal, _ = librosa.load(audio_file_path, sr=sample_rate)
    feat = extract_song_feature(audio_signal, sample_rate)
    tree = get_song_tree()
    tree.add_item(label, audio_file_path, feat)
    return {"status": "success", "label": label, "total_samples": len(tree.items)}

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
            # Fallback for backward compatibility if only embeddings passed
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

    matches = tree.query(query_vec, top_k=top_k)
    best_match = matches[0] if matches else None
    sim = best_match["similarity"] if best_match else 0.0

    print(f"Tree search complete. Best match: {best_match['label'] if best_match else 'None'} (Similarity: {sim:.4f})")
    return {
        "status": "success",
        "matches": matches,
        "best_match": best_match,
        "similarity": sim
    }
