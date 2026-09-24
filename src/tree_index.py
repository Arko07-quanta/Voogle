import os
import json
import numpy as np
from scipy.spatial import KDTree

class AudioTreeIndex:
    """
    Classical metric space search structure (KD-Tree / Spatial Tree)
    for indexing and searching audio feature vectors locally.
    Supports persistent storage (JSON/NPZ), multi-sample label grouping,
    and incremental updates.
    """
    def __init__(self, index_name, dimension, storage_dir="indexes"):
        self.index_name = index_name
        self.dimension = dimension
        self.storage_dir = storage_dir
        self.meta_file = os.path.join(storage_dir, f"{index_name}_meta.json")
        self.vectors_file = os.path.join(storage_dir, f"{index_name}_vectors.npy")
        
        self.items = []       # List of dicts: {"id": int, "label": str, "path": str}
        self.vectors = np.empty((0, dimension), dtype=np.float32)
        self.tree = None
        
        os.makedirs(storage_dir, exist_ok=True)
        self.load()

    def add_item(self, label, file_path, vector):
        """
        Adds an audio sample vector to the tree index.
        Multiple samples under the same label (e.g. same speaker or artist) are supported.
        """
        vector = np.asarray(vector, dtype=np.float32).flatten()
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension mismatch. Expected {self.dimension}, got {len(vector)}")

        # Normalize vector for cosine distance indexing
        norm = np.linalg.norm(vector)
        unit_vector = vector / (norm + 1e-12)

        item_id = len(self.items)
        self.items.append({"id": item_id, "label": label, "path": file_path})
        
        if self.vectors.shape[0] == 0:
            self.vectors = unit_vector[np.newaxis, :]
        else:
            self.vectors = np.vstack([self.vectors, unit_vector[np.newaxis, :]])
            
        self._rebuild_tree()
        self.save()

    def _rebuild_tree(self):
        if self.vectors.shape[0] > 0:
            self.tree = KDTree(self.vectors)
        else:
            self.tree = None

    def query(self, query_vector, top_k=5, aggregate_by_label=True):
        """
        Searches the tree for the closest matches using Euclidean distance on unit vectors
        (equivalent to cosine similarity ranking: d^2 = 2 - 2*cos_sim).
        
        When aggregate_by_label is True, multiple samples from the same person/entity
        are pooled (using max similarity and sample counts) to present a consolidated ranking.
        """
        if self.tree is None or len(self.items) == 0:
            return []

        query_vec = np.asarray(query_vector, dtype=np.float32).flatten()
        norm = np.linalg.norm(query_vec)
        unit_query = query_vec / (norm + 1e-12)

        k = min(max(top_k * 3, 10), len(self.items))
        distances, indices = self.tree.query(unit_query, k=k)

        if k == 1:
            distances = [distances]
            indices = [indices]

        raw_matches = []
        for dist, idx in zip(distances, indices):
            item = self.items[idx]
            cos_sim = 1.0 - (float(dist) ** 2) / 2.0
            raw_matches.append({
                "label": item["label"],
                "path": item["path"],
                "distance": float(dist),
                "similarity": float(np.clip(cos_sim, -1.0, 1.0))
            })

        if not aggregate_by_label:
            return raw_matches[:top_k]

        # Group by label (e.g. same speaker recorded multiple times)
        grouped = {}
        for m in raw_matches:
            lbl = m["label"]
            if lbl not in grouped:
                grouped[lbl] = {
                    "label": lbl,
                    "best_similarity": m["similarity"],
                    "similarities": [m["similarity"]],
                    "sample_paths": [m["path"]],
                    "sample_count": 1
                }
            else:
                grouped[lbl]["similarities"].append(m["similarity"])
                grouped[lbl]["sample_paths"].append(m["path"])
                grouped[lbl]["sample_count"] += 1
                if m["similarity"] > grouped[lbl]["best_similarity"]:
                    grouped[lbl]["best_similarity"] = m["similarity"]

        # Sort aggregated labels by their top similarity
        sorted_labels = sorted(grouped.values(), key=lambda x: x["best_similarity"], reverse=True)
        
        results = []
        for g in sorted_labels[:top_k]:
            results.append({
                "label": g["label"],
                "similarity": g["best_similarity"],
                "avg_similarity": float(np.mean(g["similarities"])),
                "sample_count": g["sample_count"],
                "path": g["sample_paths"][0]
            })
        return results

    def get_label_stats(self):
        """
        Returns summary of indexed labels and how many samples each has.
        """
        counts = {}
        for item in self.items:
            lbl = item["label"]
            counts[lbl] = counts.get(lbl, 0) + 1
        return counts

    def save(self):
        with open(self.meta_file, 'w', encoding='utf-8') as f:
            json.dump(self.items, f, indent=2)
        np.save(self.vectors_file, self.vectors)

    def load(self):
        if os.path.exists(self.meta_file) and os.path.exists(self.vectors_file):
            try:
                with open(self.meta_file, 'r', encoding='utf-8') as f:
                    self.items = json.load(f)
                self.vectors = np.load(self.vectors_file)
                self._rebuild_tree()
            except Exception as e:
                print(f"Failed to load tree index {self.index_name}: {e}")
