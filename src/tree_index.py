import os
import json
import numpy as np
from scipy.spatial import KDTree

class AudioTreeIndex:
    """
    Classical metric space search structure (KD-Tree / Spatial Tree)
    for indexing and searching audio feature vectors locally.
    Supports persistent storage (JSON/NPZ) and incremental updates.
    """
    def __init__(self, index_name, dimension, storage_dir="indexes"):
        self.index_name = index_name
        self.dimension = dimension
        self.storage_dir = storage_dir
        self.meta_file = os.path.join(storage_dir, f"{index_name}_meta.json")
        self.vectors_file = os.path.join(storage_dir, f"{index_name}_vectors.npy")
        
        self.items = []       # List of dicts: {"label": str, "path": str, "id": int}
        self.vectors = np.empty((0, dimension), dtype=np.float32)
        self.tree = None
        
        os.makedirs(storage_dir, exist_ok=True)
        self.load()

    def add_item(self, label, file_path, vector):
        """
        Adds a single audio sample vector to the tree index.
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

    def add_items_batch(self, items_list):
        """
        Batch add items: list of (label, file_path, vector) tuples.
        """
        for label, path, vec in items_list:
            vec = np.asarray(vec, dtype=np.float32).flatten()
            if len(vec) != self.dimension:
                continue
            norm = np.linalg.norm(vec)
            unit_vec = vec / (norm + 1e-12)
            self.items.append({"id": len(self.items), "label": label, "path": path})
            if self.vectors.shape[0] == 0:
                self.vectors = unit_vec[np.newaxis, :]
            else:
                self.vectors = np.vstack([self.vectors, unit_vec[np.newaxis, :]])
        self._rebuild_tree()
        self.save()

    def _rebuild_tree(self):
        """
        Reconstructs the scipy spatial KDTree over the indexed points.
        """
        if self.vectors.shape[0] > 0:
            self.tree = KDTree(self.vectors)
        else:
            self.tree = None

    def query(self, query_vector, top_k=3):
        """
        Searches the tree for the closest matches using Euclidean distance on unit vectors
        (equivalent to cosine similarity ranking: d^2 = 2 - 2*cos_sim).
        """
        if self.tree is None or len(self.items) == 0:
            return []

        query_vec = np.asarray(query_vector, dtype=np.float32).flatten()
        norm = np.linalg.norm(query_vec)
        unit_query = query_vec / (norm + 1e-12)

        k = min(top_k, len(self.items))
        distances, indices = self.tree.query(unit_query, k=k)

        if k == 1:
            distances = [distances]
            indices = [indices]

        matches = []
        for dist, idx in zip(distances, indices):
            item = self.items[idx]
            # Convert Euclidean distance on unit sphere to cosine similarity: cos_sim = 1 - (dist^2 / 2)
            cos_sim = 1.0 - (float(dist) ** 2) / 2.0
            matches.append({
                "label": item["label"],
                "path": item["path"],
                "distance": float(dist),
                "similarity": float(np.clip(cos_sim, -1.0, 1.0))
            })
        return matches

    def save(self):
        """
        Persists metadata and vectors to disk.
        """
        with open(self.meta_file, 'w', encoding='utf-8') as f:
            json.dump(self.items, f, indent=2)
        np.save(self.vectors_file, self.vectors)

    def load(self):
        """
        Loads metadata and vectors from disk if present.
        """
        if os.path.exists(self.meta_file) and os.path.exists(self.vectors_file):
            try:
                with open(self.meta_file, 'r', encoding='utf-8') as f:
                    self.items = json.load(f)
                self.vectors = np.load(self.vectors_file)
                self._rebuild_tree()
            except Exception as e:
                print(f"Failed to load tree index {self.index_name}: {e}")
