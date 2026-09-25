import os
import json
from collections import defaultdict

class FingerprintIndex:
    """
    Inverted Index Database for Audio Fingerprints (Constellation Hashes).
    Maps Hash -> [(Label, Time_Offset), ...]
    """
    def __init__(self, index_name="fingerprints", storage_dir="indexes"):
        self.storage_path = os.path.join(storage_dir, f"{index_name}.json")
        self.index = defaultdict(list)
        os.makedirs(storage_dir, exist_ok=True)
        self.load()

    def add_hashes(self, label, hashes):
        """
        Adds a list of (hash, time_offset) pairs to the inverted index.
        """
        for h, t in hashes:
            self.index[str(h)].append((label, int(t)))
        self.save()

    def query(self, query_hashes):
        """
        Queries the database with a list of (hash, time_offset) pairs from a short query snippet.
        Uses a Hough Transform (histogram of time offsets) to find the most coherent match.
        """
        if not self.index:
            return None, 0

        # Tally matches: label -> (database_time_offset - query_time_offset) -> count
        # This aligns the peak structure in time.
        tally = defaultdict(lambda: defaultdict(int))
        
        for q_hash, q_t in query_hashes:
            h_str = str(q_hash)
            if h_str in self.index:
                for db_label, db_t in self.index[h_str]:
                    delta_t = db_t - q_t
                    tally[db_label][delta_t] += 1
                    
        if not tally:
            return None, 0
            
        # Find the label and time-offset that has the maximum number of aligned peaks
        best_label = None
        max_aligned_peaks = 0
        
        for label, offsets in tally.items():
            max_for_label = max(offsets.values()) if offsets else 0
            if max_for_label > max_aligned_peaks:
                max_aligned_peaks = max_for_label
                best_label = label
                
        return best_label, max_aligned_peaks

    def save(self):
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(self.index, f)

    def load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.index = defaultdict(list, data)
            except Exception as e:
                print(f"Failed to load fingerprint index: {e}")
