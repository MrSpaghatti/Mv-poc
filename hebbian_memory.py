import json
import os
import uuid
from typing import List, Dict, Any, Optional

class HebbianRAG:
    """
    A memory module that wraps a memvid-like client with a usage-based weighting system (Hebbian Learning).

    The 'memvid_client' is expected to follow this interface:
    - store(content: str, metadata: dict) -> None
    - retrieve(query: str) -> List[SearchResult]

    where SearchResult is an object or dict with attributes:
    - content: str
    - score: float
    - metadata: dict (containing 'id')
    """

    def __init__(self, memvid_client, weights_file: str = "weights.json"):
        """
        Initialize the HebbianRAG system.

        Args:
            memvid_client: An instance of a wrapper/client for memvid storage and retrieval.
            weights_file: Path to the JSON file where synaptic weights are stored.
        """
        self.memvid = memvid_client
        self.weights_file = weights_file
        self.weights: Dict[str, float] = {}
        self._load_weights()

    def _load_weights(self):
        """Load weights from the JSON file."""
        if os.path.exists(self.weights_file):
            try:
                with open(self.weights_file, 'r') as f:
                    self.weights = json.load(f)
            except json.JSONDecodeError:
                self.weights = {}
        else:
            self.weights = {}

    def _save_weights(self):
        """Save weights to the JSON file."""
        with open(self.weights_file, 'w') as f:
            json.dump(self.weights, f, indent=4)

    def store(self, content: str) -> str:
        """
        Save content to memvid and initialize its weight to 1.0.

        Returns:
            The generated memory_id.
        """
        memory_id = str(uuid.uuid4())
        # Store in memvid with the generated ID in metadata
        # Assuming the client supports passing metadata
        self.memvid.store(content, metadata={"id": memory_id})

        # Initialize weight
        self.weights[memory_id] = 1.0
        self._save_weights()

        return memory_id

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Query memvid for top results and re-rank based on weights.

        Returns:
            A list of re-ranked results. Each result is a dictionary.
        """
        # Query memvid (assuming it returns top 10 by default or we can specify)
        # We assume the client's retrieve method returns objects that have .score and .metadata
        raw_results = self.memvid.retrieve(query)

        processed_results = []

        for res in raw_results:
            # Handle if res is dict or object
            if isinstance(res, dict):
                metadata = res.get('metadata', {})
                score = res.get('score', 0.0)
                content = res.get('content', "")
            else:
                metadata = getattr(res, 'metadata', {})
                score = getattr(res, 'score', 0.0)
                content = getattr(res, 'content', "") # or .text

            mem_id = metadata.get('id')

            # If we don't have an ID (legacy data?), treat weight as 1.0
            weight = self.weights.get(mem_id, 1.0) if mem_id else 1.0

            # Re-rank: simple multiplication or another formula?
            # Requirement: "Higher weight = higher position"
            adjusted_score = score * weight

            processed_results.append({
                "id": mem_id,
                "content": content,
                "original_score": score,
                "weight": weight,
                "adjusted_score": adjusted_score,
                "metadata": metadata
            })

        # Sort by adjusted_score descending
        processed_results.sort(key=lambda x: x['adjusted_score'], reverse=True)

        return processed_results

    def reinforce(self, memory_id: str, feedback: int):
        """
        Reinforce or decay the weight of a memory.

        Args:
            memory_id: The ID of the memory to update.
            feedback: +1 (positive) or -1 (negative).
        """
        if memory_id not in self.weights:
            # If it's a valid ID but not in weights (e.g. loaded from existing store), init to 1.0
            self.weights[memory_id] = 1.0

        current = self.weights[memory_id]

        # --- FIX: LINEAR GROWTH WITH CAP ---
        MAX_WEIGHT = 3.0
        MIN_WEIGHT = 0.1

        if feedback > 0:
            # Add 0.2, cap at 3.0
            new_weight = min(current + 0.2, MAX_WEIGHT)
            print(f" >> Promoting Memory {memory_id[:8]}... (Weight: {current:.2f} -> {new_weight:.2f})")
        elif feedback < 0:
            # Subtract 0.2, floor at 0.1
            new_weight = max(current - 0.2, MIN_WEIGHT)
            print(f" >> Burying Memory {memory_id[:8]}... (Weight: {current:.2f} -> {new_weight:.2f})")
        else:
            new_weight = current

        self.weights[memory_id] = new_weight
        self._save_weights()
