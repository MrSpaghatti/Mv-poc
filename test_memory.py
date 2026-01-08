import unittest
import os
import json
from typing import List, Dict, Any, Optional
from hebbian_memory import HebbianRAG

class MockSearchResult:
    def __init__(self, content, score, metadata):
        self.content = content
        self.score = score
        self.metadata = metadata

class MockMemvid:
    def __init__(self):
        self.storage = []

    def store(self, content: str, metadata: Optional[Dict] = None):
        entry = {
            "content": content,
            "metadata": metadata or {}
        }
        self.storage.append(entry)

    def retrieve(self, query: str) -> List[MockSearchResult]:
        # Simple mock retrieval: returns all items,
        # with a mock score based on basic string matching or just random/fixed
        results = []
        for item in self.storage:
            # Simple score: 0.8 if query in content, else 0.5
            score = 0.8 if query in item['content'] else 0.5
            results.append(MockSearchResult(
                content=item['content'],
                score=score,
                metadata=item['metadata']
            ))
        # Return sorted by score initially (simulating vector search)
        results.sort(key=lambda x: x.score, reverse=True)
        return results

class TestHebbianRAG(unittest.TestCase):
    def setUp(self):
        self.weights_file = "test_weights.json"
        if os.path.exists(self.weights_file):
            os.remove(self.weights_file)

        self.mock_memvid = MockMemvid()
        self.rag = HebbianRAG(self.mock_memvid, weights_file=self.weights_file)

    def tearDown(self):
        if os.path.exists(self.weights_file):
            os.remove(self.weights_file)

    def test_store_and_init_weight(self):
        mem_id = self.rag.store("The sky is blue.")
        self.assertTrue(mem_id)
        self.assertEqual(self.rag.weights[mem_id], 1.0)

        # Verify file creation
        self.assertTrue(os.path.exists(self.weights_file))
        with open(self.weights_file, 'r') as f:
            data = json.load(f)
            self.assertEqual(data[mem_id], 1.0)

    def test_retrieve_ranking(self):
        # Store two items
        id1 = self.rag.store("Python is a snake.")
        id2 = self.rag.store("Python is a language.")

        # Both match "Python", so mock score will be equal (0.8)
        # Weights are 1.0
        results = self.rag.retrieve("Python")
        self.assertEqual(len(results), 2)
        # Order is indeterminate or insertion order if scores equal.
        # Let's verify scores are 0.8 * 1.0 = 0.8
        self.assertEqual(results[0]['adjusted_score'], 0.8)
        self.assertEqual(results[1]['adjusted_score'], 0.8)

        # Reinforce id2 (Language)
        self.rag.reinforce(id2, feedback=1) # 1.0 * 1.1 = 1.1

        # Retrieve again
        results_new = self.rag.retrieve("Python")

        # id2 should now be first because 0.8 * 1.1 > 0.8 * 1.0
        self.assertEqual(results_new[0]['id'], id2)
        self.assertAlmostEqual(results_new[0]['weight'], 1.1)
        self.assertAlmostEqual(results_new[0]['adjusted_score'], 0.88)

        self.assertEqual(results_new[1]['id'], id1)
        self.assertEqual(results_new[1]['weight'], 1.0)
        self.assertEqual(results_new[1]['adjusted_score'], 0.8)

    def test_reinforce_decay(self):
        mem_id = self.rag.store("Temporary fact.")
        self.rag.reinforce(mem_id, feedback=-1)
        self.assertAlmostEqual(self.rag.weights[mem_id], 0.9)

if __name__ == '__main__':
    unittest.main()
