"""Verification tests for Mode B (Research Synthesizer) & Synthesis Caching.
Validates 6-section academic extraction, source preservation, and Redis synthesis caching.
"""

import os
import sys
import unittest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ["LLM_PROVIDER"] = "offline"

from extractor import extract_documents, calculate_sha256
from chunker import chunk_documents
from vector_store import InMemoryVectorStore
from cache import RedisRAGCache
from llm import generate_research_synthesis, parse_synthesis_sections

class TestResearchSynthesizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_path = PROJECT_ROOT / "data" / "sample_transformers.txt"
        cls.sample_text = cls.sample_path.read_text(encoding="utf-8")
        cls.doc_hash = calculate_sha256(cls.sample_text)
        
        docs = extract_documents(cls.sample_text, filename="sample_transformers.txt")
        cls.chunks = chunk_documents(docs, chunk_size=300, chunk_overlap=50)

        cls.vector_store = InMemoryVectorStore()
        cls.vector_store.add_documents(cls.chunks, doc_version=cls.doc_hash)

        cls.cache = RedisRAGCache(default_ttl=3600)
        cls.cache.clear()

    def test_01_synthesis_sections_generation(self):
        """Verifies that synthesis generates all 6 mandatory academic sections."""
        res = generate_research_synthesis(self.chunks, provider="offline")
        self.assertIn("sections", res)
        sections = res["sections"]

        expected_keys = [
            "literature_synthesis",
            "paper_comparison",
            "common_findings",
            "contradictions_differences",
            "research_gaps",
            "potential_directions"
        ]
        for key in expected_keys:
            self.assertIn(key, sections, f"Section '{key}' must be present in synthesis output.")
            self.assertTrue(len(sections[key].strip()) > 0, f"Section '{key}' must not be empty.")

    def test_02_source_attribution_in_synthesis(self):
        """Verifies that synthesis output contains grounded [Source X] citations."""
        res = generate_research_synthesis(self.chunks, provider="offline")
        raw = res["raw_text"]
        self.assertIn("[Source", raw, "Synthesis must contain grounded [Source X] citations.")
        self.assertTrue(len(res["sources"]) > 0, "Source passages metadata must be preserved.")

    def test_03_synthesis_redis_caching(self):
        """Verifies Cache-Aside storage and retrieval of literature synthesis in Redis."""
        scope_hash = "scope_test_1234"
        model_name = "test-model"

        # 1. First check: Miss
        cached = self.cache.get_synthesis(self.doc_hash, model_name, scope_hash)
        self.assertIsNone(cached, "Initial synthesis query must produce a cache miss.")

        # 2. Generate and Cache
        res = generate_research_synthesis(self.chunks, provider="offline")
        ok = self.cache.set_synthesis(self.doc_hash, model_name, scope_hash, res, ttl=60)
        self.assertTrue(ok, "Failed to cache synthesis payload in Redis.")

        # 3. Second check: Hit
        hit = self.cache.get_synthesis(self.doc_hash, model_name, scope_hash)
        self.assertIsNotNone(hit, "Second synthesis query must produce a Redis cache hit.")
        self.assertEqual(hit["provider"], res["provider"])
        self.assertIn("sections", hit)
        self.assertEqual(len(hit["sections"]), 6)

if __name__ == "__main__":
    unittest.main()
