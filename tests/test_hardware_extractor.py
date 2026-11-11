"""Verification Suite for CircuitMind Hardware Pinout & Register Map Extractor.
Validates hardware intent detection, pinout/register table formatting,
grounded citations, and Redis Cache-Aside performance.
"""

import sys
import unittest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from extractor import extract_documents, calculate_sha256
from chunker import chunk_documents
from vector_store import InMemoryVectorStore
from cache import RedisRAGCache
from llm import (
    is_hardware_spec_query,
    generate_grounded_answer,
    generate_hardware_spec_matrix
)

class TestHardwareExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        os.environ["LLM_PROVIDER"] = "offline"

        # Load the ECE Hardware Reference document
        cls.doc_path = PROJECT_ROOT / "data" / "esp32_hardware_reference.md"
        cls.doc_text = cls.doc_path.read_text(encoding="utf-8")
        cls.doc_hash = calculate_sha256(cls.doc_text)

        docs = extract_documents(cls.doc_text, filename="esp32_hardware_reference.md")
        cls.chunks = chunk_documents(docs, chunk_size=400, chunk_overlap=80)

        cls.vector_store = InMemoryVectorStore()
        cls.vector_store.add_documents(cls.chunks, doc_version=cls.doc_hash)

        cls.cache = RedisRAGCache(default_ttl=5)
        cls.cache.clear()

    def test_01_detect_hardware_spec_queries(self):
        """Validates detection of pinout and register queries."""
        self.assertTrue(is_hardware_spec_query("What are the input-only pins on the ESP32?"))
        self.assertTrue(is_hardware_spec_query("Show me GPIO34 pinout and alternate functions"))
        self.assertTrue(is_hardware_spec_query("What registers control GPIO output?"))
        self.assertTrue(is_hardware_spec_query("List strapping pins and reset values"))
        self.assertTrue(is_hardware_spec_query("What are the ADC channels for GPIO pins?"))

        # Non-hardware queries must return False
        self.assertFalse(is_hardware_spec_query("What is self-attention mechanism?"))
        self.assertFalse(is_hardware_spec_query("How do you train an acoustic neural model?"))

    def test_02_extract_pinout_table_offline(self):
        """Verifies pinout extraction produces structured Markdown table with citations."""
        query = "What are the input-only pins and strapping pins on the ESP32?"
        retrieved = self.vector_store.similarity_search(query, top_k=4)
        self.assertGreater(len(retrieved), 0)

        ans, meta = generate_grounded_answer(
            query=query,
            retrieved_chunks=retrieved,
            provider="offline",
            return_metadata=True
        )

        self.assertTrue(meta.get("is_hardware_spec"), "Metadata must flag query as hardware spec.")
        self.assertIn("|", ans, "Answer must contain Markdown table pipe characters.")
        self.assertIn("Hardware Pinout & Register Specification Table", ans)
        self.assertIn("[Source", ans, "All extracted claims must cite [Source X].")

    def test_03_extract_register_map_offline(self):
        """Verifies register map query extracts registers and offsets."""
        query = "What are the GPIO output registers and offsets like GPIO_OUT_REG?"
        retrieved = self.vector_store.similarity_search(query, top_k=4)

        ans = generate_hardware_spec_matrix(query, retrieved, provider="offline")
        self.assertIn("|", ans)
        self.assertIn("GPIO_OUT_REG", ans)
        self.assertIn("[Source", ans)

    def test_04_cache_hit_on_hardware_query(self):
        """Verifies pinout & register tables are stored and retrieved from Redis."""
        query = "ESP32 strapping pins boot mode"
        retrieved = self.vector_store.similarity_search(query, top_k=3)
        ans, meta = generate_grounded_answer(query, retrieved, provider="offline", return_metadata=True)

        sources = [
            {"source": c["source"], "page": c["page"], "chunk_id": c["chunk_id"], "snippet": c["text"][:100]}
            for c in retrieved
        ]

        # Store in cache
        stored = self.cache.set(
            query=query,
            doc_version=self.doc_hash,
            model_name="test-model",
            answer=ans,
            sources=sources,
            ttl=5
        )
        self.assertTrue(stored)

        # Retrieve from cache
        cached = self.cache.get(
            query="  esp32 strapping pins boot mode  ",
            doc_version=self.doc_hash,
            model_name="test-model"
        )
        self.assertIsNotNone(cached)
        self.assertIn("|", cached["answer"])
        self.assertEqual(len(cached["sources"]), 3)
        self.assertEqual(cached["sources"][0]["source"], "esp32_hardware_reference.md")

if __name__ == "__main__":
    unittest.main(verbosity=2)
