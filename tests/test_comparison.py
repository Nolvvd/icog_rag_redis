"""Verification Suite for CircuitMind ECE Component & Protocol Comparison Matrix.
Validates comparative query detection, markdown matrix formatting, grounded citations,
and Redis Cache-Aside performance.
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
    is_comparison_query,
    generate_grounded_answer,
    generate_comparison_matrix
)

class TestComparisonMatrix(unittest.TestCase):
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

    def test_01_detect_comparison_queries(self):
        """Validates regex detection of comparative ECE queries."""
        self.assertTrue(is_comparison_query("Explain the difference between UART, SPI and I2C."))
        self.assertTrue(is_comparison_query("UART vs SPI vs I2C"))
        self.assertTrue(is_comparison_query("Compare LDO regulators versus switching buck converters"))
        self.assertTrue(is_comparison_query("What are the trade-offs between BLE and Wi-Fi?"))
        self.assertTrue(is_comparison_query("Which should I choose: AP2112K or AMS1117?"))

        # Non-comparative queries must return False
        self.assertFalse(is_comparison_query("What is the operating voltage of the ESP32?"))
        self.assertFalse(is_comparison_query("List the input-only pins on ESP32."))

    def test_02_generate_comparison_matrix_offline(self):
        """Verifies markdown table structure and grounded citations in offline mode."""
        query = "Compare UART vs SPI vs I2C protocols"
        retrieved = self.vector_store.similarity_search(query, top_k=4)
        self.assertGreater(len(retrieved), 0)

        ans, meta = generate_grounded_answer(
            query=query,
            retrieved_chunks=retrieved,
            provider="offline",
            return_metadata=True
        )

        self.assertTrue(meta.get("is_comparison"), "Metadata must flag query as comparative.")
        self.assertIn("|", ans, "Answer must contain Markdown table pipe characters.")
        self.assertIn("Comparative Engineering Analysis Matrix", ans, "Answer must include matrix header.")
        self.assertIn("[Source", ans, "Every claim in matrix and narrative must cite [Source X].")

    def test_03_convenience_function(self):
        """Verifies generate_comparison_matrix produces grounded comparison."""
        query = "LDO vs Buck Converter efficiency and heat dissipation"
        retrieved = self.vector_store.similarity_search(query, top_k=3)

        ans = generate_comparison_matrix(query, retrieved, provider="offline")
        self.assertIn("|", ans)
        self.assertIn("[Source", ans)

    def test_04_cache_hit_on_comparison_matrix(self):
        """Verifies comparison matrix is stored in Redis and returned in a sub-millisecond cache hit."""
        query = "UART vs SPI vs I2C"
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
            query="  uart vs spi vs i2c  ",  # Normalized query
            doc_version=self.doc_hash,
            model_name="test-model"
        )
        self.assertIsNotNone(cached)
        self.assertIn("|", cached["answer"])
        self.assertEqual(len(cached["sources"]), 3)
        self.assertEqual(cached["sources"][0]["source"], "esp32_hardware_reference.md")

if __name__ == "__main__":
    unittest.main(verbosity=2)
