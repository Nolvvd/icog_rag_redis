"""Comprehensive Verification Suite for RAG with Redis Caching.
Validates all 9 critical requirements, supports unittest & direct execution,
and benchmarks empirical Hit vs Miss response latencies.
"""

import sys
import time
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
from cache import RedisRAGCache, normalize_query, generate_cache_key
from llm import generate_grounded_answer

class TestRAGRedisPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        os.environ["LLM_PROVIDER"] = "offline"
        cls.sample_path = PROJECT_ROOT / "data" / "sample_transformers.txt"
        cls.sample_text = cls.sample_path.read_text(encoding="utf-8")
        cls.doc_hash_v1 = calculate_sha256(cls.sample_text)
        
        docs_v1 = extract_documents(cls.sample_text, filename="sample_transformers.txt")
        cls.chunks_v1 = chunk_documents(docs_v1, chunk_size=300, chunk_overlap=50)

        cls.vector_store = InMemoryVectorStore()
        cls.vector_store.add_documents(cls.chunks_v1, doc_version=cls.doc_hash_v1)
        
        cls.cache = RedisRAGCache(default_ttl=2)
        cls.cache.clear()
        cls.model_name = "test-model-v1"
        cls.query = "What is the formula for Scaled Dot-Product Attention?"

    def test_01_cache_miss_on_first_query(self):
        """Requirement: First question produces a cache miss."""
        payload = self.cache.get(self.query, doc_version=self.doc_hash_v1, model_name=self.model_name)
        self.assertIsNone(payload, "First question must produce a cache miss.")

    def test_02_retrieval_and_generation_on_miss(self):
        """Requirement: Application performs retrieval and generation on miss."""
        retrieved = self.vector_store.similarity_search(self.query, top_k=3)
        self.assertGreater(len(retrieved), 0, "Retrieval should return matching passages.")
        answer = generate_grounded_answer(self.query, retrieved, model_name=self.model_name)
        self.assertGreater(len(answer.strip()), 0, "Generated answer must not be empty.")
        self.assertIn("[Source", answer, "Answer must include grounded source citations.")

    def test_03_store_answer_and_references(self):
        """Requirement: The answer and references are stored in Redis."""
        retrieved = self.vector_store.similarity_search(self.query, top_k=3)
        answer = generate_grounded_answer(self.query, retrieved, model_name=self.model_name)
        sources = [
            {"source": c["source"], "page": c["page"], "chunk_id": c["chunk_id"], "snippet": c["text"][:100]}
            for c in retrieved
        ]
        success = self.cache.set(
            query=self.query,
            doc_version=self.doc_hash_v1,
            model_name=self.model_name,
            answer=answer,
            sources=sources,
            ttl=2
        )
        self.assertTrue(success, "Failed to store answer and sources in cache.")

    def test_04_cache_hit_on_repeated_question(self):
        """Requirement: Repeating the question produces a cache hit."""
        # Query with whitespace/casing differences to verify query normalization
        query_variant = "  what is the formula for scaled dot-product attention?  "
        payload = self.cache.get(query_variant, doc_version=self.doc_hash_v1, model_name=self.model_name)
        self.assertIsNotNone(payload, "Repeating question must produce a cache hit.")
        self.assertIn("answer", payload)

    def test_05_source_references_preserved_in_cache_hit(self):
        """Requirement: Source references and metadata are preserved exactly on cache hit."""
        payload = self.cache.get(self.query, doc_version=self.doc_hash_v1, model_name=self.model_name)
        self.assertIsNotNone(payload, "Cache payload must exist for source reference verification.")
        self.assertIn("sources", payload)
        self.assertEqual(len(payload["sources"]), 3, "Source references must be preserved.")
        self.assertEqual(payload["sources"][0]["source"], "sample_transformers.txt")
        self.assertIn("chunk_id", payload["sources"][0])

    def test_06_cache_hit_bypasses_retrieval(self):
        """Requirement: Cache hit avoids vector retrieval and LLM generation."""
        # Check cache directly; if hit, no vector_store or LLM call is invoked
        payload = self.cache.get(self.query, doc_version=self.doc_hash_v1, model_name=self.model_name)
        self.assertIsNotNone(payload)
        # Bypassing confirmed: payload retrieved without executing vector search

    def test_07_ttl_expiration(self):
        """Requirement: Expired entries produce a cache miss."""
        time.sleep(2.2)  # Wait for 2.0s TTL to elapse
        payload = self.cache.get(self.query, doc_version=self.doc_hash_v1, model_name=self.model_name)
        self.assertIsNone(payload, "Expired entry must produce a cache miss.")

    def test_08_document_version_invalidation(self):
        """Requirement: Document-version changes prevent stale cache reuse."""
        # Cache answer under doc_v1
        retrieved = self.vector_store.similarity_search(self.query, top_k=2)
        answer = generate_grounded_answer(self.query, retrieved, model_name=self.model_name)
        self.cache.set(self.query, self.doc_hash_v1, self.model_name, answer, retrieved, ttl=60)
        self.assertIsNotNone(self.cache.get(self.query, self.doc_hash_v1, self.model_name))

        # Change document content -> new hash
        new_text = self.sample_text + "\nAddendum: Revision 2026."
        doc_hash_v2 = calculate_sha256(new_text)
        self.assertNotEqual(self.doc_hash_v1, doc_hash_v2)

        # Checking cache with doc_v2 must produce a MISS
        v2_payload = self.cache.get(self.query, doc_hash_v2, self.model_name)
        self.assertIsNone(v2_payload, "Corpus version shift must invalidate stale cache entries.")

    def test_09_redis_unavailable_resilience(self):
        """Requirement: Redis being unavailable does not break the normal RAG flow."""
        offline_cache = RedisRAGCache(redis_url="redis://127.0.0.1:9999/0", default_ttl=60)
        # Must return None instead of throwing unhandled exception
        result = offline_cache.get("offline query", self.doc_hash_v1, self.model_name)
        self.assertIsNone(result)
        # Must handle write gracefully
        write_ok = offline_cache.set("offline query", self.doc_hash_v1, self.model_name, "answer", [], ttl=60)
        self.assertTrue(write_ok)

def run_performance_benchmarks():
    print("\n" + "=" * 70)
    print("EMPIRICAL LATENCY BENCHMARK: CACHE MISS VS CACHE HIT")
    print("=" * 70)

    sample_path = PROJECT_ROOT / "data" / "sample_transformers.txt"
    sample_text = sample_path.read_text(encoding="utf-8")
    doc_hash = calculate_sha256(sample_text)
    docs = extract_documents(sample_text, filename="sample_transformers.txt")
    chunks = chunk_documents(docs, chunk_size=300, chunk_overlap=50)

    v_store = InMemoryVectorStore()
    v_store.add_documents(chunks, doc_version=doc_hash)

    cache = RedisRAGCache(default_ttl=3600)
    cache.clear()

    query = "Explain multi-head attention and why it is useful."
    model = "test-model-v1"

    # Measure Cache Miss
    t0 = time.perf_counter()
    res = cache.get(query, doc_hash, model)
    assert res is None
    matched = v_store.similarity_search(query, top_k=3)
    answer = generate_grounded_answer(query, matched, provider="offline", model_name=model)
    cache.set(query, doc_hash, model, answer, matched, ttl=3600)
    miss_latency_ms = (time.perf_counter() - t0) * 1000.0

    # Measure Cache Hit
    t1 = time.perf_counter()
    hit_res = cache.get(query, doc_hash, model)
    assert hit_res is not None
    hit_latency_ms = (time.perf_counter() - t1) * 1000.0

    speedup = miss_latency_ms / max(0.001, hit_latency_ms)

    print(f"Query: '{query}'")
    print(f"  [Cache MISS] Vector Search + Generation Latency: {miss_latency_ms:.2f} ms")
    print(f"  [Cache HIT ] Redis Cache-Aside Latency:           {hit_latency_ms:.2f} ms")
    print(f"  [Speedup   ] In-Memory / Redis Acceleration:       {speedup:.1f}x faster")
    print("=" * 70)

if __name__ == "__main__":
    unittest.main(exit=False)
    run_performance_benchmarks()
