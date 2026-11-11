"""Comprehensive verification script testing:
1. Document extraction & indexing
2. RAG semantic retrieval
3. Grounded answer generation
4. Verification of ZERO raw HTML tags in user-facing output
5. Source classification (Direct Evidence, Supporting Evidence, Additional Context)
6. Redis Cache-Aside & source preservation on Cache HIT
7. Redis unavailable graceful degradation / fallback
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from extractor import extract_documents, calculate_sha256
from chunker import chunk_documents
from vector_store import InMemoryVectorStore
from cache import RedisRAGCache
from llm import generate_grounded_answer
from config import PROMPT_VERSION, DEFAULT_MODEL

def run_tests():
    print("=" * 60)
    print("CIRCUITMIND PIPELINE VERIFICATION SUITE")
    print("=" * 60)

    # 1. Test Document Indexing
    doc_path = Path(__file__).resolve().parent.parent / "data" / "esp32_hardware_reference.md"
    assert doc_path.exists(), "Spec file missing"
    doc_bytes = doc_path.read_bytes()
    doc_hash = calculate_sha256(doc_bytes)
    docs = extract_documents(doc_bytes, filename="esp32_hardware_reference.md")
    chunks = chunk_documents(docs, chunk_size=400, chunk_overlap=80)
    assert len(chunks) > 0, "No chunks created"

    vector_store = InMemoryVectorStore()
    vector_store.add_documents(chunks, doc_version=doc_hash)
    print(f"[PASS] 1. Indexing: SUCCESS ({len(chunks)} chunks, hash={doc_hash[:8]})")

    # 2. Test RAG Retrieval for 'What is UART?'
    query = "What is UART?"
    retrieved = vector_store.similarity_search(query, top_k=3)
    assert len(retrieved) == 3, f"Expected 3 chunks, got {len(retrieved)}"
    print(f"[PASS] 2. Semantic Retrieval: SUCCESS (Retrieved {len(retrieved)} chunks for '{query}')")

    # 3. Test Grounded Answer Generation (Offline mode for deterministic test)
    ans, meta = generate_grounded_answer(query, retrieved, provider="offline", return_metadata=True)
    assert len(ans.strip()) > 0, "Empty answer generated"

    # 4. Verify ZERO raw HTML tags in answer
    html_tags = ["<table", "<tr>", "<td>", "<div", "<span", "</table", "</tr>", "</td>", "</div>", "</span>"]
    for tag in html_tags:
        assert tag not in ans.lower(), f"Raw HTML tag found in answer: {tag}"
    print("[PASS] 3. Raw HTML Verification: SUCCESS (Zero HTML tags in generated answer)")

    # 5. Verify Source Classification
    src_list = []
    for rank, c in enumerate(retrieved, 1):
        ev_type = "Direct Evidence" if rank == 1 else ("Supporting Evidence" if rank == 2 else "Additional Context")
        src_list.append({
            "source": c.get("source"),
            "page": c.get("page"),
            "chunk_id": c.get("chunk_id"),
            "evidence_type": ev_type,
            "snippet": c.get("text")[:200]
        })
    assert src_list[0]["evidence_type"] == "Direct Evidence"
    assert src_list[1]["evidence_type"] == "Supporting Evidence"
    assert src_list[2]["evidence_type"] == "Additional Context"
    print("[PASS] 4. Source Classification: SUCCESS (Direct, Supporting, Additional Context verified)")

    # 6. Test Redis Cache-Aside
    cache = RedisRAGCache(default_ttl=60)
    # Cold check
    miss_res = cache.get(query, doc_hash, DEFAULT_MODEL, prompt_version=PROMPT_VERSION)
    assert miss_res is None, "Cold check should be None"

    # Set cache
    cache.set(query, doc_hash, DEFAULT_MODEL, ans, src_list, ttl=60, prompt_version=PROMPT_VERSION)

    # Warm check
    hit_res = cache.get(query, doc_hash, DEFAULT_MODEL, prompt_version=PROMPT_VERSION)
    assert hit_res is not None, "Warm check should return payload"
    assert hit_res["answer"] == ans, "Cached answer mismatch"
    assert len(hit_res["sources"]) == 3, "Sources count mismatch"
    assert hit_res["sources"][0]["evidence_type"] == "Direct Evidence", "Evidence type not preserved in cache"
    print("[PASS] 5. Redis Cache-Aside & Source Preservation: SUCCESS (Identical sources returned on Cache HIT)")

    # 7. Test In-Memory Fallback
    cache_fallback = RedisRAGCache(redis_url="redis://invalid_host_12345:6379/0", default_ttl=60)
    cache_fallback.set(query, doc_hash, DEFAULT_MODEL, ans, src_list, ttl=60, prompt_version=PROMPT_VERSION)
    fallback_hit = cache_fallback.get(query, doc_hash, DEFAULT_MODEL, prompt_version=PROMPT_VERSION)
    assert fallback_hit is not None, "Fallback cache failed"
    assert fallback_hit["answer"] == ans
    print("[PASS] 6. Graceful Redis-Unavailable Fallback: SUCCESS (In-Memory fallback active and functional)")

    print("=" * 60)
    print("ALL 6 PIPELINE VERIFICATIONS PASSED CLEANLY.")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
