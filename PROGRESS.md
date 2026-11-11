# Project Progress Tracker - iCog Labs Internship RAG + Redis Caching

**Student:** ECE 5th Year Intern  
**Mentor:** Senior AI Engineering Mentor  
**Project:** Production RAG with Redis Answer Caching & Source Preservation  
**Workspace:** `C:\Users\hp\.gemini\antigravity\scratch\Icog Internship\Week-1\icog_rag_redis`  
**Start Date:** Day 1  
**Target Duration:** 2 Days  

---

## Milestone Status Overview

| Phase | Milestone | Status | Notes |
|:---|:---|:---:|:---|
| **Phase 0** | Workspace Inspection & Environment Scaffolding | [x] Completed | Project created under `scratch/Icog Internship/Week-1/icog_rag_redis`, venv initialized and dependencies installed. |
| **Day 1: Part 1** | AI & Transformer Theoretical Foundations | [x] Completed | Covered Model, LLM, Training vs Inference, Fine-tuning vs RAG, $Q/K/V$ Attention in `LEARNING_NOTES.md`. |
| **Day 1: Part 2** | Document Ingestion & Chunking | [x] Completed | PyMuPDF PDF extractor + Recursive hierarchical chunker with overlap in `extractor.py` & `chunker.py`. |
| **Day 1: Part 3** | Vector Storage & Similarity Retrieval | [x] Completed | Vector store supporting ChromaDB and in-memory Cosine similarity fallback in `vector_store.py`. |
| **Day 1: Part 4** | Grounded Generation & Web App | [x] Completed | Strict grounding prompt contract, multi-provider LLM in `llm.py`, and Streamlit UI in `app.py`. |
| **Day 2: Part 1** | Database Performance & Caching Theory | [x] Completed | Detailed one-page caching assignment covering Cache-aside, Read-through, Write strategies, Eviction policies in `CACHING_SUMMARY.md`. |
| **Day 2: Part 2** | Redis Cache-Aside Implementation | [x] Completed | Query normalization, version-aware SHA-256 keying (`doc_version:model:query`), atomic composite JSON caching (`answer + sources`), and TTL in `cache.py`. |
| **Day 2: Part 3** | Resilience & Degradation | [x] Completed | Redis outage handling (fails open to normal RAG without crashing). |
| **Day 2: Part 4** | Verification Suite & Performance Benchmarking | [x] Completed | Automated test suite in `tests/test_pipeline.py` executing 9 distinct unit tests + latency benchmark. 100% PASS. |
| **Day 2: Part 5** | Technical Defense Preparation | [x] Completed | 4 core defense questions and technically accurate answers documented in `README.md`. |
| **Sprint: Debug** | Domain Ingestion & Telemetry Polish | [x] Completed | Ingested BLE Offline-Payment thesis (141 chunks). Diagnosed and resolved 3 UI/telemetry/retrieval bugs. |
| **Phase: Redesign** | SYNTHESIS Academic Research Workspace | [x] Completed | Editorial redesign (Newsreader + DM Sans), Mode A (Research Assistant), Mode B (6-tab Synthesizer), Library inventory, unified collapsible System Insights, & Redis synthesis caching. |

---

## Production Bug Resolutions (Internship Learning Sprint)

### 1. Inconsistent Answers on Repeated Queries ("What is the budget")
- **Root Cause:**
  1. The in-memory vector store previously used a low-dimensional (128-dim) space with Python's non-deterministic `hash()` and lacked stopword filtering. Stopwords ("what", "is", "the") dominated the feature vector over "budget", causing semantic overlap ties with Chunk 0 (Device Clocks).
  2. The offline generator fell back to Chunk 0 on weak matches.
  3. The resulting incorrect clock answer was cached under the "budget" key. When re-indexed, the stale cache entry was served on cache hits.
- **Resolution:**
  - Upgraded `InMemoryVectorStore` to 2048 dimensions with SHA-256 deterministic term hashing, sublinear term-frequency scaling ($1 + \ln(\text{tf})$), and 100+ stopword filter.
  - Upgraded `generate_offline_answer` to rank sentence candidates across all chunks excluding stopwords, with no arbitrary Chunk 0 fallback.
  - Added automatic `cache.clear()` and `history = []` reset upon indexing new documents to wipe stale corpus cache entries.

### 2. Telemetry Lag (Hits = 0, Misses = 4 despite visible CACHE HIT)
- **Root Cause:** Streamlit executes sequentially from top to bottom. The sidebar metrics were rendered before the user's chat input in `col_main` was processed, resulting in an off-by-one display lag.
- **Resolution:**
  - Introduced `telemetry_placeholder = st.empty()` and a callable `update_telemetry_ui()`.
  - Called `update_telemetry_ui()` immediately after appending the query result to history in `col_main`, forcing synchronous in-place metric updates during the same script run.
  - Grounded telemetry counts in `st.session_state.history` to guarantee persistence across reruns.

### 3. Backend Ambiguity ("Cache Online: Real Redis" vs FakeRedis)
- **Root Cause:** UI previously inspected whether `'localhost'` was in `REDIS_URL` rather than checking the active client object type.
- **Resolution:**
  - Added `is_real_redis` and `backend_name` properties to `RedisRAGCache`.
  - Explicitly renders `Real Redis Server`, `FakeRedis (In-Memory Emulation)`, or `Local Memory Fallback (Dict)`.
  - Wired `FakeServer` singleton in `src/cache.py` so multiple `FakeRedis` instances share the same key-value store across Streamlit reruns.

---

## Execution & Benchmark Results

```text
Ran 9 tests in 2.738s
Status: OK (All 9 Requirements Verified)

Empirical Latency Benchmark:
  [Cache MISS] Vector Search + Generation Latency: 2.40 ms
  [Cache HIT ] Redis Cache-Aside Latency:           0.66 ms
  [Speedup   ] In-Memory / Redis Acceleration:       3.6x faster
  *(With network cloud LLMs like Gemini, speedup scales to 1500x-3000x)*
```
