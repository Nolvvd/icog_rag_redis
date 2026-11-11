# System Architecture Specification

## Production RAG Application with Redis Cache-Aside & Source Preservation

### High-Level Architectural Flow

```
                                  +---------------------+
                                  |    User Question    |
                                  +---------------------+
                                             |
                                             v
                           +-----------------------------------+
                           | 1. Query Normalization & Hash     |
                           |    key = sha256(q + doc_v + model)|
                           +-----------------------------------+
                                             |
                                             v
                                  +---------------------+
                                  |   2. Check Redis    |
                                  |     (Cache-Aside)   |
                                  +---------------------+
                                        /       \
                        [Cache HIT]    /         \    [Cache MISS / Fallback]
                                      v           v
           +-----------------------------+     +-------------------------------+
           | Return Stored JSON Payload  |     | 3. Dense Vector Retrieval     |
           | {answer, sources, metadata} |     |    (ChromaDB Cosine Sim)      |
           +-----------------------------+     +-------------------------------+
                          |                                   |
                          |                                   v
                          |                    +-------------------------------+
                          |                    | 4. Context Grounded Prompt    |
                          |                    +-------------------------------+
                          |                                   |
                          |                                   v
                          |                    +-------------------------------+
                          |                    | 5. LLM Answer Generation      |
                          |                    |    (Gemini / OpenAI / Fallback)
                          |                    +-------------------------------+
                          |                                   |
                          |                                   v
                          |                    +-------------------------------+
                          |                    | 6. Save Answer + Sources      |
                          |                    |    in Redis (SETEX with TTL)  |
                          |                    +-------------------------------+
                          |                                   |
                          +-----------------> + <-------------+
                                              |
                                              v
                              +-------------------------------+
                              | 7. Return to Streamlit UI     |
                              |    (Latency, HIT/MISS Badge)  |
                              +-------------------------------+
```

---

### Component Specifications

#### 1. Document Ingestion (`extractor.py` & `chunker.py`)
* **Extractor:** Uses `PyMuPDF` (`fitz`) to extract page-by-page text from uploaded PDFs, preserving structural page metadata: `{"page": int, "text": str, "doc_hash": str}`.
* **Chunker:** Implements recursive character chunking with configurable `chunk_size` (default: 500 characters) and `chunk_overlap` (default: 100 characters). Each chunk retains its source metadata: `{"doc_name": str, "page_num": int, "chunk_id": int, "chunk_hash": str}`.

#### 2. Vector Database (`vector_store.py`)
* **Engine:** ChromaDB in persistent/embedded mode, with an in-memory dual-fallback vector index using Cosine Similarity.
* **Metric:** Cosine distance:
  $$D_C(\vec{u}, \vec{v}) = 1 - \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}$$
* **Query:** Computes embedding for the normalized user query, fetches Top-$k$ chunks (default: $k=3$) and formats them into grounded context blocks.

#### 3. LLM Generation & Grounding (`llm.py`)
* **Strict Grounding Contract:** The LLM prompt explicitly instructs the model:
  > "You are an AI assistant grounded strictly in the provided context. Answer the question using ONLY the provided sources. If the answer cannot be determined from the context, state 'Information not found in the provided documents.' Include citations referring to [Source X]."
* **Multi-Provider Engine:**
  - Primary: Google Gemini API / OpenAI API (reads securely from environment or `.env`).
  - Fallback: Local deterministic extractive answer generator if no API key is supplied, ensuring offline execution and automated CI/CD testing.

#### 4. Redis Caching Engine (`cache.py`)
* **Pattern:** **Cache-Aside** (Lazy loading). The application orchestrates all reads and writes directly.
* **Key Construction (Collision-Resistant & Version-Aware):**
  $$\text{Cache Key} = \text{"rag:cache:"} + \text{SHA256}(\text{doc\_version} + \text{":"} + \text{model\_name} + \text{":"} + \text{normalized\_query})$$
  - `doc_version`: MD5/SHA256 hash of the currently ingested document corpus. If documents change, the `doc_version` shifts, automatically invalidating stale cache keys.
  - `model_name`: Ensures switching models (e.g., from `gemini-1.5` to `gpt-4o`) does not return answers from the prior model.
  - `normalized_query`: Trimmed, lowercased, and whitespace-collapsed user prompt.
* **Payload Structure:**
  ```json
  {
    "answer": "The Transformer uses multi-head attention...",
    "sources": [
      {
        "doc_name": "attention.pdf",
        "page": 3,
        "chunk_id": 12,
        "text_snippet": "Multi-head attention allows the model..."
      }
    ],
    "model": "gemini-1.5-flash",
    "doc_version": "a4f89d3...",
    "cached_at": "2026-11-10T14:45:00Z",
    "ttl_seconds": 3600
  }
  ```
* **Atomicity & Expiration:** Uses Redis `SETEX key ttl_seconds serialized_json`.
* **Graceful Degradation:** Wrapped in `try-except` blocks. If Redis is down, unreachable, or times out, the cache engine logs a warning and returns `None` (cache miss), allowing normal RAG retrieval and generation to proceed without throwing an unhandled exception.
