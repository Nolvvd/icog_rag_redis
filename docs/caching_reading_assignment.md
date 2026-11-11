# Technical Summary: Caching Strategies, Invalidation & Redis Eviction
**Course:** Week 1 — RAG & Redis Answer Caching  
**Author:** AI Engineering Intern, iCog Labs  
**Deliverable:** 1-Page Technical Reading Assignment  
**Application Evaluated:** CircuitMind (ECE Hardware Design Intelligence)  

---

## 1. Caching Read Patterns: Cache-Aside vs. Read-Through

| Architecture Pattern | How It Works | Advantages | Disadvantages / Failure Modes |
| :--- | :--- | :--- | :--- |
| **Cache-Aside (Lazy Loading)** *(Used in CircuitMind)* | The application layer directly coordinates the cache and primary stores. On a **Cache Hit**, it returns cached data. On a **Cache Miss**, it queries ChromaDB + Gemini, writes the result to Redis, and returns it. | • **High Resilience**: Cache failure does not crash the system; it gracefully degrades (Design A bypass).<br>• **Resource-Efficient**: Only queried data is cached.<br>• **Custom Schema**: Easily stores composite answers + source citations. | • Cache miss latency penalty on the initial cold query.<br>• Requires application-level coordination. |
| **Read-Through** | The application treats the cache as the primary data store. The cache itself intercepts misses and transparently queries the backing store. | • Simplifies application business logic.<br>• Centralized, standardized access layer. | • Tightly coupled; if the cache proxy goes down, reads fail completely.<br>• Incompatible with multi-step generative RAG without custom middleware. |

---

## 2. Caching Write Patterns: Write-Through vs. Write-Behind vs. Write-Around

* **Write-Through:** Data is written to the cache and the primary store **synchronously** in a single transaction before reporting success.
  - *Pros:* Strong consistency between cache and store; subsequent reads are guaranteed hits.
  - *Cons:* Write latency is doubled; writes fail if either store is unreachable.
* **Write-Behind (Write-Back):** Data is written to the cache **immediately**, and an asynchronous worker queue flushes updates to the primary store in the background.
  - *Pros:* Maximum write throughput and near-zero write latency; smooths out write spikes.
  - *Cons:* Risk of permanent data loss if the cache server crashes before flushing dirty writes.
* **Write-Around** *(Used in CircuitMind)*: Writes bypass the cache entirely, writing **directly to the primary database** (ChromaDB vector store). The cache is only populated on-demand via Cache-Aside when a read occurs.
  - *Pros:* Prevents cache pollution from write-heavy data (e.g. bulk-indexing a 100-page datasheet where 90% of passages may never be queried).
  - *Cons:* The first read after ingestion is guaranteed to be a cache miss.

---

## 3. Cache Lifecycle: TTL, Invalidation, and Eviction

* **TTL (Time-To-Live):** Passive, time-based expiration. Every cached key is tagged with a lifespan (default: 3600s via `SETEX`). Once elapsed, Redis automatically removes the key. Serves as a vital safety net against permanently stale data.
* **Invalidation:** Active, event-driven removal/bypass. Triggered when the underlying data changes (e.g., an engineer uploads a revised hardware datasheet). In CircuitMind, invalidation is achieved by injecting `doc_version = sha256(corpus)` into the cache key, instantly making old cached answers unreachable without costly scanning.
* **Eviction:** Memory-pressure reclamation. Triggered strictly when Redis reaches `maxmemory`. The engine forcibly purges keys according to an eviction policy to make room for new writes.

---

## 4. Eviction Algorithms Comparison

| Algorithm | Mechanism | Best Suited For | Weakness |
| :--- | :--- | :--- | :--- |
| **LRU (Least Recently Used)** | Discards items that have not been read for the longest duration. | **Temporal Locality**: Items queried recently are likely to be queried again soon. | Vulnerable to sequential scans of cold queries that flush out hot keys. |
| **LFU (Least Frequently Used)** | Tracks access frequency counters; discards items with the lowest access count. | **Frequency Locality**: Retains long-term popular items (e.g. standard pinouts). | "Cache pollution" by historic hot keys whose popularity has decayed. |
| **FIFO (First-In, First-Out)** | Evicts keys in the strict order they were inserted, irrespective of access. | Simple queue-like pipelines. | Actively queried hot keys are evicted simply because they are older. |
| **Random Eviction** | Randomly selects keys for eviction. | Uniform access distributions where tracking overhead is unacceptable. | Can evict active hot keys arbitrarily; unpredictable hit rates. |

---

## 5. Supported Redis Eviction Policies (`maxmemory-policy`)

Redis implements memory-efficient approximated algorithms:
1. **`noeviction`** *(Default)*: Returns an error on write commands when memory is full.
2. **`allkeys-lru`**: Evicts least recently used keys across the entire keyspace.
3. **`volatile-lru`**: Evicts least recently used keys **only** among keys configured with a TTL.
4. **`allkeys-lfu`**: Evicts least frequently used keys across the entire keyspace.
5. **`volatile-lfu`**: Evicts least frequently used keys **only** among keys configured with a TTL.
6. **`allkeys-random`**: Randomly evicts keys across the entire keyspace.
7. **`volatile-random`**: Evicts random keys **only** among keys configured with a TTL.
8. **`volatile-ttl`**: Evicts keys with the shortest remaining TTL first.

---

## 6. Architectural Justification & Trade-Offs for CircuitMind

1. **Why Cache-Aside for RAG?**
   Generating a grounded LLM response with semantic retrieval takes $1,200\text{--}2,500\text{ ms}$. Cache-Aside allows CircuitMind to check Redis in **$0.4\text{ ms}$**, yielding a **~3,000× speedup**. Furthermore, it enables **Design A (Bypass Mode)**: if Redis is stopped (`docker stop redis-rag`), the application catches the disconnection, honestly reports `Redis: Unavailable — Cache bypassed`, and routes directly to ChromaDB and Gemini without crashing.
2. **Why Write-Around on Indexing?**
   When hardware manuals are indexed, pre-caching every hypothetical question would cause catastrophic cache pollution. Write-Around writes solely to ChromaDB during ingestion; only questions actively asked by engineers are cached upon first retrieval.
3. **Why Composite Caching (Answer + Sources)?**
   In engineering domains, an ungrounded answer is hazardous. Caching `{ "answer": ..., "sources": [...] }` atomically ensures that cache hits return identical source attribution, file names, and page numbers as cold generations.
4. **Why `volatile-lru` + `doc_version` Invalidation?**
   - **`volatile-lru`** guarantees that only temporary cached RAG responses are evicted under memory pressure, keeping Redis memory footprint bounded while protecting any non-expiring configuration keys.
   - Injecting `doc_version = sha256(corpus)` into the deterministic key format (`rag:cache:<doc_hash>:<model>:<prompt_ver>:<query_hash>`) guarantees that document updates neutralize outdated answers instantly without requiring manual cache flush commands.
