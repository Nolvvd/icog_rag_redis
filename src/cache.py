"""Redis Cache-Aside answer caching module with multi-tier caching:
Tier 1: Deterministic SHA-256 Exact Hash Lookup (O(1), sub-millisecond)
Tier 2: Dense Vector Cosine Similarity Search (O(N), semantic hit on paraphrased questions)
Includes source reference preservation, TTL expiration, and Design A graceful degradation.
"""

import hashlib
import json
import logging
import re
import time
from typing import Dict, Any, Optional, List
import numpy as np

try:
    from vector_store import compute_text_embedding
except ImportError:
    try:
        from .vector_store import compute_text_embedding
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from vector_store import compute_text_embedding

logger = logging.getLogger(__name__)

def normalize_query(query: str) -> str:
    """Normalizes query text to maximize cache hit rate for semantically identical questions.
    
    Steps:
    1. Strip leading and trailing whitespace
    2. Convert to lowercase
    3. Collapse multiple whitespace characters into single space
    4. Strip common trailing punctuation (e.g. '?', '.', '!')
    """
    cleaned = query.strip().lower()
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.rstrip('?.! ')
    return cleaned

def generate_cache_key(
    query: str,
    doc_version: str,
    model_name: str,
    prompt_version: str = "v1.0",
    namespace: str = "rag:cache"
) -> str:
    """Constructs a collision-resistant, version-aware exact cache key.
    
    Key Structure:
    rag:cache:{doc_version}:{model_name}:{prompt_version}:{sha256(normalized_query)}
    """
    normalized = normalize_query(query)
    query_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]
    doc_hash_short = hashlib.sha256(doc_version.encode("utf-8")).hexdigest()[:12]
    return f"{namespace}:{doc_hash_short}:{model_name}:{prompt_version}:{query_hash}"

_shared_fake_server = None

def get_shared_fake_server():
    """Returns a process-wide FakeServer so FakeRedis instances share key storage."""
    global _shared_fake_server
    if _shared_fake_server is None:
        try:
            import fakeredis
            _shared_fake_server = fakeredis.FakeServer()
        except ImportError:
            _shared_fake_server = None
    return _shared_fake_server

class RedisRAGCache:
    """Production Multi-Tier Cache-Aside engine for RAG with Redis.
    
    Tier 1: Exact Hash Cache (O(1), SHA-256 over normalized text, ~0.1-0.3 ms)
    Tier 2: Semantic Vector Cache (O(N), Cosine similarity over embeddings, ~0.4-1.2 ms)
    Tier 3: Full RAG Fallback (Vector Retrieval + LLM Generation, ~1000-2500 ms)
    
    Supports Design A (Bypass Caching on Redis Failure):
    - When Redis is connected: Sub-millisecond answer caching with source preservation.
    - When Redis is unavailable: Gracefully bypasses cache without crashing,
      allowing the core RAG pipeline (ChromaDB + Gemini) to continue functioning.
    """
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        default_ttl: int = 3600,
        bypass_mode: bool = False,
        enable_semantic: bool = True,
        semantic_threshold: float = 0.85
    ):
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.bypass_mode = bypass_mode
        self.enable_semantic = enable_semantic
        self.semantic_threshold = semantic_threshold
        self.client = None
        self.is_connected = False
        self.last_error: Optional[str] = None
        self._fallback_memory: Dict[str, Dict[str, Any]] = {}
        self._fallback_semantic: Dict[str, List[Dict[str, Any]]] = {}
        
        # Telemetry metrics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "writes": 0,
            "errors": 0,
            "exact_hits": 0,
            "semantic_hits": 0
        }
        self.check_connection()

    def check_connection(self) -> bool:
        """Actively checks connection to real Redis.
        Enables dynamic auto-reconnect when Redis is restarted (e.g. docker start redis-rag).
        """
        try:
            import redis
            client = redis.Redis.from_url(
                self.redis_url,
                socket_timeout=1.0,
                socket_connect_timeout=1.0,
                decode_responses=True
            )
            client.ping()
            self.client = client
            self.is_connected = True
            self.last_error = None
            return True
        except Exception as e:
            self.is_connected = False
            self.client = None
            self.last_error = str(e)

            # If bypass_mode is False (unit test emulation mode), allow FakeRedis
            if not self.bypass_mode:
                try:
                    import fakeredis
                    server = get_shared_fake_server()
                    self.client = fakeredis.FakeRedis(server=server, decode_responses=True)
                    self.is_connected = True
                    return True
                except Exception:
                    pass
            return False

    def ping_check(self) -> bool:
        """Actively checks if Redis server is reachable right now."""
        return self.check_connection()

    def get_exact(
        self,
        query: str,
        doc_version: str,
        model_name: str,
        prompt_version: str = "v1.0"
    ) -> Optional[Dict[str, Any]]:
        """Tier 1: Checks exact-match hash cache."""
        if not self.is_connected:
            self.check_connection()

        if not self.is_connected or self.client is None:
            # Memory fallback for unit tests
            key = generate_cache_key(query, doc_version, model_name, prompt_version)
            item = self._fallback_memory.get(key)
            if item and time.time() < item.get("expires_at", 0):
                payload = dict(item["data"])
                payload["key"] = key
                payload["match_type"] = "EXACT"
                payload["similarity"] = 1.0
                payload["matched_query"] = payload.get("query", query)
                return payload
            return None

        key = generate_cache_key(query, doc_version, model_name, prompt_version)
        try:
            raw_data = self.client.get(key)
            if raw_data:
                payload = json.loads(raw_data)
                payload["key"] = key
                payload["match_type"] = "EXACT"
                payload["similarity"] = 1.0
                payload["matched_query"] = payload.get("query", query)
                return payload
            return None
        except Exception as e:
            self.stats["errors"] += 1
            self.is_connected = False
            logger.warning(f"Redis GET exact failed: {e}. Bypassing cache.")
            return None

    def get_semantic(
        self,
        query: str,
        doc_version: str,
        model_name: str,
        prompt_version: str = "v1.0",
        threshold: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Tier 2: Checks semantic vector cache via Cosine Similarity over query embeddings."""
        effective_threshold = threshold if threshold is not None else self.semantic_threshold
        try:
            q_vec = compute_text_embedding(query)
            q_norm = float(np.linalg.norm(q_vec))
            if q_norm == 0:
                return None
        except Exception as e:
            logger.warning(f"Failed to embed query for semantic cache: {e}")
            return None

        doc_hash_short = hashlib.sha256(doc_version.encode("utf-8")).hexdigest()[:12]
        index_key = f"rag:semantic_idx:{doc_hash_short}:{model_name}:{prompt_version}"

        candidates = []
        if self.is_connected and self.client is not None:
            try:
                sem_keys = list(self.client.smembers(index_key))
                if sem_keys:
                    raw_items = self.client.mget(sem_keys)
                    stale_keys = []
                    for k, raw in zip(sem_keys, raw_items):
                        if raw:
                            try:
                                cand = json.loads(raw)
                                cand["_redis_key"] = k
                                candidates.append(cand)
                            except Exception:
                                pass
                        else:
                            stale_keys.append(k)
                    if stale_keys:
                        self.client.srem(index_key, *stale_keys)
            except Exception as e:
                logger.warning(f"Redis semantic lookup failed: {e}")
        else:
            # In-memory semantic candidates
            mem_items = self._fallback_semantic.get(index_key, [])
            now = time.time()
            valid_items = []
            for item in mem_items:
                if now < item.get("expires_at", 0):
                    valid_items.append(item)
                    candidates.append(item["payload"])
            self._fallback_semantic[index_key] = valid_items

        if not candidates:
            return None

        best_cand = None
        best_sim = -1.0

        for cand in candidates:
            emb = cand.get("embedding")
            if not emb:
                continue
            emb_vec = np.array(emb, dtype=np.float32)
            emb_norm = float(np.linalg.norm(emb_vec))
            if emb_norm == 0:
                continue
            sim = float(np.dot(q_vec, emb_vec) / (q_norm * emb_norm))
            if sim > best_sim:
                best_sim = sim
                best_cand = cand

        if best_cand and best_sim >= effective_threshold:
            payload = dict(best_cand)
            payload.pop("embedding", None)
            payload.pop("_redis_key", None)
            payload["match_type"] = "SEMANTIC"
            payload["similarity"] = round(best_sim, 4)
            payload["matched_query"] = best_cand.get("query", "")
            payload["key"] = best_cand.get("_redis_key", index_key)
            return payload

        return None

    def get(
        self,
        query: str,
        doc_version: str,
        model_name: str,
        prompt_version: str = "v1.0",
        enable_semantic: Optional[bool] = None,
        similarity_threshold: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Multi-Tier Cache Lookup:
        1. Tier 1: Check Exact Match SHA-256 hash (O(1), ~0.1 ms).
        2. Tier 2: Check Semantic Similarity Vector Match (O(N), ~0.5 ms).
        3. If both miss, return None to trigger RAG retrieval & generation.
        """
        # Tier 1: Exact Hash Match
        exact_hit = self.get_exact(query, doc_version, model_name, prompt_version)
        if exact_hit:
            self.stats["hits"] += 1
            self.stats["exact_hits"] += 1
            return exact_hit

        # Tier 2: Semantic Similarity Match
        use_semantic = enable_semantic if enable_semantic is not None else self.enable_semantic
        if use_semantic:
            semantic_hit = self.get_semantic(
                query=query,
                doc_version=doc_version,
                model_name=model_name,
                prompt_version=prompt_version,
                threshold=similarity_threshold
            )
            if semantic_hit:
                self.stats["hits"] += 1
                self.stats["semantic_hits"] += 1
                return semantic_hit

        self.stats["misses"] += 1
        return None

    def set(
        self,
        query: str,
        doc_version: str,
        model_name: str,
        answer: str,
        sources: List[Dict[str, Any]],
        ttl: Optional[int] = None,
        prompt_version: str = "v1.0"
    ) -> bool:
        """Stores a successful answer and citations into both Tier 1 and Tier 2 caches.
        
        Strict Guardrails:
        - NEVER caches empty or failed answers.
        - Serializes answer + sources together atomically.
        - Sets expiration TTL.
        - Design A: Gracefully bypasses write if Redis is unavailable.
        """
        if not answer or not answer.strip():
            logger.warning("Refusing to cache empty answer.")
            return False

        if not self.is_connected:
            self.check_connection()

        effective_ttl = ttl if ttl is not None else self.default_ttl
        exact_key = generate_cache_key(query, doc_version, model_name, prompt_version)

        payload = {
            "query": query,
            "answer": answer.strip(),
            "sources": sources,
            "model": model_name,
            "doc_version": doc_version,
            "cached_at": time.time(),
            "ttl": effective_ttl
        }
        serialized = json.dumps(payload)

        # 1. Exact Cache Storage
        if self.is_connected and self.client is not None:
            try:
                self.client.set(exact_key, serialized, ex=effective_ttl)
                self.stats["writes"] += 1
            except Exception as e:
                self.stats["errors"] += 1
                self.is_connected = False
                logger.warning(f"Redis SET failed: {e}. Bypassing cache.")
                return True
        else:
            self._fallback_memory[exact_key] = {
                "data": payload,
                "expires_at": time.time() + effective_ttl
            }
            self.stats["writes"] += 1

        # 2. Semantic Cache Storage
        if self.enable_semantic:
            try:
                q_vec = compute_text_embedding(query)
                doc_hash_short = hashlib.sha256(doc_version.encode("utf-8")).hexdigest()[:12]
                normalized = normalize_query(query)
                query_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]

                sem_key = f"rag:semantic:{doc_hash_short}:{model_name}:{prompt_version}:{query_hash}"
                index_key = f"rag:semantic_idx:{doc_hash_short}:{model_name}:{prompt_version}"

                sem_payload = dict(payload)
                sem_payload["embedding"] = q_vec.tolist()

                if self.is_connected and self.client is not None:
                    self.client.set(sem_key, json.dumps(sem_payload), ex=effective_ttl)
                    self.client.sadd(index_key, sem_key)
                    self.client.expire(index_key, effective_ttl * 2)
                else:
                    if index_key not in self._fallback_semantic:
                        self._fallback_semantic[index_key] = []
                    self._fallback_semantic[index_key].append({
                        "key": sem_key,
                        "payload": sem_payload,
                        "expires_at": time.time() + effective_ttl
                    })
            except Exception as e:
                logger.warning(f"Failed to record semantic cache entry: {e}")

        return True

    def get_synthesis(self, doc_version: str, model_name: str, scope_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieves cached literature synthesis if available."""
        doc_hash_short = hashlib.sha256(doc_version.encode("utf-8")).hexdigest()[:12]
        key = f"rag:synthesis:{doc_hash_short}:{model_name}:{scope_hash}"
        if self.is_connected and self.client is not None:
            try:
                raw = self.client.get(key)
                if raw:
                    self.stats["hits"] += 1
                    return json.loads(raw)
            except Exception:
                pass
        item = self._fallback_memory.get(key)
        if item and time.time() < item.get("expires_at", 0):
            self.stats["hits"] += 1
            return dict(item["data"])
        return None

    def set_synthesis(
        self,
        doc_version: str,
        model_name: str,
        scope_hash: str,
        data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Caches literature synthesis atomically with TTL."""
        doc_hash_short = hashlib.sha256(doc_version.encode("utf-8")).hexdigest()[:12]
        key = f"rag:synthesis:{doc_hash_short}:{model_name}:{scope_hash}"
        effective_ttl = ttl or self.default_ttl
        serialized = json.dumps(data)
        if self.is_connected and self.client is not None:
            try:
                self.client.set(key, serialized, ex=effective_ttl)
                self.stats["writes"] += 1
                return True
            except Exception:
                pass
        self._fallback_memory[key] = {
            "data": data,
            "expires_at": time.time() + effective_ttl
        }
        self.stats["writes"] += 1
        return True

    def clear(self) -> None:
        """Clears all cached RAG entries (Tier 1 exact, Tier 2 semantic, synthesis)."""
        if self.is_connected and self.client is not None:
            try:
                keys = self.client.keys("rag:*")
                if keys:
                    self.client.delete(*keys)
            except Exception:
                pass
        self._fallback_memory.clear()
        self._fallback_semantic.clear()
        self.stats["hits"] = 0
        self.stats["misses"] = 0
        self.stats["writes"] = 0
        self.stats["exact_hits"] = 0
        self.stats["semantic_hits"] = 0

    @property
    def hit_rate(self) -> float:
        total = self.stats["hits"] + self.stats["misses"]
        return (self.stats["hits"] / total * 100.0) if total > 0 else 0.0

    @property
    def is_real_redis(self) -> bool:
        """Returns True if connected to an actual external Redis server instance."""
        return self.is_connected and self.client is not None and "FakeRedis" not in type(self.client).__name__

    @property
    def backend_name(self) -> str:
        """Returns the human-readable name of the active caching engine."""
        if self.is_real_redis:
            return "Real Redis Server"
        elif self.is_connected and self.client is not None:
            return "FakeRedis (In-Memory Emulation)"
        else:
            return "Unavailable — Cache Bypassed"
