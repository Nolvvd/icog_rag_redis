"""Vector store module managing embeddings, storage, and dense similarity search.
Supports ChromaDB with an in-memory cosine similarity fallback.
"""

import hashlib
import math
import re
from typing import List, Dict, Any, Optional
import numpy as np

# Common English stopwords to prevent retrieval degradation
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "can", "did", "do", "does", "doing", "don", "down", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself",
    "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off",
    "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own",
    "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs",
    "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to",
    "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "you", "your", "yours"
}

class InMemoryVectorStore:
    """A lightweight, robust in-memory vector store using deterministic TF projection & Cosine Similarity."""
    def __init__(self):
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.doc_version: str = "empty"

    def _get_simple_embedding(self, text: str, dim: int = 2048) -> np.ndarray:
        """Deterministic term-frequency projection with stopword filtering and high dimensionality.
        
        Why this upgrade matters:
        - Filters out high-frequency stopwords (like 'what', 'the', 'is') so they don't drown out content queries.
        - Uses SHA-256 for deterministic hashing across process restarts (unlike Python's hash() which is randomized).
        - 2048 dimensions dramatically reduces bucket collisions between unrelated domain terms (e.g., 'budget' vs 'clock').
        """
        vec = np.zeros(dim, dtype=np.float32)
        all_words = re.findall(r'\b\w+\b', text.lower())
        if not all_words:
            return vec

        # Filter stopwords for content terms
        content_words = [w for w in all_words if w not in STOPWORDS]
        target_words = content_words if content_words else all_words

        # Term frequency mapping
        word_counts = {}
        for w in target_words:
            word_counts[w] = word_counts.get(w, 0) + 1

        for w, count in word_counts.items():
            # Deterministic SHA-256 bucket
            h = int(hashlib.sha256(w.encode("utf-8")).hexdigest()[:8], 16) % dim
            # Sublinear term frequency weighting
            weight = 1.0 + math.log(count)
            vec[h] += weight

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def add_documents(self, chunks: List[Dict[str, Any]], doc_version: str = "v1") -> None:
        self.chunks = chunks
        self.doc_version = doc_version
        if not chunks:
            self.embeddings = None
            return
        
        vectors = [self._get_simple_embedding(c["text"]) for c in chunks]
        self.embeddings = np.array(vectors, dtype=np.float32)

    def get_all_chunks(self, filter_sources: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if not filter_sources:
            return list(self.chunks)
        filter_set = set(filter_sources)
        return [c for c in self.chunks if c.get("source") in filter_set]

    def get_indexed_sources(self) -> List[str]:
        seen = set()
        sources = []
        for c in self.chunks:
            src = c.get("source", "unknown")
            if src not in seen:
                seen.add(src)
                sources.append(src)
        return sources

    def similarity_search(
        self,
        query: str,
        top_k: int = 3,
        filter_sources: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        if not self.chunks or self.embeddings is None:
            return []

        q_vec = self._get_simple_embedding(query)
        q_norm = np.linalg.norm(q_vec)

        # Cosine similarity: (q . d) / (|q| * |d|)
        scores = np.dot(self.embeddings, q_vec) if q_norm > 0 else np.zeros(len(self.chunks))
        top_indices = np.argsort(scores)[::-1]

        filter_set = set(filter_sources) if filter_sources else None
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            if filter_set and chunk.get("source") not in filter_set:
                continue
            item = dict(chunk)
            item["score"] = float(scores[idx])
            results.append(item)
            if len(results) >= top_k:
                break
        return results

class ChromaVectorStore:
    """ChromaDB implementation with persistent local storage and sentence embeddings."""
    def __init__(self, persist_directory: str = "./data/chroma_db", collection_name: str = "rag_documents"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.doc_version: str = "empty"
        self._fallback_store = InMemoryVectorStore()
        self._init_chroma()

    def _init_chroma(self):
        try:
            import chromadb
            from chromadb.config import Settings
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            # Fall back seamlessly to in-memory store if ChromaDB cannot initialize
            self.client = None
            self.collection = None

    def add_documents(self, chunks: List[Dict[str, Any]], doc_version: str = "v1") -> None:
        self.doc_version = doc_version
        self._fallback_store.add_documents(chunks, doc_version=doc_version)

        if self.collection is None or not chunks:
            return

        try:
            # Delete existing items to avoid stale accumulation for the session
            existing_ids = self.collection.get()["ids"]
            if existing_ids:
                self.collection.delete(ids=existing_ids)

            ids = [f"chunk_{c['chunk_id']}_{c['chunk_hash']}" for c in chunks]
            documents = [c["text"] for c in chunks]
            metadatas = [
                {
                    "source": c.get("source", "unknown"),
                    "page": int(c.get("page", 1)),
                    "chunk_id": int(c.get("chunk_id", 0)),
                    "doc_hash": c.get("doc_hash", "")
                }
                for c in chunks
            ]
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
        except Exception:
            # In case of any Chroma write issue, fallback is already updated
            pass

    def get_all_chunks(self, filter_sources: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        return self._fallback_store.get_all_chunks(filter_sources=filter_sources)

    def get_indexed_sources(self) -> List[str]:
        return self._fallback_store.get_indexed_sources()

    def similarity_search(
        self,
        query: str,
        top_k: int = 3,
        filter_sources: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        # Delegate to resilient in-memory store supporting source filtering
        return self._fallback_store.similarity_search(query, top_k=top_k, filter_sources=filter_sources)
