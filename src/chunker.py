"""Recursive text chunking module with semantic boundary preservation and overlap.
Ensures context continuity across chunk boundaries while attaching granular metadata.
"""

import hashlib
import re
from typing import List, Dict, Any

def recursive_split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    """Splits text recursively using paragraphs, sentences, and spaces as separators.
    
    Why this matters in RAG:
    - Splitting strictly by fixed characters cuts words and sentences in half.
    - Splitting hierarchically (paragraphs -> sentences -> words) preserves the natural
      semantic units that the embedding model needs to produce accurate vector representations.
    - Overlap ensures facts spanning chunk boundaries are not lost.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    
    def _split(sub_text: str, sep_idx: int) -> List[str]:
        if len(sub_text) <= chunk_size or sep_idx >= len(separators):
            return [sub_text.strip()] if sub_text.strip() else []
        
        sep = separators[sep_idx]
        if sep == "":
            # Fallback: slice by characters
            chunks = []
            start = 0
            while start < len(sub_text):
                end = min(start + chunk_size, len(sub_text))
                chunk = sub_text[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                start += max(1, chunk_size - chunk_overlap)
            return chunks

        splits = sub_text.split(sep)
        chunks = []
        current_chunk = []
        current_len = 0

        for part in splits:
            part_len = len(part) + (len(sep) if current_chunk else 0)
            if current_len + part_len <= chunk_size:
                current_chunk.append(part)
                current_len += part_len
            else:
                if current_chunk:
                    joined = sep.join(current_chunk).strip()
                    if joined:
                        chunks.append(joined)
                    # Prepare next chunk with overlap
                    # Keep trailing elements that fit within overlap limit
                    overlap_parts = []
                    overlap_len = 0
                    for prev_part in reversed(current_chunk):
                        if overlap_len + len(prev_part) <= chunk_overlap:
                            overlap_parts.insert(0, prev_part)
                            overlap_len += len(prev_part)
                        else:
                            break
                    current_chunk = overlap_parts + [part]
                    current_len = sum(len(p) for p in current_chunk) + len(sep) * (len(current_chunk) - 1)
                else:
                    # Single part exceeds chunk_size, split with next granular separator
                    deeper_chunks = _split(part, sep_idx + 1)
                    chunks.extend(deeper_chunks)
                    current_chunk = []
                    current_len = 0

        if current_chunk:
            joined = sep.join(current_chunk).strip()
            if joined:
                chunks.append(joined)

        return chunks

    return _split(text, 0)

def chunk_documents(documents: List[Dict[str, Any]], chunk_size: int = 500, chunk_overlap: int = 100) -> List[Dict[str, Any]]:
    """Takes extracted documents/pages and breaks them into chunks with metadata."""
    chunked_results = []
    chunk_counter = 0

    for doc in documents:
        page_text = doc.get("text", "")
        page_num = doc.get("page", 1)
        source = doc.get("source", "unknown")
        doc_hash = doc.get("doc_hash", "")

        raw_chunks = recursive_split_text(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        for chunk_text in raw_chunks:
            if not chunk_text:
                continue
            chunk_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()[:16]
            chunked_results.append({
                "chunk_id": chunk_counter,
                "text": chunk_text,
                "source": source,
                "page": page_num,
                "doc_hash": doc_hash,
                "chunk_hash": chunk_hash
            })
            chunk_counter += 1

    return chunked_results
