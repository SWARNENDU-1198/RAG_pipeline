"""
chunking/fixed_size.py

Baseline fixed-size text chunker with configurable token/word overlap.
"""

from typing import List, Dict, Any


class FixedSizeChunker:
    """Fixed-size chunker splitting text into fixed word counts with overlap."""

    def __init__(self, chunk_size: int = 150, overlap: int = 30):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str, doc_id: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        words = text.split()
        if not words:
            return []

        chunks = []
        step = max(1, self.chunk_size - self.overlap)
        
        chunk_idx = 0
        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.chunk_size]
            chunk_str = " ".join(chunk_words)
            
            chunk_meta = (metadata or {}).copy()
            chunk_meta.update({
                "strategy": "fixed_size",
                "chunk_index": chunk_idx,
                "word_count": len(chunk_words)
            })

            chunks.append({
                "chunk_id": f"{doc_id}_fs_{chunk_idx}",
                "doc_id": doc_id,
                "text": chunk_str,
                "metadata": chunk_meta
            })
            chunk_idx += 1
            
            if i + self.chunk_size >= len(words):
                break

        return chunks

    def chunk_dataset(self, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_chunks = []
        for passage in passages:
            doc_id = passage.get("doc_id", "doc_unknown")
            text = passage.get("text", "")
            meta = passage.get("metadata", {})
            meta["topic"] = passage.get("topic", "general")
            chunks = self.chunk_text(text, doc_id=doc_id, metadata=meta)
            all_chunks.extend(chunks)
        return all_chunks
