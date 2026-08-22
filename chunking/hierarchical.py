"""
chunking/hierarchical.py

Hierarchical (Parent-Child) chunker.
Generates large parent chunks for generation context and small child chunks
for fine-grained vector & keyword retrieval.
"""

from typing import List, Dict, Any


class HierarchicalChunker:
    """Hierarchical chunker producing linked parent and child chunks."""

    def __init__(self, parent_size: int = 350, child_size: int = 80, child_overlap: int = 20):
        self.parent_size = parent_size
        self.child_size = child_size
        self.child_overlap = child_overlap

    def chunk_text(self, text: str, doc_id: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        words = text.split()
        if not words:
            return []

        parent_chunks = []
        child_chunks = []

        # Create parent chunks
        parent_idx = 0
        for i in range(0, len(words), self.parent_size):
            p_words = words[i : i + self.parent_size]
            p_text = " ".join(p_words)
            parent_id = f"{doc_id}_parent_{parent_idx}"
            
            parent_chunks.append({
                "parent_id": parent_id,
                "text": p_text,
                "doc_id": doc_id
            })

            # Create child chunks for this parent chunk
            step = max(1, self.child_size - self.child_overlap)
            child_idx = 0
            for c_start in range(0, len(p_words), step):
                c_words = p_words[c_start : c_start + self.child_size]
                c_text = " ".join(c_words)
                
                chunk_meta = (metadata or {}).copy()
                chunk_meta.update({
                    "strategy": "hierarchical",
                    "doc_id": doc_id,
                    "parent_id": parent_id,
                    "parent_text": p_text,
                    "parent_index": parent_idx,
                    "child_index": child_idx
                })

                child_chunks.append({
                    "chunk_id": f"{parent_id}_child_{child_idx}",
                    "doc_id": doc_id,
                    "text": c_text,
                    "metadata": chunk_meta
                })
                child_idx += 1
                if c_start + self.child_size >= len(p_words):
                    break

            parent_idx += 1

        return child_chunks

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
