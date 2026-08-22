"""
chunking/metadata_aware.py

Metadata-aware chunker that preserves MS MARCO passage boundaries,
structural formatting (paragraphs/sections), and attaches rich doc/query metadata.
"""

from typing import List, Dict, Any


class MetadataAwareChunker:
    """Metadata-aware chunker preserving passage structure and rich document tags."""

    def __init__(self, max_passage_chars: int = 600):
        self.max_passage_chars = max_passage_chars

    def chunk_text(self, text: str, doc_id: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        # Split on paragraph / structural line breaks if present
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [text]

        chunks = []
        chunk_idx = 0

        for para in paragraphs:
            # If paragraph exceeds max_passage_chars, split on sentence boundary
            if len(para) > self.max_passage_chars:
                sentences = [s.strip() for s in para.replace("!", ".").replace("?", ".").split(".") if s.strip()]
                sub_chunk = ""
                for s in sentences:
                    if len(sub_chunk) + len(s) + 1 > self.max_passage_chars and sub_chunk:
                        chunk_meta = (metadata or {}).copy()
                        chunk_meta.update({
                            "strategy": "metadata_aware",
                            "doc_id": doc_id,
                            "chunk_index": chunk_idx,
                            "char_length": len(sub_chunk),
                            "structure_type": "paragraph_segment"
                        })
                        chunks.append({
                            "chunk_id": f"{doc_id}_meta_{chunk_idx}",
                            "doc_id": doc_id,
                            "text": sub_chunk.strip(),
                            "metadata": chunk_meta
                        })
                        chunk_idx += 1
                        sub_chunk = s + "."
                    else:
                        sub_chunk = (sub_chunk + " " + s + ".").strip()
                if sub_chunk:
                    chunk_meta = (metadata or {}).copy()
                    chunk_meta.update({
                        "strategy": "metadata_aware",
                        "doc_id": doc_id,
                        "chunk_index": chunk_idx,
                        "char_length": len(sub_chunk),
                        "structure_type": "paragraph_segment"
                    })
                    chunks.append({
                        "chunk_id": f"{doc_id}_meta_{chunk_idx}",
                        "doc_id": doc_id,
                        "text": sub_chunk,
                        "metadata": chunk_meta
                    })
                    chunk_idx += 1
            else:
                chunk_meta = (metadata or {}).copy()
                chunk_meta.update({
                    "strategy": "metadata_aware",
                    "doc_id": doc_id,
                    "chunk_index": chunk_idx,
                    "char_length": len(para),
                    "structure_type": "full_paragraph"
                })
                chunks.append({
                    "chunk_id": f"{doc_id}_meta_{chunk_idx}",
                    "doc_id": doc_id,
                    "text": para,
                    "metadata": chunk_meta
                })
                chunk_idx += 1

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
