"""
chunking/semantic.py

Embedding-similarity based semantic chunker.
Splits text into sentences, calculates distance between consecutive sentence embeddings,
and places chunk breakpoints at semantic shift boundaries.
"""

import re
import numpy as np
from typing import List, Dict, Any


class SemanticChunker:
    """Semantic chunker using sentence embedding similarity breakpoints."""

    def __init__(self, distance_threshold_percentile: float = 75.0, embedder=None):
        self.distance_threshold_percentile = distance_threshold_percentile
        self.embedder = embedder

    def _split_into_sentences(self, text: str) -> List[str]:
        # Split on standard punctuation while preserving Indic & English sentence endings
        raw_sentences = re.split(r'(?<=[.!?|।])\s+', text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        return sentences if sentences else [text]

    def _compute_fallback_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Lightweight bag-of-words / character n-gram embedding fallback when transformer is unavailable."""
        vocab = {}
        for s in sentences:
            for word in s.lower().split():
                if word not in vocab:
                    vocab[word] = len(vocab)
        
        dim = max(1, len(vocab))
        embeddings = np.zeros((len(sentences), dim), dtype=np.float32)
        for i, s in enumerate(sentences):
            for word in s.lower().split():
                if word in vocab:
                    embeddings[i, vocab[word]] += 1.0
            norm = np.linalg.norm(embeddings[i])
            if norm > 0:
                embeddings[i] /= norm
        return embeddings

    def _get_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        if self.embedder is not None:
            try:
                embeddings = self.embedder.encode(sentences, show_progress_bar=False)
                return np.array(embeddings)
            except Exception:
                pass
        return self._compute_fallback_embeddings(sentences)

    def chunk_text(self, text: str, doc_id: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        sentences = self._split_into_sentences(text)
        if len(sentences) <= 1:
            chunk_meta = (metadata or {}).copy()
            chunk_meta.update({"strategy": "semantic", "sentence_count": len(sentences)})
            return [{
                "chunk_id": f"{doc_id}_sem_0",
                "doc_id": doc_id,
                "text": text,
                "metadata": chunk_meta
            }]

        embeddings = self._get_sentence_embeddings(sentences)
        
        # Calculate cosine distances between adjacent sentence embeddings
        distances = []
        for i in range(len(embeddings) - 1):
            norm1 = np.linalg.norm(embeddings[i])
            norm2 = np.linalg.norm(embeddings[i + 1])
            if norm1 > 0 and norm2 > 0:
                similarity = np.dot(embeddings[i], embeddings[i + 1]) / (norm1 * norm2)
            else:
                similarity = 0.0
            distances.append(1.0 - similarity)

        if distances:
            threshold = np.percentile(distances, self.distance_threshold_percentile)
        else:
            threshold = 0.5

        # Group sentences into semantic chunks
        chunks = []
        current_chunk_sentences = [sentences[0]]
        chunk_idx = 0

        for i, dist in enumerate(distances):
            if dist > threshold:
                # Boundary break
                chunk_str = " ".join(current_chunk_sentences)
                chunk_meta = (metadata or {}).copy()
                chunk_meta.update({
                    "strategy": "semantic",
                    "chunk_index": chunk_idx,
                    "sentence_count": len(current_chunk_sentences)
                })
                chunks.append({
                    "chunk_id": f"{doc_id}_sem_{chunk_idx}",
                    "doc_id": doc_id,
                    "text": chunk_str,
                    "metadata": chunk_meta
                })
                chunk_idx += 1
                current_chunk_sentences = [sentences[i + 1]]
            else:
                current_chunk_sentences.append(sentences[i + 1])

        if current_chunk_sentences:
            chunk_str = " ".join(current_chunk_sentences)
            chunk_meta = (metadata or {}).copy()
            chunk_meta.update({
                "strategy": "semantic",
                "chunk_index": chunk_idx,
                "sentence_count": len(current_chunk_sentences)
            })
            chunks.append({
                "chunk_id": f"{doc_id}_sem_{chunk_idx}",
                "doc_id": doc_id,
                "text": chunk_str,
                "metadata": chunk_meta
            })

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
