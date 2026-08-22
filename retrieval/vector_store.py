"""
retrieval/vector_store.py

Ultra-lightweight FAISS Vector Store with disk caching and Indic subword/n-gram embeddings.
Zero PyTorch / SentenceTransformer overhead (< 50MB RAM footprint).
Seamlessly runs within 512MB RAM free-tier cloud limits with instant sub-second indexing.
"""

import os
import re
import json
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS module not found. Falling back to NumPy exact inner product search.")


class FAISSVectorStore:
    """FAISS-based dense vector store using lightweight Indic subword n-gram embeddings."""

    def __init__(self, dimension: int = 768, api_key: Optional[str] = None):
        self.dimension = dimension
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.index = None
        self.chunks: List[Dict[str, Any]] = []
        self.index_file = os.path.join(CACHE_DIR, "faiss_index.bin")
        self.chunks_file = os.path.join(CACHE_DIR, "chunks_cache.json")
        self.emb_file = os.path.join(CACHE_DIR, "embeddings.npy")

    def _encode_text(self, text: str) -> np.ndarray:
        """Generates a dense normalized subword n-gram feature vector for multilingual text."""
        vec = np.zeros(self.dimension, dtype=np.float32)
        if not text:
            return vec

        text_lower = text.lower()
        # Extract word tokens including all Indic unicode ranges
        words = re.findall(r'[\w\u0900-\u0D7F]+', text_lower)
        for w in words:
            # Word-level hash
            idx = abs(hash(f"w_{w}")) % self.dimension
            vec[idx] += 2.0

            # Subword 3-gram & 4-gram character hashes for cross-lingual / morphological matching
            for n in (3, 4):
                if len(w) >= n:
                    for i in range(len(w) - n + 1):
                        ngram = w[i:i + n]
                        n_idx = abs(hash(f"ng_{ngram}")) % self.dimension
                        vec[n_idx] += 1.0

        # L2 Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def _encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """Batch encodes list of texts into normalized 2D numpy array."""
        embeddings = [self._encode_text(t) for t in texts]
        return np.array(embeddings, dtype=np.float32)

    def load_cached_index(self) -> bool:
        """Attempts to load pre-computed index and chunks from disk cache."""
        try:
            if os.path.exists(self.chunks_file) and (os.path.exists(self.index_file) or os.path.exists(self.emb_file)):
                with open(self.chunks_file, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)

                if FAISS_AVAILABLE and os.path.exists(self.index_file):
                    self.index = faiss.read_index(self.index_file)
                    self.dimension = self.index.d
                    logger.info(f"Loaded cached FAISS index ({len(self.chunks)} chunks, dim={self.dimension}) from disk.")
                    return True
                elif os.path.exists(self.emb_file):
                    self.index = np.load(self.emb_file)
                    self.dimension = self.index.shape[1]
                    logger.info(f"Loaded cached NumPy embeddings ({len(self.chunks)} chunks, dim={self.dimension}) from disk.")
                    return True
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Will rebuild index.")
        return False

    def build_index(self, chunks: List[Dict[str, Any]], force_rebuild: bool = False):
        """Builds dense FAISS index in memory and caches to disk."""
        if not chunks:
            logger.warning("Empty chunk list provided to FAISS index.")
            return

        if not force_rebuild and self.load_cached_index() and len(self.chunks) == len(chunks):
            logger.info("Using cached FAISS vector index.")
            return

        self.chunks = chunks
        texts = [c.get("retrieval_context", c.get("text", "")) for c in chunks]
        embeddings = self._encode(texts, is_query=False)
        self.dimension = embeddings.shape[1]

        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(embeddings)
            try:
                faiss.write_index(self.index, self.index_file)
            except Exception as e:
                logger.warning(f"Could not save FAISS index to disk: {e}")
        else:
            self.index = embeddings
            try:
                np.save(self.emb_file, embeddings)
            except Exception as e:
                logger.warning(f"Could not save embeddings to disk: {e}")

        try:
            with open(self.chunks_file, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Could not save chunks cache: {e}")

        logger.info(f"FAISS index built & cached successfully with {len(chunks)} chunks (dimension: {self.dimension}).")

    def search(self, query_text: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Searches FAISS vector store with query text and returns top_k (chunk, score) pairs."""
        if not self.chunks or self.index is None:
            return []

        q_emb = self._encode([query_text], is_query=True)

        if FAISS_AVAILABLE and isinstance(self.index, faiss.Index):
            scores, indices = self.index.search(q_emb, min(top_k, len(self.chunks)))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunks) and idx >= 0:
                    results.append((self.chunks[idx], float(score)))
            return results
        else:
            matrix = self.index
            sims = np.dot(matrix, q_emb[0])
            top_indices = np.argsort(sims)[::-1][:top_k]
            results = []
            for idx in top_indices:
                results.append((self.chunks[idx], float(sims[idx])))
            return results
