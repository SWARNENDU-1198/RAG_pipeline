"""
retrieval/vector_store.py

FAISS In-Memory Vector Store with disk caching for low-latency dense vector search.
Uses multilingual embeddings (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2).
Supports exact cosine similarity / Inner Product indexing and disk persistence.
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Tuple

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
    """FAISS-based dense vector store with cosine similarity index and disk caching."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.model = None
        self.index = None
        self.chunks: List[Dict[str, Any]] = []
        self.dimension = None
        self.index_file = os.path.join(CACHE_DIR, "faiss_index.bin")
        self.chunks_file = os.path.join(CACHE_DIR, "chunks_cache.json")
        self.emb_file = os.path.join(CACHE_DIR, "embeddings.npy")

    def _load_model(self):
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}...")
            try:
                import os
                import gc
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
                import torch
                torch.set_grad_enabled(False)
                torch.set_num_threads(1)
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
                gc.collect()
            except Exception as e:
                logger.warning(f"Could not load SentenceTransformer ({e}). Using lightweight TF-IDF embedder fallback.")
                self.model = "fallback"

    def _encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        self._load_model()
        if self.model != "fallback":
            formatted_texts = texts
            if "e5" in self.model_name.lower():
                prefix = "query: " if is_query else "passage: "
                formatted_texts = [prefix + t for t in texts]
            
            try:
                embeddings = self.model.encode(
                    formatted_texts,
                    batch_size=16,
                    show_progress_bar=False,
                    normalize_embeddings=True
                )
                return np.array(embeddings, dtype=np.float32)
            except Exception as e:
                logger.warning(f"Encoding failed ({e}). Falling back to lightweight vectorizer.")
                return self._fallback_encode(texts)
        else:
            return self._fallback_encode(texts)

    def _fallback_encode(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            vec = np.zeros(384, dtype=np.float32)
            words = text.lower().split()
            for w in words:
                idx = abs(hash(w)) % 384
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            embeddings.append(vec)
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
                    self._load_model()
                    return True
                elif os.path.exists(self.emb_file):
                    self.index = np.load(self.emb_file)
                    self.dimension = self.index.shape[1]
                    logger.info(f"Loaded cached NumPy embeddings ({len(self.chunks)} chunks) from disk.")
                    self._load_model()
                    return True
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Will rebuild index.")
        return False

    def build_index(self, chunks: List[Dict[str, Any]], force_rebuild: bool = False):
        if not chunks:
            logger.warning("Empty chunk list provided to FAISS index.")
            return

        if not force_rebuild and self.load_cached_index() and len(self.chunks) == len(chunks):
            self._load_model()
            return

        self.chunks = chunks
        texts = [c["text"] for c in chunks]
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
