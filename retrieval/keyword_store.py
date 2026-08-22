"""
retrieval/keyword_store.py

BM25 Keyword Store using rank_bm25.
Provides sparse retrieval with keyword matching capabilities.
"""

import re
import logging
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi
    RANK_BM25_AVAILABLE = True
except ImportError:
    RANK_BM25_AVAILABLE = False
    logger.warning("rank_bm25 module not found. Falling back to internal TF-IDF keyword scorer.")


class BM25KeywordStore:
    """BM25-based keyword store for sparse term matching."""

    def __init__(self):
        self.bm25 = None
        self.chunks: List[Dict[str, Any]] = []

    def _tokenize(self, text: str) -> List[str]:
        # Clean text and split into lowercase word tokens (handles Indic & English script)
        cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = [t for t in cleaned.split() if len(t) > 1]
        return tokens if tokens else text.lower().split()

    def build_index(self, chunks: List[Dict[str, Any]]):
        if not chunks:
            logger.warning("Empty chunk list provided to BM25 index.")
            return

        self.chunks = chunks
        corpus_tokens = [self._tokenize(c["text"]) for c in chunks]

        if RANK_BM25_AVAILABLE:
            self.bm25 = BM25Okapi(corpus_tokens)
        else:
            self.bm25 = corpus_tokens

        logger.info(f"BM25 index built successfully with {len(chunks)} chunks.")

    def search(self, query_text: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        if not self.chunks or self.bm25 is None:
            return []

        q_tokens = self._tokenize(query_text)
        if not q_tokens:
            return []

        if RANK_BM25_AVAILABLE and isinstance(self.bm25, BM25Okapi):
            scores = self.bm25.get_scores(q_tokens)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            results = []
            for idx in top_indices:
                results.append((self.chunks[idx], float(scores[idx])))
            return results
        else:
            # Fallback simple keyword frequency scorer
            scores = []
            q_set = set(q_tokens)
            for idx, c_toks in enumerate(self.bm25):
                matches = sum(1 for t in c_toks if t in q_set)
                score = float(matches) / (len(q_tokens) ** 0.5) if matches > 0 else 0.0
                scores.append((idx, score))
            scores.sort(key=lambda x: x[1], reverse=True)
            results = []
            for idx, sc in scores[:top_k]:
                results.append((self.chunks[idx], sc))
            return results
