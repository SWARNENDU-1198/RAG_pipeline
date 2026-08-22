"""
scripts/reindex_gemini.py

Re-indexes the dataset passages with Google Gemini text-embedding-004
and saves the 768-dim FAISS index and chunks cache to data/cache/.
"""

import os
import sys
import json
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from chunking.hierarchical import HierarchicalChunker
from retrieval.vector_store import FAISSVectorStore
from retrieval.keyword_store import BM25KeywordStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

def main():
    data_file = os.path.join(PROJECT_ROOT, "data", "msmarco_subset.json")
    if not os.path.exists(data_file):
        logger.error(f"Dataset file not found at {data_file}")
        return

    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        passages = data.get("passages", [])

    logger.info(f"Loaded {len(passages)} passages from {data_file}.")
    chunker = HierarchicalChunker(parent_size=350, child_size=75, child_overlap=20)
    chunks = chunker.chunk_dataset(passages)
    logger.info(f"Generated {len(chunks)} chunks.")

    logger.info("Re-building 768-dim FAISS index with Gemini text-embedding-004...")
    dense_store = FAISSVectorStore()
    dense_store.build_index(chunks, force_rebuild=True)
    logger.info("Dense FAISS index build complete!")

    logger.info("Building BM25 sparse index...")
    sparse_store = BM25KeywordStore()
    sparse_store.build_index(chunks)
    logger.info("BM25 index build complete!")

    # Verify search
    test_query = "What is Calangute Beach famous for?"
    results = dense_store.search(test_query, top_k=3)
    logger.info(f"Test query: '{test_query}'")
    for i, (chunk, score) in enumerate(results):
        logger.info(f"[{i+1}] Score: {score:.4f} | Text: {chunk['text'][:100]}...")

if __name__ == "__main__":
    main()
