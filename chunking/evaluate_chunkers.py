"""
chunking/evaluate_chunkers.py

Evaluates and compares Recall@k (k=1, 3, 5, 10) across all 4 chunking strategies:
1. Fixed-size with overlap (baseline)
2. Semantic chunking (embedding similarity breakpoints)
3. Metadata-aware chunking (preserving MS MARCO passage boundaries)
4. Hierarchical parent-child chunking

Logs comparative results table and identifies the winning strategy for the README.
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from chunking.fixed_size import FixedSizeChunker
from chunking.semantic import SemanticChunker
from chunking.metadata_aware import MetadataAwareChunker
from chunking.hierarchical import HierarchicalChunker
from data.download_dataset import download_and_prepare_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_recall_at_k(retrieved_doc_ids: List[str], ground_truth_doc_ids: List[str], k: int) -> float:
    top_k = set(retrieved_doc_ids[:k])
    gt = set(ground_truth_doc_ids)
    if not gt:
        return 0.0
    hits = top_k.intersection(gt)
    return len(hits) / len(gt)


from collections import defaultdict

def evaluate_chunking_strategy(chunker, name: str, passages: List[Dict[str, Any]], queries: List[Dict[str, Any]], max_eval_queries: int = 500) -> Dict[str, Any]:
    logger.info(f"Evaluating chunker strategy: {name}...")
    chunks = chunker.chunk_dataset(passages)
    logger.info(f"[{name}] Generated {len(chunks)} chunks from {len(passages)} passages.")

    # Build inverted index for fast keyword matching
    inverted_index = defaultdict(list)
    for idx, c in enumerate(chunks):
        words = set(c["text"].lower().split())
        for w in words:
            inverted_index[w].append(idx)

    eval_queries = queries[:max_eval_queries] if max_eval_queries else queries
    recall_1_list = []
    recall_3_list = []
    recall_5_list = []
    recall_10_list = []

    for q in eval_queries:
        q_text = q["query_text"]
        gt_docs = q.get("relevant_doc_ids", [])
        if not gt_docs:
            continue

        q_words = set(q_text.lower().split())
        candidate_counts = defaultdict(int)
        for w in q_words:
            for c_idx in inverted_index.get(w, []):
                candidate_counts[c_idx] += 1

        top_candidates = sorted(candidate_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        retrieved_doc_ids = [chunks[c_idx]["doc_id"] for c_idx, _ in top_candidates]

        recall_1_list.append(compute_recall_at_k(retrieved_doc_ids, gt_docs, 1))
        recall_3_list.append(compute_recall_at_k(retrieved_doc_ids, gt_docs, 3))
        recall_5_list.append(compute_recall_at_k(retrieved_doc_ids, gt_docs, 5))
        recall_10_list.append(compute_recall_at_k(retrieved_doc_ids, gt_docs, 10))

    mean_r1 = sum(recall_1_list) / max(1, len(recall_1_list))
    mean_r3 = sum(recall_3_list) / max(1, len(recall_3_list))
    mean_r5 = sum(recall_5_list) / max(1, len(recall_5_list))
    mean_r10 = sum(recall_10_list) / max(1, len(recall_10_list))

    return {
        "strategy": name,
        "total_chunks": len(chunks),
        "recall@1": round(mean_r1, 4),
        "recall@3": round(mean_r3, 4),
        "recall@5": round(mean_r5, 4),
        "recall@10": round(mean_r10, 4)
    }


def main():
    data_path = os.path.join(PROJECT_ROOT, "data", "msmarco_subset.json")
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    else:
        logger.info("Dataset file not found. Generating/downloading subset...")
        dataset = download_and_prepare_dataset(limit=5000)

    passages = dataset["passages"]
    queries = dataset["queries"]

    chunkers = [
        (FixedSizeChunker(chunk_size=150, overlap=30), "Fixed-Size (Baseline)"),
        (SemanticChunker(distance_threshold_percentile=70.0), "Semantic (Embedding Distance)"),
        (MetadataAwareChunker(max_passage_chars=500), "Metadata-Aware (MS MARCO Preserving)"),
        (HierarchicalChunker(parent_size=350, child_size=75, child_overlap=20), "Hierarchical (Parent-Child)")
    ]

    results = []
    for chunker, name in chunkers:
        res = evaluate_chunking_strategy(chunker, name, passages, queries)
        results.append(res)

    print("\n" + "="*80)
    print("                      CHUNKING STRATEGY EVALUATION RESULTS")
    print("="*80)
    print(f"{'Strategy':<35} | {'Chunks':<8} | {'Recall@1':<9} | {'Recall@3':<9} | {'Recall@5':<9} | {'Recall@10':<9}")
    print("-" * 88)
    for r in results:
        print(f"{r['strategy']:<35} | {r['total_chunks']:<8} | {r['recall@1']:<9.4f} | {r['recall@3']:<9.4f} | {r['recall@5']:<9.4f} | {r['recall@10']:<9.4f}")
    print("="*80)

    best_strategy = max(results, key=lambda x: x["recall@5"])
    print(f"\nWinner (highest Recall@5): {best_strategy['strategy']} with Recall@5 = {best_strategy['recall@5']:.4f}\n")

    # Save evaluation summary to JSON
    eval_file = os.path.join(PROJECT_ROOT, "chunking", "evaluation_results.json")
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Chunking evaluation saved to {eval_file}")


if __name__ == "__main__":
    main()
