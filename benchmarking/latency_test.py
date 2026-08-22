"""
benchmarking/latency_test.py

Runs >= 100 test queries through the pipeline, logging per-stage and end-to-end timings.
Saves raw records to benchmarking/latency_records.json for statistical evaluation.
"""

import os
import sys
import json
import time
import logging
from typing import List, Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.orchestrator import VoiceRAGPipelineOrchestrator
from pipeline.schemas import AudioInput

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
RECORDS_FILE = os.path.join(BENCHMARK_DIR, "latency_records.json")


def generate_benchmark_queries(num_queries: int = None) -> List[str]:
    """Generates test query set combining all dataset queries and domain queries."""
    data_file = os.path.join(PROJECT_ROOT, "data", "msmarco_subset.json")
    dataset_queries = []
    
    if os.path.exists(data_file):
        with open(data_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)
            dataset_queries = [q["query_text"].strip() for q in dataset.get("queries", []) if q.get("query_text")]

    base_queries = [
        "What is Calangute Beach famous for?",
        "Where are the remains of Saint Francis Xavier stored in Goa?",
        "Which river forms the Dudhsagar Falls in Goa?",
        "What is the capital of Goa?",
        "To whom is the Shantadurga temple in Goa dedicated?",
        "What languages does Sarvam AI speech-to-text support?",
        "What is FAISS used for in vector retrieval?",
        "How does BM25 rank retrieved passages?",
        "What is hybrid search in RAG systems?",
        "Why are grounding checks necessary in RAG pipelines?",
        "When did Chandrayaan-3 land on the moon?",
        "Who developed the UPI payment system in India?",
        "What is Ayurveda?",
        "Where did Yoga originate?",
        "Who designed the Indian Rupee symbol?"
    ]

    all_queries = []
    seen = set()
    for q in dataset_queries + base_queries:
        if q and q not in seen:
            seen.add(q)
            all_queries.append(q)

    if num_queries is None or num_queries <= 0:
        return all_queries

    benchmark_queries = []
    idx = 0
    while len(benchmark_queries) < num_queries:
        benchmark_queries.append(all_queries[idx % len(all_queries)])
        idx += 1

    return benchmark_queries[:num_queries]


def run_latency_benchmark(num_queries: int = 150):
    logger.info(f"Starting Latency Benchmark across dataset questions (evaluating {num_queries} queries)...")
    orchestrator = VoiceRAGPipelineOrchestrator()
    orchestrator.initialize_index()

    queries = generate_benchmark_queries(num_queries=num_queries)
    logger.info(f"Loaded {len(queries)} unique test questions from dataset.")
    records = []

    # Warmup
    if queries:
        orchestrator.run(AudioInput(text_override=queries[0]))

    for idx, query in enumerate(queries):
        input_data = AudioInput(text_override=query)
            
        start_time = time.perf_counter()
        output = orchestrator.run(input_data)
        elapsed_total = (time.perf_counter() - start_time) * 1000

        record = {
            "query_id": idx + 1,
            "query_text": query,
            "is_refused": output.is_refused,
            "total_latency_ms": output.total_latency_ms,
            "elapsed_measured_ms": round(elapsed_total, 2),
            "stt_latency_ms": output.stage_timings.get("stt", 0.0),
            "input_guardrail_latency_ms": output.stage_timings.get("input_guardrail", 0.0),
            "retrieval_latency_ms": output.stage_timings.get("retrieval", 0.0),
            "generation_latency_ms": output.stage_timings.get("generation", 0.0),
            "grounding_check_latency_ms": output.stage_timings.get("grounding_check", 0.0),
            "output_guardrail_latency_ms": output.stage_timings.get("output_guardrail", 0.0)
        }
        records.append(record)

        if (idx + 1) % 25 == 0 or (idx + 1) == len(queries):
            logger.info(f"Processed {idx + 1}/{len(queries)} queries. Latest retrieval latency: {record['retrieval_latency_ms']} ms, total: {record['total_latency_ms']} ms.")

    os.makedirs(BENCHMARK_DIR, exist_ok=True)
    with open(RECORDS_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    logger.info(f"Successfully saved {len(records)} benchmark records to {RECORDS_FILE}")


if __name__ == "__main__":
    run_latency_benchmark(num_queries=150)
