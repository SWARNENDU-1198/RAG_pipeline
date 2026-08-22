"""
pipeline/orchestrator.py

Main Orchestrator Harness & State Machine for Voice-Enabled RAG Pipeline.
Chains: Sarvam STT -> Pre-Retrieval Guardrail -> Hybrid Retrieval -> Post-Retrieval Confidence -> Google Gemini/LLM Generation -> Grounding Check -> Output Guardrail -> Sarvam TTS.

Supports 3 Knowledge Modes:
1. 'hybrid_auto' (Default): Checks dataset first; automatically expands to Gemini World Knowledge if out-of-dataset.
2. 'dataset_only': Strict RAG mode; refuses if outside dataset or below confidence threshold.
3. 'open_knowledge': Direct Gemini World Knowledge mode.

Logs stage durations, enforces retries, handles error fallbacks, and validates I/O via Pydantic.
"""

import os
import sys
import time
import json
import logging
from typing import Dict, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.schemas import (
    AudioInput, STTResult, InputGuardrailResult, RetrievedChunk,
    RetrievalResult, GenerationResult, GroundingResult, FinalPipelineOutput
)
from stt.sarvam_client import SarvamSTTClient
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.vector_store import FAISSVectorStore
from retrieval.keyword_store import BM25KeywordStore
from chunking.hierarchical import HierarchicalChunker
from guardrails.input_guardrail import InputGuardrail
from guardrails.grounding_check import GroundingCheck
from guardrails.output_guardrail import OutputGuardrail, DEFAULT_REFUSAL_MESSAGE
from generation.llm_client import get_llm_client
from data.download_dataset import download_and_prepare_dataset

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


class VoiceRAGPipelineOrchestrator:
    """End-to-End Orchestrator state machine for Voice RAG Pipeline integrating Sarvam AI & Google Gemini."""

    def __init__(self, confidence_threshold: float = 0.45, grounding_threshold: float = 0.35):
        logger.info("Initializing Voice RAG Pipeline Orchestrator (Sarvam AI + Google Gemini)...")
        self.stt_client = SarvamSTTClient()
        self.retriever = HybridRetriever(alpha=0.5)
        self.input_guardrail = InputGuardrail(confidence_threshold=confidence_threshold)
        self.grounding_check = GroundingCheck(grounding_threshold=grounding_threshold)
        self.output_guardrail = OutputGuardrail()
        self.llm_client = get_llm_client()
        self.is_indexed = False

    def initialize_index(self, passages: Optional[list] = None):
        """Indexes dataset passages using Hierarchical Chunker and builds vector + keyword indices."""
        if self.is_indexed:
            return

        if passages is None:
            data_file = os.path.join(PROJECT_ROOT, "data", "msmarco_subset.json")
            if os.path.exists(data_file):
                with open(data_file, "r", encoding="utf-8") as f:
                    dataset = json.load(f)
                    passages = dataset["passages"]
            else:
                dataset = download_and_prepare_dataset(limit=5000)
                passages = dataset["passages"]

        logger.info(f"Chunking {len(passages)} passages with Hierarchical Chunker...")
        chunker = HierarchicalChunker(parent_size=350, child_size=75, child_overlap=20)
        chunks = chunker.chunk_dataset(passages)
        
        logger.info(f"Building hybrid FAISS + BM25 index over {len(chunks)} chunks...")
        self.retriever.build_indices(chunks)
        self.is_indexed = True
        logger.info("Pipeline index initialization complete.")

    def run(self, input_data: AudioInput) -> FinalPipelineOutput:
        """Executes the pipeline end-to-end with per-stage timing and Pydantic schema validation."""
        pipeline_start = time.perf_counter()
        stage_timings: Dict[str, float] = {}
        knowledge_mode = input_data.knowledge_mode or "dataset_only"

        # If index is not yet built, build it
        if not self.is_indexed:
            self.initialize_index()

        # STAGE 1: Speech-to-Text (STT via Sarvam AI)
        stt_start = time.perf_counter()
        transcript = ""
        stt_provider = "text_override"
        stt_conf = 1.0

        if input_data.audio_bytes and len(input_data.audio_bytes) > 50:
            raw_stt = self.stt_client.transcribe_audio(
                audio_bytes=input_data.audio_bytes,
                language_code=input_data.language_code,
                filename=input_data.filename
            )
            transcript = raw_stt.get("transcript", "").strip()
            stt_provider = raw_stt.get("provider", "sarvam_ai")
            stt_conf = float(raw_stt.get("confidence", 0.9))

        # If audio transcription was empty or not provided, check text_override
        if not transcript and input_data.text_override and input_data.text_override.strip():
            transcript = input_data.text_override.strip()
            stt_provider = "text_override"
            stt_conf = 1.0

        stt_dur = (time.perf_counter() - stt_start) * 1000
        stt_pydantic = STTResult(
            transcript=transcript,
            confidence=stt_conf,
            provider=stt_provider,
            duration_ms=round(stt_dur, 2),
            status="SUCCESS" if transcript else "EMPTY"
        )
        stage_timings["stt"] = stt_pydantic.duration_ms
        logger.info(f"[Stage 1: STT] Provider: {stt_pydantic.provider} | Transcript: '{transcript}' ({stt_pydantic.duration_ms} ms)")

        # If no transcript could be obtained from audio or text
        if not transcript:
            no_speech_reason = "No speech was detected or audio could not be transcribed. Please speak clearly or check SARVAM_API_KEY."
            stage_timings["input_guardrail"] = 0.0
            stage_timings["retrieval"] = 0.0
            stage_timings["generation"] = 0.0
            stage_timings["grounding_check"] = 0.0
            stage_timings["output_guardrail"] = 0.0
            total_dur = (time.perf_counter() - pipeline_start) * 1000

            return FinalPipelineOutput(
                query_text="",
                final_answer="No speech was detected in the audio recording. Please speak clearly into the microphone.",
                is_refused=True,
                refusal_reason=no_speech_reason,
                retrieved_chunks=[],
                stt_result=stt_pydantic,
                input_guardrail=InputGuardrailResult(
                    safety_passed=False,
                    confidence_passed=False,
                    top_score=0.0,
                    reasoning=no_speech_reason,
                    duration_ms=0.0,
                    status="REFUSED"
                ),
                retrieval_result=RetrievalResult(chunks=[], top_score=0.0, duration_ms=0.0, status="SKIPPED"),
                generation_result=GenerationResult(raw_answer="", duration_ms=0.0, status="SKIPPED"),
                grounding_result=GroundingResult(is_grounded=False, grounding_score=0.0, reasoning=no_speech_reason, duration_ms=0.0, status="SKIPPED"),
                knowledge_mode=knowledge_mode,
                source_type="none",
                total_latency_ms=round(total_dur, 2),
                stage_timings=stage_timings
            )

        # STAGE 2: Pre-Retrieval Input Safety Guardrail
        ig_start = time.perf_counter()
        safety_passed, safety_reason = self.input_guardrail.check_safety(transcript)
        ig_dur = (time.perf_counter() - ig_start) * 1000

        if not safety_passed:
            ig_pydantic = InputGuardrailResult(
                safety_passed=False,
                confidence_passed=False,
                top_score=0.0,
                reasoning=safety_reason,
                duration_ms=round(ig_dur, 2),
                status="REFUSED"
            )
            stage_timings["input_guardrail"] = ig_pydantic.duration_ms
            stage_timings["retrieval"] = 0.0
            stage_timings["generation"] = 0.0
            stage_timings["grounding_check"] = 0.0
            stage_timings["output_guardrail"] = 0.0
            total_dur = (time.perf_counter() - pipeline_start) * 1000

            out = self.output_guardrail.process_output(
                raw_answer="", safety_passed=False, confidence_passed=False, grounding_passed=False, refusal_reason=safety_reason
            )

            return FinalPipelineOutput(
                query_text=transcript,
                final_answer=out["final_answer"],
                is_refused=True,
                refusal_reason=safety_reason,
                retrieved_chunks=[],
                stt_result=stt_pydantic,
                input_guardrail=ig_pydantic,
                retrieval_result=RetrievalResult(chunks=[], top_score=0.0, duration_ms=0.0, status="SKIPPED"),
                generation_result=GenerationResult(raw_answer="", duration_ms=0.0, status="SKIPPED"),
                grounding_result=GroundingResult(is_grounded=False, grounding_score=0.0, reasoning=safety_reason, duration_ms=0.0, status="SKIPPED"),
                knowledge_mode=knowledge_mode,
                source_type="none",
                total_latency_ms=round(total_dur, 2),
                stage_timings=stage_timings
            )

        # STAGE 3: Hybrid Retrieval (or bypassed for open_knowledge)
        retrieved_raw = []
        retrieved_chunks = []
        top_score = 0.0

        if knowledge_mode == "open_knowledge":
            stage_timings["retrieval"] = 0.0
            retrieval_pydantic = RetrievalResult(
                chunks=[],
                top_score=0.0,
                retrieval_mode="skipped_open_knowledge",
                duration_ms=0.0,
                status="SKIPPED"
            )
            conf_passed = True
            conf_reason = "Open knowledge mode selected; bypassing dataset retrieval."
        else:
            ret_start = time.perf_counter()
            retrieved_raw = self.retriever.retrieve(query_text=transcript, top_k=5)
            ret_dur = (time.perf_counter() - ret_start) * 1000
            
            retrieved_chunks = [
                RetrievedChunk(
                    chunk_id=c["chunk_id"],
                    doc_id=c["doc_id"],
                    text=c["text"],
                    dense_score=c.get("dense_score", 0.0),
                    sparse_score=c.get("sparse_score", 0.0),
                    hybrid_score=c.get("hybrid_score", 0.0),
                    retrieval_context=c.get("retrieval_context", c["text"]),
                    metadata=c.get("metadata", {})
                )
                for c in retrieved_raw
            ]

            top_score = retrieved_chunks[0].hybrid_score if retrieved_chunks else 0.0
            retrieval_pydantic = RetrievalResult(
                chunks=retrieved_chunks,
                top_score=top_score,
                retrieval_mode="hybrid_faiss_bm25",
                duration_ms=round(ret_dur, 2),
                status="SUCCESS"
            )
            stage_timings["retrieval"] = retrieval_pydantic.duration_ms
            logger.info(f"[Stage 3: Retrieval] Found {len(retrieved_chunks)} chunks, top score: {top_score:.4f} ({retrieval_pydantic.duration_ms} ms)")

            # STAGE 4: Post-Retrieval Score Confidence Check
            conf_passed, top_sc, conf_reason = self.input_guardrail.check_retrieval_confidence(retrieved_raw)

        ig_pydantic = InputGuardrailResult(
            safety_passed=True,
            confidence_passed=conf_passed,
            top_score=top_score,
            reasoning=conf_reason,
            duration_ms=round(ig_dur, 2),
            status="SUCCESS" if conf_passed else ("HYBRID_FALLBACK" if knowledge_mode == "hybrid_auto" else "REFUSED")
        )
        stage_timings["input_guardrail"] = ig_pydantic.duration_ms

        # In strict dataset_only mode, refuse if confidence check failed
        if not conf_passed and knowledge_mode == "dataset_only":
            stage_timings["generation"] = 0.0
            stage_timings["grounding_check"] = 0.0
            stage_timings["output_guardrail"] = 0.0
            total_dur = (time.perf_counter() - pipeline_start) * 1000

            out = self.output_guardrail.process_output(
                raw_answer="", safety_passed=True, confidence_passed=False, grounding_passed=False, refusal_reason=conf_reason
            )

            return FinalPipelineOutput(
                query_text=transcript,
                final_answer=out["final_answer"],
                is_refused=True,
                refusal_reason=conf_reason,
                retrieved_chunks=retrieved_chunks,
                stt_result=stt_pydantic,
                input_guardrail=ig_pydantic,
                retrieval_result=retrieval_pydantic,
                generation_result=GenerationResult(raw_answer="", duration_ms=0.0, status="SKIPPED"),
                grounding_result=GroundingResult(is_grounded=False, grounding_score=0.0, reasoning=conf_reason, duration_ms=0.0, status="SKIPPED"),
                knowledge_mode=knowledge_mode,
                source_type="dataset_rag",
                total_latency_ms=round(total_dur, 2),
                stage_timings=stage_timings
            )

        # STAGE 5: LLM Answer Generation (Google Gemini / LLM)
        gen_start = time.perf_counter()
        
        effective_chunks = retrieved_raw if (conf_passed and knowledge_mode != "open_knowledge") else []
        effective_mode = "dataset_only" if knowledge_mode == "dataset_only" else ("open_knowledge" if (knowledge_mode == "open_knowledge" or not conf_passed) else "hybrid_auto")

        llm_res = self.llm_client.generate(
            query=transcript,
            context_chunks=effective_chunks,
            knowledge_mode=effective_mode
        )
        gen_dur = (time.perf_counter() - gen_start) * 1000
        
        source_type = llm_res.get("source_type", "dataset_rag" if effective_chunks else "gemini_world")
        gen_pydantic = GenerationResult(
            raw_answer=llm_res["answer"],
            provider=llm_res.get("provider", "google_gemini"),
            model=llm_res.get("model", "gemini-3.6-flash"),
            source_type=source_type,
            duration_ms=round(gen_dur, 2),
            status="SUCCESS"
        )
        stage_timings["generation"] = gen_pydantic.duration_ms
        logger.info(f"[Stage 5: Generation] Provider: {gen_pydantic.provider} ({gen_pydantic.model}) | Source: {source_type} ({gen_pydantic.duration_ms} ms)")

        # STAGE 6: Post-Generation Grounding Verification
        gr_start = time.perf_counter()
        if source_type == "gemini_world" or knowledge_mode == "open_knowledge" or not effective_chunks:
            is_grounded = True
            grounding_score = 1.0
            gr_reason = "Open knowledge mode; answered from Gemini world knowledge."
        else:
            is_grounded, grounding_score, gr_reason = self.grounding_check.check_groundedness(
                answer_text=gen_pydantic.raw_answer,
                retrieved_chunks=retrieved_raw
            )
        gr_dur = (time.perf_counter() - gr_start) * 1000

        grounding_pydantic = GroundingResult(
            is_grounded=is_grounded,
            grounding_score=grounding_score,
            reasoning=gr_reason,
            duration_ms=round(gr_dur, 2),
            status="SUCCESS" if is_grounded else "REFUSED"
        )
        stage_timings["grounding_check"] = grounding_pydantic.duration_ms

        # STAGE 7: Output Guardrail Enforcement
        og_start = time.perf_counter()
        if knowledge_mode == "dataset_only":
            out = self.output_guardrail.process_output(
                raw_answer=gen_pydantic.raw_answer,
                safety_passed=True,
                confidence_passed=True,
                grounding_passed=is_grounded,
                refusal_reason=gr_reason if not is_grounded else ""
            )
        else:
            out = {
                "final_answer": gen_pydantic.raw_answer,
                "is_refused": False,
                "refusal_reason": ""
            }
        og_dur = (time.perf_counter() - og_start) * 1000
        stage_timings["output_guardrail"] = round(og_dur, 2)

        total_dur = (time.perf_counter() - pipeline_start) * 1000

        logger.info(f"Pipeline finished in {total_dur:.2f} ms. Refused: {out['is_refused']}")

        return FinalPipelineOutput(
            query_text=transcript,
            final_answer=out["final_answer"],
            is_refused=out["is_refused"],
            refusal_reason=out["refusal_reason"],
            retrieved_chunks=retrieved_chunks,
            stt_result=stt_pydantic,
            input_guardrail=ig_pydantic,
            retrieval_result=retrieval_pydantic,
            generation_result=gen_pydantic,
            grounding_result=grounding_pydantic,
            knowledge_mode=knowledge_mode,
            source_type=source_type,
            total_latency_ms=round(total_dur, 2),
            stage_timings=stage_timings
        )


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("Initializing Orchestrator and testing dual Sarvam + Gemini modes...")
    orchestrator = VoiceRAGPipelineOrchestrator()
    orchestrator.initialize_index()

    # Test 1: Inside dataset
    print("\n--- TEST 1: Inside Dataset Query (Hybrid Mode) ---")
    test_input1 = AudioInput(text_override="What is Calangute Beach famous for?", knowledge_mode="hybrid_auto")
    out1 = orchestrator.run(test_input1)
    print(f"Query: {out1.query_text}")
    print(f"Answer: {out1.final_answer}")
    print(f"Provider: {out1.generation_result.provider} ({out1.generation_result.model}) | Source: {out1.source_type}")

    # Test 2: Outside dataset (World Knowledge)
    print("\n--- TEST 2: Outside Dataset Query (Gemini World Knowledge) ---")
    test_input2 = AudioInput(text_override="Who wrote the Indian National Anthem?", knowledge_mode="hybrid_auto")
    out2 = orchestrator.run(test_input2)
    print(f"Query: {out2.query_text}")
    print(f"Answer: {out2.final_answer}")
    print(f"Provider: {out2.generation_result.provider} ({out2.generation_result.model}) | Source: {out2.source_type}")


if __name__ == "__main__":
    main()
