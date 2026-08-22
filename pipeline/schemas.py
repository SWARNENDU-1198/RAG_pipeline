"""
pipeline/schemas.py

Pydantic Models for Structured I/O at every stage of the Voice RAG Pipeline.
Supports dual Sarvam AI + Google Gemini hybrid and strict dataset modes.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class AudioInput(BaseModel):
    audio_bytes: Optional[bytes] = Field(default=None, description="Raw audio file bytes")
    filename: str = Field(default="audio.wav", description="Audio filename")
    language_code: str = Field(default="hi-IN", description="Language code")
    text_override: Optional[str] = Field(default=None, description="Direct text query override for text testing")
    knowledge_mode: str = Field(default="dataset_only", description="Knowledge Mode: 'dataset_only' | 'hybrid_auto' | 'open_knowledge'")


class STTResult(BaseModel):
    transcript: str = Field(..., description="Transcribed audio text")
    confidence: float = Field(default=0.9, description="STT confidence score")
    provider: str = Field(default="sarvam_ai", description="STT provider used")
    duration_ms: float = Field(..., description="Stage latency in milliseconds")
    status: str = Field(default="SUCCESS", description="Stage status")


class InputGuardrailResult(BaseModel):
    safety_passed: bool = Field(..., description="Pre-retrieval safety check result")
    confidence_passed: bool = Field(..., description="Retrieval score confidence check result")
    top_score: float = Field(default=0.0, description="Highest retrieval similarity score")
    reasoning: str = Field(..., description="Guardrail evaluation details")
    duration_ms: float = Field(..., description="Stage latency in milliseconds")
    status: str = Field(default="SUCCESS", description="Stage status")


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    dense_score: float = 0.0
    sparse_score: float = 0.0
    hybrid_score: float = 0.0
    retrieval_context: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    chunks: List[RetrievedChunk] = Field(default_factory=list)
    top_score: float = Field(default=0.0)
    retrieval_mode: str = Field(default="hybrid")
    duration_ms: float = Field(...)
    status: str = Field(default="SUCCESS")


class GenerationResult(BaseModel):
    raw_answer: str = Field(...)
    provider: str = Field(default="mock_llm")
    model: str = Field(default="mock")
    source_type: str = Field(default="dataset_rag", description="'dataset_rag' | 'gemini_world' | 'hybrid'")
    duration_ms: float = Field(...)
    status: str = Field(default="SUCCESS")


class GroundingResult(BaseModel):
    is_grounded: bool = Field(...)
    grounding_score: float = Field(default=1.0)
    reasoning: str = Field(...)
    duration_ms: float = Field(...)
    status: str = Field(default="SUCCESS")


class FinalPipelineOutput(BaseModel):
    query_text: str
    final_answer: str
    is_refused: bool
    refusal_reason: str
    retrieved_chunks: List[RetrievedChunk]
    stt_result: STTResult
    input_guardrail: InputGuardrailResult
    retrieval_result: RetrievalResult
    generation_result: GenerationResult
    grounding_result: GroundingResult
    knowledge_mode: str = Field(default="hybrid_auto")
    source_type: str = Field(default="dataset_rag")
    total_latency_ms: float
    stage_timings: Dict[str, float]
