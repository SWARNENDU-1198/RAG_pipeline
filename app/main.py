"""
app/main.py

FastAPI backend application serving:
- POST /query: Accepts audio file or text query in any Indian language, executes orchestrator pipeline, returns JSON answer & latency breakdown.
- POST /tts: Synthesizes spoken voice audio for dataset answers using Sarvam AI TTS.
- GET /health: System status check.
- GET /metrics: Latency benchmark percentiles summary.
- Static file serving for web UI frontend.
"""

import os
import sys
import logging
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.orchestrator import VoiceRAGPipelineOrchestrator
from pipeline.schemas import AudioInput

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

orchestrator: VoiceRAGPipelineOrchestrator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator
    logger.info("Initializing Voice RAG Pipeline Orchestrator on server startup...")
    orchestrator = VoiceRAGPipelineOrchestrator()
    orchestrator.initialize_index()
    logger.info("Server startup complete. Voice RAG Pipeline ready for requests.")
    yield
    logger.info("Shutting down Voice RAG Pipeline server...")


app = FastAPI(
    title="Voice-Enabled RAG Pipeline API",
    description="Speech-to-text -> Guardrailed Hybrid FAISS+BM25 Retrieval -> Grounded Answer Generation",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static directory
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    return HTMLResponse("<h2>Voice RAG Pipeline Frontend Index</h2><p>Static index.html building...</p>")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Voice-Enabled RAG Pipeline",
        "index_initialized": orchestrator.is_indexed if orchestrator else False,
        "hybrid_retriever": "FAISS (Dense) + BM25 (Sparse)"
    }


class TTSRequest(BaseModel):
    text: str
    language_code: str = "hi-IN"


@app.post("/tts")
async def synthesize_tts(req: TTSRequest):
    if orchestrator is None:
        raise HTTPException(status_code=500, detail="Pipeline orchestrator not initialized.")

    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    audio_base64 = orchestrator.stt_client.synthesize_speech(
        text=req.text,
        target_language_code=req.language_code
    )

    if not audio_base64:
        return {"audio_base64": None, "message": "TTS fallback to browser speech synthesis."}

    return {
        "audio_base64": audio_base64,
        "format": "audio/wav",
        "provider": "sarvam_ai"
    }


@app.post("/query")
async def process_query(
    audio: UploadFile = File(None),
    query_text: str = Form(None),
    language_code: str = Form("unknown"),
    knowledge_mode: str = Form("dataset_only")
):
    if orchestrator is None:
        raise HTTPException(status_code=500, detail="Pipeline orchestrator not initialized.")

    if not audio and not query_text:
        raise HTTPException(status_code=400, detail="Please provide either an audio file or query_text parameter.")

    audio_bytes = None
    filename = "input_audio.wav"

    if audio:
        filename = audio.filename or "input_audio.wav"
        audio_bytes = await audio.read()

    # Normalize empty or None query_text
    if query_text and not query_text.strip():
        query_text = None

    input_data = AudioInput(
        audio_bytes=audio_bytes,
        filename=filename,
        language_code=language_code if language_code else "unknown",
        text_override=query_text,
        knowledge_mode=knowledge_mode if knowledge_mode else "hybrid_auto"
    )

    try:
        pipeline_output = orchestrator.run(input_data)
        return pipeline_output.model_dump()
    except Exception as e:
        logger.error(f"Error processing pipeline query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline processing error: {str(e)}")


@app.get("/metrics")
async def get_metrics():
    benchmark_md = os.path.join(PROJECT_ROOT, "benchmarking", "results.md")
    if os.path.exists(benchmark_md):
        with open(benchmark_md, "r", encoding="utf-8") as f:
            content = f.read()
            return {"markdown_report": content}
    return {"message": "Run latency_test.py and compute_percentiles.py to generate metrics."}
