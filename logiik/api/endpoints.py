"""
Logiik FastAPI Endpoints.

Extends the existing cognita/api/server.py with new endpoints
for phase monitoring, knowledge base stats, GPU status,
ingestion control, and retrieval stats.

Mounts alongside the legacy server or runs standalone.
All new endpoints are prefixed /logiik/ to avoid collision
with existing cognita routes.

Run standalone:
    uvicorn logiik.api.endpoints:app --host 0.0.0.0 --port 8001
"""
import os
import sys
import subprocess
import asyncio
import time
from pathlib import Path
from typing import Any, Optional, Dict, List
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from logiik.config import CONFIG
from logiik.storage.vector_db import VectorDB
from logiik.storage.text_store import TextStore
from logiik.storage.cache import Cache
from logiik.retrieval.retrieve import Retriever
from logiik.utils.logging import get_logger

logger = get_logger("api.endpoints")

app = FastAPI(
    title="Logiik API",
    version="0.1.0",
    description="Logiik knowledge retrieval and training monitor API."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CONFIG.get("api", {}).get("cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Lazy-initialised singletons ─────────────────────────────────────────────
# Initialised on first request to avoid startup failures when
# credentials are not yet configured.

_vector_db: Optional[VectorDB] = None
_text_store: Optional[TextStore] = None
_cache: Optional[Cache] = None
_retriever: Optional[Retriever] = None

# Phase monitor state — populated by training loop via /logiik/phase/update
_phase_metrics: Dict = {
    "phase_id": 0,
    "phase": "Not started",
    "coverage_ratio": 0.0,
    "saturation_score": 0.0,
    "covered_prompts": 0,
    "total_prompts": 0,
    "is_complete": False,
    "iteration": 0,
    "last_updated": None,
}

# Training metrics state
_training_metrics: Dict = {
    "current_phase": "Not started",
    "step": 0,
    "loss": None,
    "val_loss": None,
    "val_perplexity": None,
    "generative_ratio": None,
    "examples_processed": 0,
    "bank_count": 0,
    "toss_count": 0,
    "training_active": False,
    "training_complete": False,
    "last_updated": None,
}

# Ingestion stats state
_ingestion_stats: Dict = {
    "pdfs_processed": 0,
    "chunks_new": 0,
    "chunks_duplicate": 0,
    "images_new": 0,
    "images_duplicate": 0,
    "mode": "lite_mode",
    "last_updated": None,
}

# Retrieval stats state
_retrieval_stats: Dict = {
    "total_queries": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "avg_latency_ms": 0.0,
    "last_updated": None,
}


def _get_db() -> VectorDB:
    global _vector_db
    if _vector_db is None:
        _vector_db = VectorDB()
    return _vector_db


def _get_store() -> TextStore:
    global _text_store
    if _text_store is None:
        _text_store = TextStore()
    return _text_store


def _get_cache() -> Cache:
    global _cache
    if _cache is None:
        _cache = Cache()
    return _cache


def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever(
            vector_db=_get_db(),
            text_store=_get_store(),
            cache=_get_cache(),
        )
    return _retriever


# ─── Request / Response models ────────────────────────────────────────────────

class PhaseMetricsUpdate(BaseModel):
    phase_id: int
    phase: str
    coverage_ratio: float
    saturation_score: float
    covered_prompts: int
    total_prompts: int
    is_complete: bool
    iteration: int


class TrainingMetricsUpdate(BaseModel):
    current_phase: str
    step: int
    loss: Optional[float] = None
    val_loss: Optional[float] = None
    val_perplexity: Optional[float] = None
    generative_ratio: Optional[float] = None
    examples_processed: int = 0
    bank_count: int = 0
    toss_count: int = 0
    training_active: bool = False
    training_complete: bool = False


class IngestionStatsUpdate(BaseModel):
    pdfs_processed: int
    chunks_new: int
    chunks_duplicate: int
    images_new: int
    images_duplicate: int
    mode: str = "lite_mode"


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    phase_filter: Optional[str] = None
    min_score: float = 0.0


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/logiik/health")
async def health():
    """API health check."""
    return {
        "status": "ok",
        "version": "0.1.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


# ─── Phase metrics ────────────────────────────────────────────────────────────

@app.get("/logiik/phase_metrics")
async def get_phase_metrics():
    """
    Current phase completion monitor metrics.
    Updated by training loop via POST /logiik/phase/update.
    """
    return _phase_metrics


@app.post("/logiik/phase/update")
async def update_phase_metrics(update: PhaseMetricsUpdate):
    """
    Called by training loop after each student answer to push
    latest coverage + saturation metrics to the dashboard.
    """
    global _phase_metrics
    _phase_metrics = {
        **update.model_dump(),
        "last_updated": datetime.utcnow().isoformat(),
    }
    return {"status": "updated"}


# ─── Training metrics ─────────────────────────────────────────────────────────

@app.get("/logiik/training_metrics")
async def get_training_metrics():
    """
    Current training loop metrics: loss, val_loss, phase,
    step, bank/toss counts.
    """
    return _training_metrics


@app.post("/logiik/training/update")
async def update_training_metrics(update: TrainingMetricsUpdate):
    """Called by training loop to push latest metrics."""
    global _training_metrics
    _training_metrics = {
        **update.model_dump(),
        "last_updated": datetime.utcnow().isoformat(),
    }
    return {"status": "updated"}


# ─── Knowledge base stats ─────────────────────────────────────────────────────

@app.get("/logiik/knowledge_stats")
async def get_knowledge_stats():
    """
    Knowledge base statistics from Pinecone + Firebase.
    Includes vector count, embedding metadata, and ingestion log.
    """
    try:
        db_stats = _get_db().stats()
        store_summary = _get_store().get_summary()
        return {
            "vector_db": db_stats,
            "firebase": store_summary,
            "ingestion": _ingestion_stats,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/logiik/ingestion/update")
async def update_ingestion_stats(update: IngestionStatsUpdate):
    """Called by Phase 9 pipeline after each PDF to push stats."""
    global _ingestion_stats
    _ingestion_stats = {
        **update.model_dump(),
        "last_updated": datetime.utcnow().isoformat(),
    }
    return {"status": "updated"}


# ─── GPU status ───────────────────────────────────────────────────────────────

@app.get("/logiik/gpu_status")
async def get_gpu_status():
    """
    Real-time GPU memory and utilisation stats.
    Returns CPU-mode indicator if no GPU present.
    """
    status = {
        "gpu_available": False,
        "device_name": "CPU",
        "vram_total_gb": 0.0,
        "vram_used_gb": 0.0,
        "vram_free_gb": 0.0,
        "utilisation_pct": 0.0,
        "temperature_c": None,
        "timestamp": datetime.utcnow().isoformat(),
    }

    # PyTorch VRAM stats
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total = props.total_memory / 1e9
            used = torch.cuda.memory_allocated(0) / 1e9
            free = total - used
            status.update({
                "gpu_available": True,
                "device_name": props.name,
                "vram_total_gb": round(total, 2),
                "vram_used_gb": round(used, 2),
                "vram_free_gb": round(free, 2),
            })
    except Exception:
        pass

    # nvidia-smi for utilisation + temperature
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) >= 2:
                status["utilisation_pct"] = float(parts[0].strip())
                status["temperature_c"] = float(parts[1].strip())
    except Exception:
        pass

    return status


# ─── Teacher-Student loop state ───────────────────────────────────────────────
# Holds in-memory loop instances for Phase 6 and Phase 11.
# Populated on first POST to /logiik/ts/{phase}/question.

_ts_loops: Dict[int, Any] = {}


def _get_ts_loop(phase_id: int):
    """
    Return or create a Phase7TeacherStudentLoop for the
    given phase. Phase 6 and Phase 11 both use this loop
    mechanism with different correctness thresholds.
    """
    if phase_id not in _ts_loops:
        from logiik.core.training import Phase7TeacherStudentLoop
        _ts_loops[phase_id] = Phase7TeacherStudentLoop(
            text_store=_get_store()
        )
    return _ts_loops[phase_id]


class TSQuestionRequest(BaseModel):
    prompt: str
    answer_steps: List[str]
    full_answer: str
    difficulty: float = 0.93


class TSFeedbackRequest(BaseModel):
    question_id: str
    attempt_index: int
    feedback: List[str]
    correctness: float
    suggested_improvement: Optional[str] = None


class TSAttemptRequest(BaseModel):
    question_id: str
    student_answer_steps: List[str]
    student_full_answer: str


@app.post("/logiik/ts/{phase_id}/question")
async def ts_register_question(
    phase_id: int, request: TSQuestionRequest
):
    """
    Register a teacher-generated question for the
    Phase 6 or Phase 11 teacher-student loop.

    Args (path): phase_id — 6 or 11
    Args (body): prompt, answer_steps, full_answer, difficulty

    Returns: question_id (UUID string)
    """
    if phase_id not in (6, 11):
        raise HTTPException(
            status_code=400,
            detail="Teacher-student loop only active for "
                   "phases 6 and 11."
        )
    loop = _get_ts_loop(phase_id)
    qid = loop.generate_teacher_question(
        prompt=request.prompt,
        answer_steps=request.answer_steps,
        full_answer=request.full_answer,
        difficulty=request.difficulty,
    )
    logger.info(
        f"Phase {phase_id} T-S question registered: {qid}"
    )
    return {"question_id": qid, "phase_id": phase_id}


@app.post("/logiik/ts/{phase_id}/attempt")
async def ts_student_attempt(
    phase_id: int, request: TSAttemptRequest
):
    """
    Record a student attempt for a Phase 6 or Phase 11
    teacher-student question.

    Args (path): phase_id — 6 or 11
    Args (body): question_id, student_answer_steps,
                 student_full_answer

    Returns: attempt_index
    """
    if phase_id not in (6, 11):
        raise HTTPException(
            status_code=400,
            detail="Teacher-student loop only active for "
                   "phases 6 and 11."
        )
    loop = _get_ts_loop(phase_id)
    attempt = loop.student_attempt(
        question_id=request.question_id,
        student_answer_steps=request.student_answer_steps,
        student_full_answer=request.student_full_answer,
    )
    return {
        "question_id": request.question_id,
        "attempt_index": attempt.attempt_index,
        "phase_id": phase_id,
    }


@app.post("/logiik/ts/{phase_id}/feedback")
async def ts_provide_feedback(
    phase_id: int, request: TSFeedbackRequest
):
    """
    Apply teacher feedback to a student attempt.

    Args (path): phase_id — 6 or 11
    Args (body): question_id, attempt_index, feedback,
                 correctness, suggested_improvement

    Returns:
        threshold_met: bool — True if correctness >= threshold
        correctness:   float
        phase_id:      int
    """
    if phase_id not in (6, 11):
        raise HTTPException(
            status_code=400,
            detail="Teacher-student loop only active for "
                   "phases 6 and 11."
        )
    from logiik.curriculum.phases import get_phase
    phase = get_phase(phase_id)
    threshold = (
        phase.correctness_threshold if phase else 0.90
    )

    loop = _get_ts_loop(phase_id)
    threshold_met = loop.provide_feedback(
        question_id=request.question_id,
        attempt_index=request.attempt_index,
        feedback=request.feedback,
        correctness=request.correctness,
        suggested_improvement=request.suggested_improvement,
    )
    return {
        "threshold_met": threshold_met,
        "correctness": request.correctness,
        "threshold": threshold,
        "phase_id": phase_id,
    }


@app.get("/logiik/ts/{phase_id}/metrics")
async def ts_get_metrics(phase_id: int):
    """
    Return aggregate teacher-student loop metrics
    for Phase 6 or Phase 11.
    """
    if phase_id not in (6, 11):
        raise HTTPException(
            status_code=400,
            detail="Teacher-student loop only active for "
                   "phases 6 and 11."
        )
    loop = _get_ts_loop(phase_id)
    return {
        "phase_id": phase_id,
        "metrics": loop.get_metrics(),
    }


# ─── Curriculum ───────────────────────────────────────────────────────────────

@app.get("/logiik/curriculum")
async def get_curriculum():
    """
    Return full 12-phase curriculum structure grouped by track.
    Used by dashboard Phase Monitor tab.
    """
    from logiik.curriculum.phases import PHASES, get_phases_by_track
    tracks = [
        "foundation", "language", "domain",
        "execution", "integration", "capstone"
    ]
    track_colors = {
        "foundation": "#3b82f6",
        "language":   "#22c55e",
        "domain":     "#a855f7",
        "execution":  "#f97316",
        "integration":"#eab308",
        "capstone":   "#ef4444",
    }
    result = {}
    for track in tracks:
        phases = get_phases_by_track(track)
        result[track] = {
            "color": track_colors[track],
            "phases": [
                {
                    "id": p.id,
                    "name": p.name,
                    "display_name": p.display_name,
                    "generative_ratio": p.generative_ratio,
                    "correctness_threshold": p.correctness_threshold,
                    "teacher_student": p.teacher_student,
                    "duration": p.duration,
                }
                for p in phases
            ]
        }
    return {
        "total_phases": len(PHASES),
        "tracks": result,
        "timestamp": datetime.utcnow().isoformat(),
    }


# ─── Retrieval stats ──────────────────────────────────────────────────────────

@app.get("/logiik/retrieval_stats")
async def get_retrieval_stats():
    """Cache hit rates, query counts, and vector DB backend info."""
    retriever_stats = _get_retriever().stats()
    return {
        **_retrieval_stats,
        **retriever_stats,
        "timestamp": datetime.utcnow().isoformat(),
    }


# ─── Query endpoint ───────────────────────────────────────────────────────────

@app.post("/logiik/query")
async def query_knowledge(request: QueryRequest):
    """
    Retrieve relevant knowledge chunks for a query.
    Returns top-k chunks with scores, sources, and assembled context.

    Args (JSON body):
        query:        Natural language query string.
        top_k:        Number of results (default 5).
        phase_filter: Optional phase name to filter results
                      e.g. 'phase9' or 'phase8'.
        min_score:    Minimum similarity threshold (default 0.0).
    """
    global _retrieval_stats

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    start = asyncio.get_event_loop().time()

    try:
        filter_dict = None
        if request.phase_filter:
            filter_dict = {"phase": {"$eq": request.phase_filter}}

        retriever = _get_retriever()
        chunks = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            filter=filter_dict,
            min_score=request.min_score,
        )
        context = retriever.build_context(
            query=request.query,
            top_k=request.top_k,
            filter=filter_dict,
        )

        latency_ms = (asyncio.get_event_loop().time() - start) * 1000
        cache_hits = sum(1 for c in chunks if c.cache_hit)

        # Update retrieval stats
        _retrieval_stats["total_queries"] += 1
        _retrieval_stats["cache_hits"] += cache_hits
        _retrieval_stats["cache_misses"] += len(chunks) - cache_hits
        # Rolling average latency
        prev_avg = _retrieval_stats["avg_latency_ms"]
        total_q = _retrieval_stats["total_queries"]
        _retrieval_stats["avg_latency_ms"] = round(
            (prev_avg * (total_q - 1) + latency_ms) / total_q, 2
        )
        _retrieval_stats["last_updated"] = datetime.utcnow().isoformat()

        return {
            "query": request.query,
            "results": [
                {
                    "id": c.id,
                    "text": c.text,
                    "score": round(c.score, 4),
                    "source": c.source,
                    "metadata": c.metadata,
                    "cache_hit": c.cache_hit,
                }
                for c in chunks
            ],
            "context": context,
            "latency_ms": round(latency_ms, 2),
            "cache_hits": cache_hits,
            "total_results": len(chunks),
        }

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Q&A generation training loop ────────────────────────────────────────────

_qa_task: Optional[asyncio.Task] = None
_qa_stop_flag: bool = False
_TEACHER_CFG = Path(__file__).parents[2] / "configs" / "teacher_config.yaml"


def _load_curriculum_topics() -> list:
    """Load all topics from teacher_config.yaml nlp_curriculum, flat list."""
    try:
        import yaml
        with open(_TEACHER_CFG) as f:
            cfg = yaml.safe_load(f)
        phases = cfg.get("nlp_curriculum", {})
        topics = []
        for phase_topics in phases.values():
            if isinstance(phase_topics, list):
                topics.extend(phase_topics)
        return topics
    except Exception as e:
        logger.warning("Could not load curriculum topics: %s", e)
        return ["NLP fundamentals", "language model training", "text classification"]


async def _qa_generation_loop(api_key: str, base_url: str, model_id: str):
    """Background task: cycle through curriculum topics generating Q&A pairs."""
    global _qa_stop_flag, _training_metrics

    import openai as _openai
    import json as _json

    client = _openai.OpenAI(api_key=api_key, base_url=base_url)
    topics = _load_curriculum_topics()
    total = len(topics)
    examples_done = 0
    phase_idx = 0

    _training_metrics["training_active"] = True
    _training_metrics["training_complete"] = False
    _training_metrics["current_phase"] = topics[0] if topics else "generating"
    _training_metrics["last_updated"] = datetime.utcnow().isoformat()
    logger.info("Q&A generation started — %d topics", total)

    try:
        while not _qa_stop_flag:
            topic = topics[phase_idx % total]
            difficulty = min(0.1 + (examples_done / max(total * 3, 1)) * 0.8, 0.9)

            prompt = (
                f"Generate a training Q&A example about: {topic}\n"
                f"Difficulty: {difficulty*100:.0f}%\n\n"
                "Respond with ONLY a JSON object — no markdown, no explanation — with these keys:\n"
                '{"question":"...","answers":["...","...","...","...","..."],'
                '"correct_indices":[0],"explanation":"...","domain":"..."}'
            )
            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, lambda: client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": "You are a teacher AI. Always respond with valid JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=1,
                    max_tokens=600,
                ))
                raw = (response.choices[0].message.content or "").strip()
                # Strip markdown code fences if present
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                raw = raw.strip()
                data = _json.loads(raw)
                examples_done += 1
                phase_idx += 1

                _training_metrics["examples_processed"] = examples_done
                _training_metrics["bank_count"] = examples_done
                _training_metrics["step"] = examples_done
                _training_metrics["current_phase"] = data.get("domain", topic)
                _training_metrics["last_updated"] = datetime.utcnow().isoformat()
                logger.info("Generated Q&A #%d — %s", examples_done, topic[:60])

            except Exception as e:
                err = str(e)
                logger.warning("Q&A generation error (topic=%s): %s", topic, err)
                if "429" in err or "rate_limit" in err.lower():
                    await asyncio.sleep(10)
                else:
                    await asyncio.sleep(3)
                continue

            await asyncio.sleep(3)  # stay within rate limits

    finally:
        _training_metrics["training_active"] = False
        _training_metrics["last_updated"] = datetime.utcnow().isoformat()
        logger.info("Q&A generation stopped — %d examples generated", examples_done)


class TrainingStartRequest(BaseModel):
    api_key: str
    base_url: str
    model_id: str


@app.post("/train/start")
async def start_training(req: TrainingStartRequest):
    """Start the Q&A generation loop in the background."""
    global _qa_task, _qa_stop_flag

    if _qa_task and not _qa_task.done():
        return {"status": "already_running"}

    _qa_stop_flag = False
    _qa_task = asyncio.create_task(
        _qa_generation_loop(req.api_key, req.base_url, req.model_id)
    )
    return {"status": "training_started"}


@app.post("/train/stop")
async def stop_training():
    """Stop the Q&A generation loop."""
    global _qa_task, _qa_stop_flag

    _qa_stop_flag = True
    if _qa_task and not _qa_task.done():
        _qa_task.cancel()
        try:
            await _qa_task
        except asyncio.CancelledError:
            pass
    _qa_task = None
    _training_metrics["training_active"] = False
    logger.info("Training stopped by user")
    return {"status": "training_stopped"}


# ─── Static dashboard ─────────────────────────────────────────────────────────

_dashboard_path = Path(__file__).parents[1] / "dashboard"
if _dashboard_path.exists():
    app.mount(
        "/",
        StaticFiles(directory=str(_dashboard_path), html=True),
        name="logiik_dashboard"
    )


if __name__ == "__main__":
    import uvicorn
    port = CONFIG.get("api", {}).get("port", 8001)
    uvicorn.run(app, host="0.0.0.0", port=port)
