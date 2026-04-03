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
_TRAINING_DATA_DIR = Path(__file__).parents[2] / "knowledge_base" / "training_data"


# ── Saturation helpers ────────────────────────────────────────────────────────

def _word_set(text: str) -> set:
    return set(text.lower().split())


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _saturation_score(recent_questions: List[str], pool: List[str]) -> float:
    """
    Average of each recent question's max Jaccard similarity against the pool.
    High score (→1.0) means new questions are very similar to existing ones = saturated.
    Low score (→0.0) means new questions are novel.
    """
    if not pool or not recent_questions:
        return 0.0
    pool_sets = [_word_set(q) for q in pool]
    scores = []
    for q in recent_questions:
        q_set = _word_set(q)
        scores.append(max((_jaccard(q_set, p) for p in pool_sets), default=0.0))
    return sum(scores) / len(scores)


# ── Per-track generation limits ───────────────────────────────────────────────

_TRACK_LIMITS = {
    "foundation":  {"min": 10, "max": 20},   # diverse facts, not drills
    "language":    {"min": 15, "max": 25},
    "domain":      {"min": 20, "max": 35},
    "execution":   {"min": 15, "max": 25},
    "integration": {"min": 20, "max": 35},
    "capstone":    {"min": 25, "max": 40},
}

# ── Deduplication ─────────────────────────────────────────────────────────────

_DEDUP_THRESHOLD = 0.65  # Jaccard — skip generation if new Q is this similar to any in pool

def _is_duplicate(question: str, pool: List[str]) -> bool:
    """Return True if question is too similar to any question already in pool."""
    if not pool:
        return False
    q_set = _word_set(question)
    return any(_jaccard(q_set, _word_set(p)) >= _DEDUP_THRESHOLD for p in pool)


# ── Phase-aware prompt construction ──────────────────────────────────────────

def _phase_prompt(phase_name: str, track: str, description: str,
                  topic: str, q_type: str, difficulty: float) -> tuple:
    """
    Build (system_msg, user_msg, expected_json_schema) for the given phase track.
    Returns a 3-tuple: (sys_msg, user_msg, schema_hint).
    """
    diff_pct = f"{difficulty * 100:.0f}%"
    q_type_readable = q_type.replace("_", " ")

    if track == "foundation":
        if "memorization" in phase_name:
            sys_msg = (
                "You are an expert scientific educator generating high-quality "
                "supervised fine-tuning data. Each question must test a DIFFERENT "
                "specific fact — never rephrase a question angle already used."
            )
            user_msg = (
                f"Generate a '{q_type_readable}' multiple-choice question about: {topic}\n"
                f"Difficulty: {diff_pct}\n\n"
                "Requirements:\n"
                "- Test ONE precise, specific fact (not a vague overview question)\n"
                "- Wrong answers must be plausible but unambiguously incorrect to an expert\n"
                "- Approach the topic from a fresh angle not yet covered\n\n"
                "Respond with ONLY raw JSON:\n"
                '{"question":"...","answers":["A","B","C","D","E"],'
                '"correct_indices":[0],"explanation":"...","domain":"..."}'
            )
            schema = "mcq"
        else:  # generation phase
            sys_msg = (
                "You are an expert scientific educator generating open-ended "
                "fine-tuning data. Answers must be original and substantive, "
                "not a rephrasing of the question."
            )
            user_msg = (
                f"Generate a '{q_type_readable}' open-ended question about: {topic}\n"
                f"Difficulty: {diff_pct}\n\n"
                "Requirements:\n"
                "- No multiple choice — requires a generated explanation\n"
                "- Answer: 2-4 sentences of precise scientific language\n"
                "- Explanation: expand on the mechanism or principle\n\n"
                "Respond with ONLY raw JSON:\n"
                '{"question":"...","answer":"...","explanation":"...","domain":"..."}'
            )
            schema = "open"

    elif track == "language":
        sys_msg = (
            "You are a scientific writing and reasoning expert generating training data "
            "that teaches precise scientific language, statistical interpretation, "
            "and research methodology."
        )
        user_msg = (
            f"Generate a '{q_type_readable}' question about: {topic}\n"
            f"Difficulty: {diff_pct}\n"
            f"Phase goal: {description}\n\n"
            "Requirements:\n"
            "- Use precise scientific register throughout\n"
            "- Provide step-by-step reasoning in the answer\n"
            "- Ground every claim in a specific principle, statistical concept, or mechanism\n\n"
            "Respond with ONLY raw JSON:\n"
            '{"question":"...","answer":"...","reasoning_steps":["step 1","step 2"],"domain":"..."}'
        )
        schema = "reasoning"

    elif track == "domain":
        sys_msg = (
            "You are a domain expert generating training data for deep scientific reasoning. "
            "Every answer must cite specific mechanisms, molecular pathways, or physical "
            "principles — no generic placeholders allowed."
        )
        user_msg = (
            f"Generate a '{q_type_readable}' question about: {topic}\n"
            f"Difficulty: {diff_pct}\n"
            f"Phase goal: {description}\n\n"
            "Requirements:\n"
            "- Name the specific mechanism, pathway, protein, or equation involved\n"
            "- Stepwise reasoning required — do not jump to conclusions\n"
            "- Identify at least one confounder, limitation, or assumption\n\n"
            "Respond with ONLY raw JSON:\n"
            '{"question":"...","answer":"...","reasoning_steps":["step 1","step 2"],'
            '"mechanisms":["specific mechanism"],"caveats":["limitation"],"domain":"..."}'
        )
        schema = "domain"

    elif track == "execution":
        if "coding" in phase_name or "computing" in phase_name:
            sys_msg = (
                "You are a scientific computing expert generating training data for "
                "research-grade code generation. Every task must be grounded in a real "
                "scientific research workflow — no toy examples."
            )
            user_msg = (
                f"Generate a '{q_type_readable}' coding task about: {topic}\n"
                f"Difficulty: {diff_pct}\n\n"
                "Requirements:\n"
                "- Real scientific research context with realistic data/constraints\n"
                "- Complete, runnable code with comments\n"
                "- Explain what each section does and why\n\n"
                "Respond with ONLY raw JSON:\n"
                '{"question":"...","code":"...","explanation":"...","language":"python","domain":"..."}'
            )
            schema = "code"
        else:
            sys_msg = "You are an expert in research engineering, reliability, and failure analysis."
            user_msg = (
                f"Generate a '{q_type_readable}' scenario about: {topic}\n"
                f"Difficulty: {diff_pct}\n"
                f"Phase goal: {description}\n\n"
                "Respond with ONLY raw JSON:\n"
                '{"question":"...","answer":"...","reasoning_steps":["step 1"],"domain":"..."}'
            )
            schema = "reasoning"

    elif track in ("integration", "capstone"):
        sys_msg = (
            "You are a scientific reasoning expert generating training data for high-level "
            "synthesis and judgment under uncertainty. Answers must connect mechanisms across "
            "domains and express calibrated confidence — not false certainty."
        )
        user_msg = (
            f"Generate a '{q_type_readable}' question about: {topic}\n"
            f"Difficulty: {diff_pct}\n"
            f"Phase goal: {description}\n\n"
            "Requirements:\n"
            "- Connect mechanisms or findings across at least two scientific domains\n"
            "- Express calibrated confidence (0.0–1.0) — uncertainty is a feature, not a flaw\n"
            "- Explicitly state what is unknown or contested\n"
            "- Reasoning chain must be traceable step by step\n\n"
            "Respond with ONLY raw JSON:\n"
            '{"question":"...","answer":"...","reasoning_chain":["step 1","step 2"],'
            '"confidence":0.80,"caveats":["what is unknown"],"domain":"..."}'
        )
        schema = "synthesis"

    else:
        sys_msg = "You are an expert teacher generating scientific training data."
        user_msg = (
            f"Generate a '{q_type_readable}' question about: {topic}\n"
            f"Difficulty: {diff_pct}\n\n"
            "Respond with ONLY raw JSON:\n"
            '{"question":"...","answer":"...","explanation":"...","domain":"..."}'
        )
        schema = "open"

    return sys_msg, user_msg, schema


def _build_completion(data: dict, schema: str) -> str:
    """Convert parsed API response into a clean completion string for training."""
    if schema == "mcq":
        answers      = data.get("answers", [])
        correct_idxs = data.get("correct_indices", [0])
        correct_text = [answers[i] for i in correct_idxs if i < len(answers)]
        return "\n".join(correct_text) + "\n\nExplanation: " + data.get("explanation", "")

    if schema == "code":
        return (
            "```" + data.get("language", "python") + "\n"
            + data.get("code", "") + "\n```"
            + "\n\n" + data.get("explanation", "")
        )

    # All other schemas have an "answer" field + optional structured extras
    result = data.get("answer", data.get("completion", ""))

    steps = data.get("reasoning_steps") or data.get("reasoning_chain") or []
    if steps:
        result += "\n\nReasoning:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))

    mechanisms = data.get("mechanisms", [])
    if mechanisms:
        result += "\n\nKey mechanisms:\n" + "\n".join(f"- {m}" for m in mechanisms)

    caveats = data.get("caveats", [])
    if caveats:
        result += "\n\nCaveats / limitations:\n" + "\n".join(f"- {c}" for c in caveats)

    confidence = data.get("confidence")
    if confidence is not None:
        result += f"\n\nConfidence: {float(confidence):.0%}"

    return result


# ── Curriculum + phase config loaders ────────────────────────────────────────

def _load_curriculum_phases() -> tuple:
    """
    Load nlp_curriculum and generation settings from teacher_config.yaml.
    Returns (phase_list, min_per_topic, max_per_topic, check_interval).
    phase_list: [(phase_id, phase_name, [topics]), ...]
    """
    try:
        import yaml
        with open(_TEACHER_CFG) as f:
            cfg = yaml.safe_load(f)
        curriculum = cfg.get("nlp_curriculum", {})
        cur_cfg = cfg.get("teacher", {}).get("curriculum", {})
        min_per_topic     = cur_cfg.get("min_examples_per_topic", 50)
        max_per_topic     = cur_cfg.get("max_examples_per_topic", 500)
        check_interval    = cur_cfg.get("saturation_check_interval", 20)
        phase_list = [
            (idx + 1, name, topics)
            for idx, (name, topics) in enumerate(curriculum.items())
            if isinstance(topics, list)
        ]
        return phase_list, min_per_topic, max_per_topic, check_interval
    except Exception as e:
        logger.warning("Could not load curriculum: %s", e)
        return [(1, "nlp_fundamentals", ["language model training"])], 50, 500, 20


def _load_phase_criteria() -> dict:
    """
    Load completion_criteria from phases.py keyed by phase name.
    Falls back to safe defaults if import fails.
    """
    defaults = {"coverage_ratio": 0.95, "saturation_score": 0.90}
    try:
        from logiik.curriculum.phases import PHASES
        return {p.name: p.completion_criteria for p in PHASES}
    except Exception as e:
        logger.warning("Could not load phase configs: %s", e)
        return {}


def _match_phase_criteria(phase_name: str, criteria_map: dict) -> dict:
    """Find completion criteria for a phase name, with fuzzy fallback."""
    if phase_name in criteria_map:
        return criteria_map[phase_name]
    # fuzzy: find the phase whose name overlaps most with ours
    clean = phase_name.lower().replace(" ", "_").replace("&", "and")
    for key in criteria_map:
        if key in clean or clean in key:
            return criteria_map[key]
    return {"coverage_ratio": 0.95, "saturation_score": 0.90}


def _jsonl_path(phase_id: int, phase_name: str) -> Path:  # noqa: E302
    """Return the JSONL file path for a phase (sft_trainer naming convention)."""
    _TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return _TRAINING_DATA_DIR / f"phase_{phase_id:02d}_{phase_name}.jsonl"


def _append_record(path: Path, record: dict) -> None:
    """Append one JSON record as a line to a JSONL file."""
    import json as _json
    with open(path, "a", encoding="utf-8") as f:
        f.write(_json.dumps(record, ensure_ascii=False) + "\n")


def _count_existing(path: Path) -> int:
    """Count lines already written to a JSONL file (for resume support)."""
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


async def _qa_generation_loop(api_key: str, base_url: str, model_id: str):
    """
    Self-determining Q&A generation loop.

    For each phase → each topic:
      - Generates questions continuously, saving each to JSONL on disk
      - After min_examples_per_topic, checks saturation every check_interval
      - Saturation = average max-Jaccard of recent questions vs. the full pool
      - Topic is done when saturation >= phase threshold OR count >= max cap
    Phase advances when coverage_ratio of saturated topics >= phase threshold.
    All thresholds come from phases.py completion_criteria — no hardcoding.

    Files: knowledge_base/training_data/phase_01_memorization.jsonl etc.
    Resume-safe: counts existing lines, reloads questions from disk to continue
    saturation tracking where it left off.
    """
    global _qa_stop_flag, _training_metrics, _phase_metrics

    import openai as _openai
    import json as _json

    client = _openai.OpenAI(api_key=api_key, base_url=base_url)
    _fb_store = TextStore()
    phases, min_per_topic, max_per_topic, check_interval = _load_curriculum_phases()
    criteria_map = _load_phase_criteria()
    total_phases = len(phases)

    total_saved = sum(
        _count_existing(_jsonl_path(pid, pname))
        for pid, pname, _ in phases
    )

    _training_metrics.update({
        "training_active": True, "training_complete": False,
        "examples_processed": total_saved, "bank_count": total_saved,
        "step": total_saved,
        "current_phase": phases[0][1] if phases else "generating",
        "last_updated": datetime.utcnow().isoformat(),
    })
    logger.info(
        "Q&A generation started — %d phases | min %d / max %d per topic | %d already on disk",
        total_phases, min_per_topic, max_per_topic, total_saved,
    )

    async def _call_teacher(phase_cfg, topic: str, q_type: str, difficulty: float) -> tuple:
        """Returns (parsed_data_dict, schema_str)."""
        sys_msg, user_msg, schema = _phase_prompt(
            phase_cfg.name, phase_cfg.track, phase_cfg.description,
            topic, q_type, difficulty,
        )
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user",   "content": user_msg},
        ]
        loop = asyncio.get_event_loop()

        def _do_call(use_json_format: bool):
            kwargs = dict(model=model_id, messages=messages, temperature=1, max_tokens=2000)
            if use_json_format:
                kwargs["response_format"] = {"type": "json_object"}
            return client.chat.completions.create(**kwargs)

        try:
            response = await loop.run_in_executor(None, lambda: _do_call(True))
        except Exception as e:
            if "response_format" in str(e).lower() or "400" in str(e):
                response = await loop.run_in_executor(None, lambda: _do_call(False))
            else:
                raise

        raw = (response.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return _json.loads(raw.strip()), schema

    def _read_questions_from_disk(path: Path) -> List[str]:
        """Reload question texts already saved (for saturation tracking on resume)."""
        if not path.exists():
            return []
        questions = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        questions.append(_json.loads(line).get("prompt", ""))
                    except Exception:
                        pass
        return questions

    _backoff_counts: dict = {}

    from logiik.curriculum.phases import get_phase as _get_phase

    try:
        for phase_id, phase_name, topics in phases:
            if _qa_stop_flag:
                break

            phase_cfg = _get_phase(phase_id)
            track     = phase_cfg.track if phase_cfg else "foundation"
            q_types   = (phase_cfg.question_types if phase_cfg else ["factual_recall"])
            limits    = _TRACK_LIMITS.get(track, {"min": 15, "max": 25})
            phase_min = limits["min"]
            phase_max = limits["max"]

            criteria = _match_phase_criteria(phase_name, criteria_map)
            phase_coverage_threshold  = criteria.get("coverage_ratio", 0.95)
            phase_sat_threshold       = criteria.get("saturation_score", 0.90)
            max_iter                  = criteria.get("max_iterations", None)

            out_path   = _jsonl_path(phase_id, phase_name)
            topic_count = len(topics)
            phase_saved = _count_existing(out_path)

            logger.info(
                "Phase %d/%d: %s | %d topics | coverage≥%.0f%% sat≥%.0f%% | %d on disk",
                phase_id, total_phases, phase_name, topic_count,
                phase_coverage_threshold * 100, phase_sat_threshold * 100, phase_saved,
            )
            _training_metrics["current_phase"] = phase_name

            # Re-read existing questions per topic for saturation continuity on resume
            all_disk_questions = _read_questions_from_disk(out_path)

            saturated_topics = 0
            topics_per_q_count: List[int] = []  # examples generated per topic

            for t_idx, topic in enumerate(topics):
                if _qa_stop_flag:
                    break

                # Estimate how many examples are already on disk for this topic
                # (approximate — stored sequentially, so divide equally)
                slice_start = sum(topics_per_q_count) if topics_per_q_count else 0
                topic_pool: List[str] = all_disk_questions[slice_start:]

                count_this_topic = len(topic_pool)
                topic_saturated  = False
                current_sat      = 0.0
                q_type_idx       = count_this_topic % len(q_types)  # resume rotation

                logger.info(
                    "  Topic %d/%d [%s]: %s  (%d on disk, max %d)",
                    t_idx + 1, topic_count, track, topic[:50], count_this_topic, phase_max,
                )

                ex_idx = count_this_topic  # continue from where we left off

                while not _qa_stop_flag:
                    # ── Hard cap (phase-track-specific) ───────────────────
                    if ex_idx >= phase_max:
                        logger.info(
                            "  Topic '%s' hit phase cap (%d) — moving on", topic[:40], phase_max
                        )
                        topic_saturated = True
                        break

                    # ── Saturation check (after minimum floor) ────────────
                    if ex_idx >= phase_min and ex_idx % check_interval == 0:
                        recent = topic_pool[-check_interval:]
                        prior  = topic_pool[:-check_interval]
                        current_sat = _saturation_score(recent, prior) if prior else 0.0
                        logger.info(
                            "  Saturation check — topic '%s': %.3f (threshold %.2f) @ %d examples",
                            topic[:40], current_sat, phase_sat_threshold, ex_idx,
                        )
                        if current_sat >= phase_sat_threshold:
                            logger.info(
                                "  Topic '%s' SATURATED at %d examples (sat=%.3f)",
                                topic[:40], ex_idx, current_sat,
                            )
                            topic_saturated = True
                            break

                    # ── Generate one Q&A ──────────────────────────────────
                    q_type     = q_types[q_type_idx % len(q_types)]
                    difficulty = round(
                        0.1 + min(ex_idx / max(phase_min * 4, 1), 1.0) * 0.8, 2
                    )
                    try:
                        data, schema = await _call_teacher(phase_cfg, topic, q_type, difficulty)

                        question = data.get("question", "")
                        if not question:
                            continue

                        # ── Deduplication: skip near-identical questions ───
                        if _is_duplicate(question, topic_pool):
                            logger.debug(
                                "  Skipped duplicate question for topic '%s'", topic[:40]
                            )
                            continue

                        completion = _build_completion(data, schema)
                        record = {
                            "prompt":          question,
                            "completion":      completion,
                            "phase_id":        phase_id,
                            "phase_name":      phase_name,
                            "track":           track,
                            "question_type":   q_type,
                            "topic":           topic,
                            "domain":          data.get("domain", topic),
                            "difficulty":      difficulty,
                            "schema":          schema,
                        }
                        # Preserve MCQ fields when present
                        if schema == "mcq":
                            record["answers"]         = data.get("answers", [])
                            record["correct_indices"] = data.get("correct_indices", [0])

                        _append_record(out_path, record)
                        _record_id = f"p{phase_id:02d}_{total_saved:06d}"
                        _fb_ok = _fb_store.store_training_record(_record_id, record)
                        if _fb_ok:
                            logger.info("Firebase synced record %s  [%s]", _record_id, q_type)
                        else:
                            logger.warning("Firebase sync FAILED for record %s", _record_id)
                        topic_pool.append(question)
                        q_type_idx += 1

                        ex_idx      += 1
                        phase_saved += 1
                        total_saved += 1

                    except Exception as e:
                        err = str(e)
                        logger.warning("Q&A error (topic=%s): %s", topic[:40], err)
                        _overloaded = (
                            "429" in err
                            or "rate_limit" in err.lower()
                            or "engine_overloaded" in err.lower()
                            or "overloaded" in err.lower()
                        )
                        if _overloaded:
                            _backoff_key = topic[:40]
                            _backoff_counts[_backoff_key] = _backoff_counts.get(_backoff_key, 0) + 1
                            wait = min(120, 15 * (2 ** (_backoff_counts[_backoff_key] - 1)))
                        else:
                            _backoff_counts.pop(topic[:40], None)
                            wait = 4
                        await asyncio.sleep(wait)
                        continue

                    # ── Update live metrics ───────────────────────────────
                    coverage_now = round(saturated_topics / max(topic_count, 1), 4)
                    sat_now      = round(current_sat, 4)
                    _phase_metrics.update({
                        "phase_id":        phase_id,
                        "phase":           phase_name,
                        "coverage_ratio":  coverage_now,
                        "saturation_score": sat_now,
                        "covered_prompts": saturated_topics,
                        "total_prompts":   topic_count,
                        "is_complete":     False,
                        "iteration":       total_saved,
                        "last_updated":    datetime.utcnow().isoformat(),
                    })
                    _training_metrics.update({
                        "examples_processed": total_saved,
                        "bank_count":         total_saved,
                        "step":               total_saved,
                        "current_phase":      phase_name,
                        "last_updated":       datetime.utcnow().isoformat(),
                    })
                    logger.info(
                        "SAVED #%d  phase=%d/%d  topic=%d/%d  ex=%d  sat=%.3f  %s",
                        total_saved, phase_id, total_phases,
                        t_idx + 1, topic_count, ex_idx, current_sat, topic[:40],
                    )
                    await asyncio.sleep(3)

                    # ── Phase-level max_iterations guard ─────────────────
                    if max_iter and total_saved >= max_iter:
                        logger.info(
                            "Phase %d hit max_iterations (%d) — advancing", phase_id, max_iter
                        )
                        _qa_stop_flag = False  # don't stop entirely, just break inner
                        break

                topics_per_q_count.append(ex_idx)
                if topic_saturated:
                    saturated_topics += 1

                # ── Check phase coverage threshold ────────────────────────
                coverage_ratio = saturated_topics / max(topic_count, 1)
                if coverage_ratio >= phase_coverage_threshold:
                    logger.info(
                        "Phase %d (%s) coverage threshold met (%.1f%%) — advancing",
                        phase_id, phase_name, coverage_ratio * 100,
                    )
                    break  # move to next phase

            # Phase complete
            _phase_metrics.update({
                "phase_id": phase_id, "phase": phase_name,
                "coverage_ratio": min(saturated_topics / max(topic_count, 1), 1.0),
                "saturation_score": 1.0,
                "covered_prompts": saturated_topics, "total_prompts": topic_count,
                "is_complete": True,
                "iteration": total_saved,
                "last_updated": datetime.utcnow().isoformat(),
            })
            logger.info(
                "Phase %d (%s) COMPLETE — %d topics saturated, %d Q&As in %s",
                phase_id, phase_name, saturated_topics, phase_saved, out_path.name,
            )

        if not _qa_stop_flag:
            _training_metrics["training_complete"] = True
            logger.info("ALL PHASES COMPLETE — %d total Q&As in %s", total_saved, _TRAINING_DATA_DIR)

    finally:
        _training_metrics["training_active"] = False
        _training_metrics["last_updated"] = datetime.utcnow().isoformat()
        logger.info("Generation stopped — %d Q&As on disk", total_saved)


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
