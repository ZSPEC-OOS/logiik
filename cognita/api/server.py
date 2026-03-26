"""
FastAPI server for NERO - Provides REST API and WebSocket
for real-time training updates.
"""
import asyncio
import json
from pathlib import Path
from typing import List, Optional, Dict

import yaml

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from cognita.core.brain import NEROBrain
from cognita.core.teacher_interface import KimiK2Teacher, TeacherOrchestrator
from cognita.training.curriculum import GenerativeCurriculum
from cognita.storage.checkpoint_manager import KnowledgeBaseManager
from cognita.storage.firebase_memory import FirebaseMemory
from cognita.storage.question_bank import QuestionBank

app = FastAPI(title="NERO API", version="1.0.0")

# CORS for dashboard communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
brain: Optional[NEROBrain] = None
teacher: Optional[TeacherOrchestrator] = None
knowledge_manager: Optional[KnowledgeBaseManager] = None
question_bank: Optional[QuestionBank] = None
training_active = False
training_complete = False
_topics_description: str = ""
_repeat_threshold: int = 75
_final_report: Optional[Dict] = None
_nlp_phase_topics: Dict[str, List[str]] = {}
_topics_per_session: int = 5

# Load NLP curriculum from teacher_config.yaml on startup
_TEACHER_CONFIG_PATH = Path(__file__).parents[2] / "configs" / "teacher_config.yaml"

def _load_nlp_curriculum() -> tuple[Dict[str, List[str]], int]:
    """Read phase_topics and topics_per_session from teacher_config.yaml."""
    if not _TEACHER_CONFIG_PATH.exists():
        return {}, 5
    with open(_TEACHER_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f) or {}
    phase_topics = cfg.get("nlp_curriculum", {})
    topics_per_session = (
        cfg.get("teacher", {}).get("curriculum", {}).get("topics_per_session", 5)
    )
    return phase_topics, topics_per_session


class TrainingConfig(BaseModel):
    teacher_api_key: str
    teacher_base_url: str                   # e.g. https://api.moonshot.cn/v1
    teacher_model: str                      # e.g. kimi-k2-5
    topics_description: str                 # free-form paragraph describing training focus
    question_repeat_threshold: int = 75     # halt when toss log reaches this count
    knowledge_base_path: str = "./knowledge_base"


class QuestionRequest(BaseModel):
    question: str
    require_original: bool = True


@app.get("/health")
async def health_check():
    """API health check endpoint."""
    return {
        "status": "ok",
        "brain_loaded": brain is not None,
        "teacher_loaded": teacher is not None,
        "training_active": training_active
    }


@app.get("/browse-folder")
async def browse_folder():
    """Open a native OS folder-picker dialog and return the chosen path."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        path = filedialog.askdirectory(title="Select Knowledge Base Folder")
        root.destroy()
        return {"path": path or ""}
    except Exception as e:
        return {"path": "", "error": str(e)}


@app.post("/initialize")
async def initialize_system(config: TrainingConfig):
    """Initialize brain, teacher, knowledge base, and question bank."""
    global brain, teacher, knowledge_manager, question_bank
    global _topics_description, _repeat_threshold, training_complete, _final_report
    global _nlp_phase_topics, _topics_per_session

    try:
        # Firebase is always active — config is hardcoded in firebase_memory.py
        firebase = FirebaseMemory()

        # Initialize knowledge manager — local storage + cloud sync
        knowledge_manager = KnowledgeBaseManager(
            config.knowledge_base_path,
            firebase=firebase,
        )

        # Initialize brain
        brain = NEROBrain()

        # Initialize Kimi K2.5 teacher
        teacher_interface = KimiK2Teacher(
            api_key=config.teacher_api_key,
            base_url=config.teacher_base_url,
            model=config.teacher_model,
        )
        teacher = TeacherOrchestrator(teacher_interface)

        # Initialize question bank
        question_bank = QuestionBank(config.knowledge_base_path)
        _topics_description = config.topics_description
        _repeat_threshold = config.question_repeat_threshold
        training_complete = False
        _final_report = None
        _nlp_phase_topics, _topics_per_session = _load_nlp_curriculum()

        # Load existing knowledge if available
        kb_summary = knowledge_manager.get_attachable_knowledge_summary()
        if kb_summary["checkpoints_count"] > 0:
            latest = kb_summary["latest_checkpoint"]
            if latest:
                brain.load_knowledge_state(latest["path"])

        return {
            "status": "initialized",
            "knowledge_base": kb_summary
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train/start")
async def start_training():
    """Start training process."""
    global training_active

    if not all([brain, teacher, knowledge_manager]):
        raise HTTPException(status_code=400, detail="System not initialized")

    training_active = True
    asyncio.create_task(training_loop())

    return {"status": "training_started"}


@app.post("/train/stop")
async def stop_training():
    """Stop training process."""
    global training_active
    training_active = False
    return {"status": "training_stopped"}


def _run_validation(dataset, batch_size: int) -> Dict:
    """Compute validation loss and perplexity with no_grad. Returns metrics dict."""
    import math
    val_ds = dataset.val_dataset
    if val_ds is None or len(val_ds) == 0:
        return {}

    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    total_batches = 0

    brain.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch.input_ids.to(brain.device)
            attention_mask = batch.attention_mask.to(brain.device)
            labels = batch.labels.to(brain.device)

            outputs = brain(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            if outputs["loss"] is not None:
                total_loss += outputs["loss"].item()
                total_batches += 1

    brain.train()

    if total_batches == 0:
        return {}

    avg_loss = total_loss / total_batches
    return {"val_loss": avg_loss, "val_perplexity": math.exp(min(avg_loss, 20))}


async def training_loop():
    """Background training loop — real forward/backward/optimizer steps."""
    global training_active, training_complete, _final_report

    BATCH_SIZE = 8
    GRAD_ACCUM_STEPS = 4
    LR = 2e-4
    WARMUP_STEPS = 100
    VAL_EVERY_STEPS = 50          # Run validation every N optimizer steps
    PLATEAU_PATIENCE = 3          # Advance phase after N val checks with no improvement
    PLATEAU_MIN_DELTA = 0.01      # Minimum improvement to reset patience counter

    curriculum = GenerativeCurriculum(
        teacher,
        brain.tokenizer,
        topics_description=_topics_description,
        phase_topics=_nlp_phase_topics,
        topics_per_session=_topics_per_session,
    )

    optimizer = AdamW(
        list(brain.model.parameters()) + list(brain.generative_head.parameters()),
        lr=LR,
        weight_decay=0.01,
    )
    scheduler = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=WARMUP_STEPS)

    brain.train()
    step = 0
    optimizer_step = 0
    optimizer.zero_grad()

    best_val_loss = float("inf")
    plateau_count = 0
    current_dataset = None

    while training_active:
        # Check repeat threshold before generating new batch
        if question_bank and question_bank.toss_count >= _repeat_threshold:
            training_active = False
            training_complete = True
            _final_report = question_bank.generate_report(_topics_description)
            break

        current_dataset = curriculum.generate_phase_batch(
            batch_size=20,
            question_bank=question_bank,
        )
        loader = DataLoader(current_dataset, batch_size=BATCH_SIZE, shuffle=True)

        for batch in loader:
            if not training_active:
                break

            input_ids = batch.input_ids.to(brain.device)
            attention_mask = batch.attention_mask.to(brain.device)
            labels = batch.labels.to(brain.device)
            teacher_logits = batch.teacher_logits.to(brain.device) if batch.teacher_logits is not None else None

            outputs = brain(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                teacher_logits=teacher_logits,
            )

            loss = outputs["loss"] / GRAD_ACCUM_STEPS
            loss.backward()

            step += 1
            if step % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(brain.model.parameters()) + list(brain.generative_head.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optimizer_step += 1

                # Validation pass
                if optimizer_step % VAL_EVERY_STEPS == 0:
                    val_metrics = _run_validation(current_dataset, BATCH_SIZE)
                    val_loss = val_metrics.get("val_loss")

                    brain.training_history.append({
                        "step": optimizer_step,
                        "phase": curriculum.phase_names[curriculum.current_phase],
                        "loss": (loss * GRAD_ACCUM_STEPS).item(),
                        **val_metrics,
                    })

                    # Plateau detection for phase advancement
                    if val_loss is not None:
                        if best_val_loss - val_loss > PLATEAU_MIN_DELTA:
                            best_val_loss = val_loss
                            plateau_count = 0
                        else:
                            plateau_count += 1

                        if plateau_count >= PLATEAU_PATIENCE:
                            if curriculum.advance_phase():
                                best_val_loss = float("inf")
                                plateau_count = 0

                # Checkpoint every 100 optimizer steps
                if optimizer_step % 100 == 0:
                    last = brain.training_history[-1] if brain.training_history else {}
                    knowledge_manager.save_checkpoint(
                        brain.model.state_dict(),
                        {
                            "loss": last.get("loss"),
                            "val_loss": last.get("val_loss"),
                            "val_perplexity": last.get("val_perplexity"),
                            "step": optimizer_step,
                            "phase": curriculum.current_phase,
                        },
                    )

            else:
                real_loss = (loss * GRAD_ACCUM_STEPS).item()
                brain.training_history.append({
                    "step": optimizer_step,
                    "phase": curriculum.phase_names[curriculum.current_phase],
                    "loss": real_loss,
                })

            await asyncio.sleep(0)  # Yield control to event loop

    brain.eval()
    training_active = False


@app.websocket("/ws/training")
async def training_websocket(websocket: WebSocket):
    """WebSocket for real-time training updates."""
    await websocket.accept()
    try:
        while True:
            if brain:
                last = brain.training_history[-1] if brain.training_history else {}
                metrics = {
                    "loss":               last.get("loss"),
                    "val_loss":           last.get("val_loss"),
                    "val_perplexity":     last.get("val_perplexity"),
                    "phase":              last.get("phase"),
                    "step":               last.get("step", 0),
                    "examples_processed": len(brain.training_history),
                    "bank_count":         question_bank.bank_count if question_bank else 0,
                    "toss_count":         question_bank.toss_count if question_bank else 0,
                    "repeat_threshold":   _repeat_threshold,
                    "training_complete":  training_complete,
                    "timestamp":          asyncio.get_event_loop().time(),
                }
                await websocket.send_json(metrics)

            await asyncio.sleep(2)  # Update every 2 seconds

    except WebSocketDisconnect:
        print("Client disconnected from training websocket")


@app.post("/ask")
async def ask_ai(request: QuestionRequest):
    """Query the trained AI."""
    if not brain:
        raise HTTPException(status_code=400, detail="Brain not initialized")

    response = brain.generate_original_answer(
        request.question,
        min_confidence=0.7 if request.require_original else 0.5
    )

    return response


@app.get("/knowledge/summary")
async def knowledge_summary():
    """Get attachable knowledge base summary."""
    if not knowledge_manager:
        raise HTTPException(status_code=400, detail="Knowledge manager not initialized")

    return knowledge_manager.get_attachable_knowledge_summary()


@app.post("/knowledge/export")
async def export_knowledge(output_path: str = ".", name: str = "latest"):
    """Export knowledge base as a portable package."""
    if not knowledge_manager:
        raise HTTPException(status_code=400, detail="Knowledge manager not initialized")

    output = knowledge_manager.export_knowledge_package(output_path, name)
    return {"status": "exported", "path": str(output)}


@app.get("/logs/question-bank")
async def get_question_bank():
    """Return all unique questions that have been transmitted to NERO."""
    if not question_bank:
        raise HTTPException(status_code=400, detail="System not initialized")
    return {
        "count": question_bank.bank_count,
        "entries": question_bank.get_bank(),
    }


@app.get("/logs/toss-log")
async def get_toss_log():
    """Return all duplicate questions that were discarded."""
    if not question_bank:
        raise HTTPException(status_code=400, detail="System not initialized")
    return {
        "count": question_bank.toss_count,
        "threshold": _repeat_threshold,
        "entries": question_bank.get_toss_log(),
    }


@app.get("/logs/report")
async def get_report():
    """Return the final training completion report (only available after training completes)."""
    if not question_bank:
        raise HTTPException(status_code=400, detail="System not initialized")
    if not training_complete:
        return {
            "status": "in_progress",
            "bank_count": question_bank.bank_count,
            "toss_count": question_bank.toss_count,
            "threshold": _repeat_threshold,
        }
    return _final_report


# Serve static dashboard — must be mounted last so API routes take priority
_dashboard_path = Path(__file__).parents[2] / "dashboard"
if _dashboard_path.exists():
    app.mount("/", StaticFiles(directory=str(_dashboard_path), html=True), name="dashboard")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
