"""
FastAPI server for NERO - Provides REST API and WebSocket
for real-time training updates.
"""
import asyncio
import json
from pathlib import Path
from typing import List, Optional, Dict

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
training_active = False


class TrainingConfig(BaseModel):
    teacher_api_key: str
    teacher_base_url: str           # e.g. https://api.moonshot.cn/v1
    teacher_model: str              # e.g. kimi-k2-5
    topics: List[str]
    total_examples: int = 100
    knowledge_base_path: str = "./knowledge_base"
    # Firebase — optional. Omit to run local-only.
    firebase_credential_path: Optional[str] = None
    brain_id: str = "default"


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


@app.post("/initialize")
async def initialize_system(config: TrainingConfig):
    """Initialize brain, teacher, and knowledge base."""
    global brain, teacher, knowledge_manager

    try:
        # Optional Firebase cloud memory
        firebase = None
        if config.firebase_credential_path:
            firebase = FirebaseMemory(
                brain_id=config.brain_id,
                credential_path=config.firebase_credential_path,
            )

        # Initialize knowledge manager — local brain, optional cloud memory
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


async def training_loop():
    """Background training loop — real forward/backward/optimizer steps."""
    global training_active

    BATCH_SIZE = 8
    GRAD_ACCUM_STEPS = 4
    LR = 2e-4
    WARMUP_STEPS = 100
    PHASE_ADVANCE_STEPS = 1000

    curriculum = GenerativeCurriculum(teacher, brain.tokenizer)

    optimizer = AdamW(
        list(brain.model.parameters()) + list(brain.generative_head.parameters()),
        lr=LR,
        weight_decay=0.01,
    )
    scheduler = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=WARMUP_STEPS)

    brain.train()
    step = 0
    optimizer.zero_grad()

    while training_active:
        dataset = curriculum.generate_phase_batch(batch_size=20)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

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

            real_loss = (loss * GRAD_ACCUM_STEPS).item()
            brain.training_history.append({
                "step": step,
                "phase": curriculum.phase_names[curriculum.current_phase],
                "loss": real_loss,
            })

            # Checkpoint every 100 optimizer steps
            if step % (100 * GRAD_ACCUM_STEPS) == 0:
                knowledge_manager.save_checkpoint(
                    brain.model.state_dict(),
                    {
                        "loss": real_loss,
                        "step": step,
                        "phase": curriculum.current_phase,
                    },
                )

            await asyncio.sleep(0)  # Yield control to event loop

        # Advance curriculum phase based on steps
        if step >= PHASE_ADVANCE_STEPS * (curriculum.current_phase + 1):
            curriculum.advance_phase()

    brain.eval()
    training_active = False


@app.websocket("/ws/training")
async def training_websocket(websocket: WebSocket):
    """WebSocket for real-time training updates."""
    await websocket.accept()
    try:
        while True:
            if brain and training_active:
                last = brain.training_history[-1] if brain.training_history else {}
                metrics = {
                    "loss": last.get("loss"),
                    "phase": last.get("phase"),
                    "step": last.get("step", 0),
                    "examples_processed": len(brain.training_history),
                    "timestamp": asyncio.get_event_loop().time(),
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


# Serve static dashboard — must be mounted last so API routes take priority
_dashboard_path = Path(__file__).parents[2] / "dashboard"
if _dashboard_path.exists():
    app.mount("/", StaticFiles(directory=str(_dashboard_path), html=True), name="dashboard")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
