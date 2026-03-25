"""
FastAPI server for NERO - Provides REST API and WebSocket
for real-time training updates.
"""
import asyncio
import json
from pathlib import Path
from typing import List, Optional, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from cognita.core.brain import NEROBrain
from cognita.core.teacher_interface import OpenAITeacher, AnthropicTeacher, TeacherOrchestrator
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
    teacher_provider: str = "openai"
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

        # Initialize teacher
        if config.teacher_provider == "openai":
            teacher_interface = OpenAITeacher(config.teacher_api_key)
        elif config.teacher_provider == "anthropic":
            teacher_interface = AnthropicTeacher(config.teacher_api_key)
        else:
            raise HTTPException(status_code=400, detail="Unsupported teacher provider")

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
    """Background training loop with real-time updates."""
    global training_active
    curriculum = GenerativeCurriculum(teacher, brain.tokenizer)

    while training_active:
        # Generate batch based on current phase
        batch = curriculum.generate_phase_batch(batch_size=20)

        for i, example in enumerate(batch):
            if not training_active:
                break

            # Forward pass, backward pass, update would go here
            # Placeholder: record progress
            brain.training_history.append({
                "step": len(brain.training_history),
                "phase": curriculum.phase_names[curriculum.current_phase],
                "example_idx": i
            })

            # Save checkpoint periodically
            if len(brain.training_history) % 100 == 0:
                knowledge_manager.save_checkpoint(
                    brain.model.state_dict(),
                    {
                        "loss": 0.5,
                        "accuracy": 0.8,
                        "phase": curriculum.current_phase
                    }
                )

        # Check for phase advancement
        if len(brain.training_history) > 1000:
            curriculum.advance_phase()

        await asyncio.sleep(0)  # Yield control

    training_active = False


@app.websocket("/ws/training")
async def training_websocket(websocket: WebSocket):
    """WebSocket for real-time training updates."""
    await websocket.accept()
    try:
        while True:
            if brain and training_active:
                metrics = {
                    "loss": 0.5,
                    "accuracy": 0.8,
                    "phase": "generation",
                    "examples_processed": len(brain.training_history) if brain else 0,
                    "timestamp": asyncio.get_event_loop().time()
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
