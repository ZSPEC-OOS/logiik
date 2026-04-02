"""
Logiik GPU Session Manager.

Manages on-demand model loading, query serving, and safe
GPU teardown for rented GPU instances.

Operational workflow:
  1. You manually start the GPU instance on provider dashboard.
  2. Run: python -m logiik.session_manager.query_server
  3. Model loads from local disk or S3 automatically.
  4. Send queries via POST /ask.
  5. POST /shutdown when done — model unloads, VRAM freed.
  6. Stop GPU instance on provider dashboard.

Auto-expiry: session shuts down automatically after
config.session.auto_expiry_minutes of inactivity.

S3 model loading: enabled when config.session.model_source
is set to 's3' and AWS credentials are in .env.
Currently deferred — set model_source: 'local' until S3
bucket is configured.
"""
import os
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any

import torch

from logiik.config import CONFIG
from logiik.retrieval.retrieve import Retriever
from logiik.session_manager.utils.helpers import (
    SessionLogger, get_gpu_snapshot
)

_logger = SessionLogger("session_manager")


class LogiikSession:
    """
    On-demand Logiik session controller.

    Handles model load → query serving → safe shutdown.
    Designed for rented GPU instances where uptime costs money.

    Args:
        model_path:  Local path to model weights directory.
                     Ignored if model_source='s3'.
        use_gpu:     Force GPU (True) or CPU (False).
                     Default: auto-detect via torch.cuda.
        auto_expiry_minutes: Inactivity timeout before auto-shutdown.
                     0 = disabled.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_gpu: bool = True,
        auto_expiry_minutes: Optional[int] = None,
    ):
        self._cfg = CONFIG.get("session", {})

        # Resolve model path
        self._model_source = self._cfg.get("model_source", "local")
        self._model_path = Path(
            model_path or self._cfg.get("model_path", "./knowledge_base")
        )

        # Device
        self._device = self._resolve_device(use_gpu)
        _logger.info(f"Device resolved: {self._device}")

        # Auto-expiry
        expiry = (
            auto_expiry_minutes
            if auto_expiry_minutes is not None
            else self._cfg.get("auto_expiry_minutes", 30)
        )
        self._expiry_seconds = expiry * 60 if expiry > 0 else 0

        # State
        self._model = None
        self._model_loaded = False
        self._last_query_time = time.time()
        self._query_count = 0
        self._session_start = time.time()
        self._shutdown_requested = False

        # Retriever (lazy — init after model load)
        self._retriever: Optional[Retriever] = None

        # Auto-expiry watchdog thread
        self._watchdog_thread: Optional[threading.Thread] = None
        if self._expiry_seconds > 0:
            self._start_watchdog()

        _logger.info(
            f"LogiikSession initialised: "
            f"model_source={self._model_source}, "
            f"device={self._device}, "
            f"auto_expiry={expiry}min"
        )

    # ─── Public API ───────────────────────────────────────────────────────

    def load_model(self):
        """
        Load Logiik model onto device.
        Called automatically on first query if not called explicitly.

        Model source is determined by config.session.model_source:
          'local' — load from self._model_path (default)
          's3'    — download from S3 then load (deferred)
        """
        if self._model_loaded:
            _logger.info("Model already loaded — skipping.")
            return

        _logger.info(
            f"Loading model from {self._model_source}: "
            f"{self._model_path}"
        )

        if self._model_source == "s3":
            self._load_from_s3()
        else:
            self._load_from_local()

        # Initialise retriever after model load
        try:
            self._retriever = Retriever()
            _logger.info("Retriever initialised.")
        except Exception as e:
            _logger.warning(
                f"Retriever init failed: {e}. "
                "Knowledge retrieval will be unavailable."
            )

        self._model_loaded = True
        gpu = get_gpu_snapshot()
        _logger.info(
            f"Model loaded. "
            f"VRAM used: {gpu.get('vram_used_gb', 0):.1f}GB"
        )

    def query(self, user_input: str) -> Dict[str, Any]:
        """
        Process a user query and return an answer.

        Workflow:
          1. Retrieve relevant knowledge chunks via RAG.
          2. Prepend context to query.
          3. Generate answer via loaded model.

        Args:
            user_input: Natural language question string.

        Returns:
            Dict with 'answer', 'context_chunks', 'confidence',
            'latency_ms', 'query_count'.
        """
        if self._shutdown_requested:
            raise RuntimeError(
                "Session is shutting down. No new queries accepted."
            )

        if not self._model_loaded:
            _logger.info("Model not loaded — loading now...")
            self.load_model()

        start = time.time()
        self._last_query_time = time.time()
        self._query_count += 1

        _logger.info(
            f"Query #{self._query_count}: "
            f"{user_input[:80]}{'...' if len(user_input) > 80 else ''}"
        )

        # Step 1: Retrieve context
        context = ""
        context_chunks = []
        if self._retriever:
            try:
                chunks = self._retriever.retrieve(user_input, top_k=5)
                context = self._retriever.build_context(user_input, top_k=5)
                context_chunks = [
                    {
                        "id": c.id,
                        "score": round(c.score, 4),
                        "source": c.source,
                        "text": c.text[:300],
                    }
                    for c in chunks
                ]
            except Exception as e:
                _logger.warning(f"Retrieval failed: {e}. Proceeding without context.")

        # Step 2: Generate answer
        answer = self._generate(user_input, context)

        latency_ms = round((time.time() - start) * 1000, 2)
        _logger.info(f"Query answered in {latency_ms}ms")

        return {
            "answer": answer,
            "context_chunks": context_chunks,
            "latency_ms": latency_ms,
            "query_count": self._query_count,
        }

    def shutdown(self):
        """
        Safely unload model and free GPU memory.
        Call before stopping GPU instance.
        """
        if self._shutdown_requested:
            return
        self._shutdown_requested = True
        _logger.info("Shutdown requested — unloading model...")

        try:
            if self._model is not None:
                del self._model
                self._model = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                _logger.info("CUDA cache cleared.")

            self._model_loaded = False
            uptime = round((time.time() - self._session_start) / 60, 1)
            _logger.info(
                f"Session terminated. "
                f"Uptime: {uptime}min, "
                f"Queries served: {self._query_count}"
            )
        except Exception as e:
            _logger.error(f"Error during shutdown: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Return current session status snapshot."""
        uptime = round((time.time() - self._session_start) / 60, 1)
        idle = round((time.time() - self._last_query_time) / 60, 1)
        expiry_remaining = None
        if self._expiry_seconds > 0:
            elapsed_idle = time.time() - self._last_query_time
            remaining = max(0, self._expiry_seconds - elapsed_idle)
            expiry_remaining = round(remaining / 60, 1)

        return {
            "model_loaded": self._model_loaded,
            "model_source": self._model_source,
            "device": str(self._device),
            "query_count": self._query_count,
            "uptime_minutes": uptime,
            "idle_minutes": idle,
            "expiry_remaining_minutes": expiry_remaining,
            "shutdown_requested": self._shutdown_requested,
            "gpu": get_gpu_snapshot(),
        }

    # ─── Model loading ────────────────────────────────────────────────────

    def _load_from_local(self):
        """Load model from local knowledge_base directory."""
        if not self._model_path.exists():
            _logger.warning(
                f"Model path not found: {self._model_path}. "
                "Starting with no model — retrieval only mode."
            )
            self._model = None
            return

        try:
            # Import here to avoid circular import
            from cognita.core.brain import NEROBrain
            brain = NEROBrain()
            brain.load_knowledge_state(self._model_path)
            brain.eval()
            self._model = brain
            _logger.info(
                f"NEROBrain loaded from {self._model_path}"
            )
        except Exception as e:
            _logger.warning(
                f"NEROBrain load failed: {e}. "
                "Running in retrieval-only mode."
            )
            self._model = None

    def _load_from_s3(self):
        """
        Download model weights from S3 then load locally.
        DEFERRED — S3 bucket not yet configured.
        Falls back to local load with a warning.
        """
        _logger.warning(
            "S3 model loading is not yet configured. "
            "Set up S3 bucket and add AWS credentials to .env. "
            "Falling back to local model load."
        )
        self._model_source = "local"
        self._load_from_local()

    # ─── Answer generation ────────────────────────────────────────────────

    def _generate(self, query: str, context: str) -> str:
        """
        Generate answer from model.
        Falls back to context-only response if model not loaded.
        """
        # Retrieval-only mode (no model loaded)
        if self._model is None:
            if context:
                return (
                    f"[Retrieval-only mode — no model loaded]\n\n"
                    f"Retrieved context:\n{context}"
                )
            return (
                "[No model loaded and no relevant context found. "
                "Load a trained model to generate answers.]"
            )

        try:
            full_query = (
                f"Context:\n{context}\n\nQuestion: {query}"
                if context else query
            )
            result = self._model.generate_original_answer(
                full_query,
                min_confidence=0.5
            )
            return result.get("answer", "No answer generated.")
        except Exception as e:
            _logger.error(f"Generation failed: {e}")
            return f"Generation error: {e}"

    # ─── Auto-expiry watchdog ─────────────────────────────────────────────

    def _start_watchdog(self):
        """Start background thread that shuts down on inactivity."""
        def watchdog():
            _logger.info(
                f"Auto-expiry watchdog started: "
                f"timeout={self._expiry_seconds//60}min"
            )
            while not self._shutdown_requested:
                time.sleep(30)  # Check every 30 seconds
                if self._shutdown_requested:
                    break
                idle = time.time() - self._last_query_time
                if idle >= self._expiry_seconds:
                    _logger.info(
                        f"Auto-expiry triggered after "
                        f"{idle/60:.1f}min idle. Shutting down."
                    )
                    self.shutdown()
                    break

        self._watchdog_thread = threading.Thread(
            target=watchdog, daemon=True
        )
        self._watchdog_thread.start()

    # ─── Device resolution ────────────────────────────────────────────────

    @staticmethod
    def _resolve_device(use_gpu: bool) -> str:
        if not use_gpu:
            return "cpu"
        try:
            if torch.cuda.is_available():
                vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                name = torch.cuda.get_device_name(0)
                _logger.info(
                    f"GPU detected: {name} ({vram:.1f}GB VRAM)"
                )
                return "cuda"
            else:
                _logger.info(
                    "No GPU detected — using CPU. "
                    "Inference will be slow on large models."
                )
                return "cpu"
        except Exception:
            return "cpu"
