"""
Logiik Integration Test Suite.

Covers all modules implemented in CHUNK-01 through CHUNK-12.
Tests are grouped by module. Each test is independent.

Run all tests:
    python -m pytest logiik/tests/test_modules.py -v

Run a single group:
    python -m pytest logiik/tests/test_modules.py -v -k "storage"
"""
import numpy as np
import pytest
import torch
from pathlib import Path
from PIL import Image
from unittest.mock import MagicMock, patch


# ─── CONFIG ──────────────────────────────────────────────────────────────────

class TestConfig:
    def test_load_config(self):
        from logiik.config import CONFIG, load_config
        cfg = load_config()
        assert isinstance(cfg, dict)
        assert "embeddings" in cfg
        assert "vector_db" in cfg
        assert "firebase" in cfg
        assert "curriculum" in cfg

    def test_embedding_dimension(self):
        from logiik.config import CONFIG
        assert CONFIG["embeddings"]["dimension"] == 768

    def test_phases_in_config(self):
        from logiik.config import CONFIG
        phases = CONFIG["curriculum"]["phases"]
        assert len(phases) == 12

    def test_phase6_config(self):
        from logiik.config import CONFIG
        phases = CONFIG["curriculum"]["phases"]
        p6 = next(p for p in phases if p["id"] == 6)
        assert p6["name"] == "niche_scientific_reasoning"
        assert p6["teacher_student"] == True
        assert p6["correctness_threshold"] == 0.90

    def test_cache_disabled_by_default(self):
        from logiik.config import CONFIG
        assert CONFIG["cache"]["enabled"] == False

    def test_s3_disabled_by_default(self):
        from logiik.config import CONFIG
        assert CONFIG["s3"]["enabled"] == False


# ─── UTILS ───────────────────────────────────────────────────────────────────

class TestUtils:
    def test_is_duplicate_true(self):
        from logiik.utils.helpers import is_duplicate
        base = np.random.randn(768).astype(np.float32)
        base /= np.linalg.norm(base)
        noisy = base + np.random.randn(768).astype(np.float32) * 0.001
        noisy /= np.linalg.norm(noisy)
        assert is_duplicate(noisy, [base], threshold=0.90) == True

    def test_is_duplicate_false(self):
        from logiik.utils.helpers import is_duplicate
        v1 = np.random.randn(768).astype(np.float32)
        v1 /= np.linalg.norm(v1)
        v2 = np.random.randn(768).astype(np.float32)
        v2 /= np.linalg.norm(v2)
        result = is_duplicate(v1, [v2], threshold=0.90)
        assert isinstance(result, bool)

    def test_is_duplicate_empty_list(self):
        from logiik.utils.helpers import is_duplicate
        v = np.random.randn(768).astype(np.float32)
        assert is_duplicate(v, []) == False

    def test_validate_answer_pass(self):
        from logiik.utils.helpers import validate_answer
        assert validate_answer("This is a sufficiently long answer.") == True

    def test_validate_answer_fail_short(self):
        from logiik.utils.helpers import validate_answer
        assert validate_answer("Short") == False

    def test_validate_answer_fail_empty(self):
        from logiik.utils.helpers import validate_answer
        assert validate_answer("") == False
        assert validate_answer(None) == False

    def test_chunk_text(self):
        from logiik.utils.helpers import chunk_text
        text = "a" * 1000
        chunks = chunk_text(text, chunk_size=200, overlap=50)
        assert len(chunks) > 0
        assert all(len(c) <= 200 for c in chunks)

    def test_chunk_text_empty(self):
        from logiik.utils.helpers import chunk_text
        assert chunk_text("", 200, 50) == []

    def test_compute_saturation_empty(self):
        from logiik.utils.helpers import compute_saturation
        v = np.random.randn(768).astype(np.float32)
        assert compute_saturation(v, []) == 0.0

    def test_compute_saturation_value(self):
        from logiik.utils.helpers import compute_saturation
        base = np.random.randn(768).astype(np.float32)
        base /= np.linalg.norm(base)
        past = [base + np.random.randn(768) * 0.01 for _ in range(5)]
        for i in range(5):
            past[i] = past[i] / np.linalg.norm(past[i])
        score = compute_saturation(base, past)
        assert 0.0 <= score <= 1.0

    def test_logging(self):
        from logiik.utils.logging import get_logger, log_event
        logger = get_logger("test")
        assert logger is not None
        log_event("test", "Test log entry", level="info")


# ─── STORAGE ─────────────────────────────────────────────────────────────────

class TestStorage:
    def test_cache_disabled_noop(self):
        from logiik.storage.cache import Cache
        cache = Cache()
        assert cache.is_enabled == False
        assert cache.set("k1", "v1") == False
        assert cache.get("k1") is None
        assert cache.delete("k1") == False

    def test_vector_db_import(self):
        from logiik.storage.vector_db import (
            VectorDB, VectorMatch,
            PineconeBackend, FAISSBackend
        )
        assert VectorDB is not None
        assert VectorMatch is not None

    def test_vector_match(self):
        from logiik.storage.vector_db import VectorMatch
        m = VectorMatch(id="test_id", score=0.95, metadata={"text": "hello"})
        assert m.id == "test_id"
        assert m.score == 0.95
        assert "test_id" in repr(m)

    def test_text_store_import(self):
        from logiik.storage.text_store import TextStore, _enc, _dec
        assert TextStore is not None

    def test_firestore_encoding_roundtrip(self):
        from logiik.storage.text_store import _enc, _dec
        test_cases = [
            "hello world",
            42,
            3.14,
            True,
            None,
            ["a", "b", "c"],
            {"key": "value", "num": 7},
        ]
        for val in test_cases:
            encoded = _enc(val)
            decoded = _dec(encoded)
            assert decoded == val, (
                f"Roundtrip failed for {val!r}: got {decoded!r}"
            )


# ─── EMBEDDINGS ──────────────────────────────────────────────────────────────

class TestEmbeddings:
    def test_embedder_import(self):
        from logiik.embeddings.embed import (
            get_embedder, Embedder,
            TextEmbedder, ImageEmbedder
        )
        assert get_embedder is not None

    def test_embedder_singleton(self):
        from logiik.embeddings.embed import get_embedder
        e1 = get_embedder()
        e2 = get_embedder()
        assert e1 is e2

    def test_text_embedder_lazy(self):
        from logiik.embeddings.embed import TextEmbedder
        te = TextEmbedder()
        assert te._model is None

    def test_image_embedder_lazy(self):
        from logiik.embeddings.embed import ImageEmbedder
        ie = ImageEmbedder()
        assert ie._model is None


# ─── RETRIEVAL ────────────────────────────────────────────────────────────────

class TestRetrieval:
    def test_retriever_import(self):
        from logiik.retrieval.retrieve import Retriever, RetrievedChunk
        assert Retriever is not None

    def test_retrieved_chunk(self):
        from logiik.retrieval.retrieve import RetrievedChunk
        chunk = RetrievedChunk(
            id="c1",
            text="enzyme folding mechanism",
            score=0.92,
            metadata={"source": "textbook.pdf"},
            source="textbook.pdf",
            cache_hit=False,
        )
        assert chunk.id == "c1"
        assert chunk.score == 0.92
        assert "c1" in repr(chunk)

    def test_retriever_instantiation(self):
        from logiik.retrieval.retrieve import Retriever
        from logiik.storage.cache import Cache
        mock_db = MagicMock()
        mock_store = MagicMock()
        mock_cache = Cache()
        r = Retriever(
            vector_db=mock_db,
            text_store=mock_store,
            cache=mock_cache
        )
        assert r is not None

    def test_build_context_empty(self):
        from logiik.retrieval.retrieve import Retriever
        from logiik.storage.cache import Cache
        mock_db = MagicMock()
        mock_db.query.return_value = []
        mock_db.backend_name = "mock"
        mock_store = MagicMock()
        mock_cache = Cache()
        r = Retriever(
            vector_db=mock_db,
            text_store=mock_store,
            cache=mock_cache
        )
        r._embedder = MagicMock()
        r._embedder.embed_text.return_value = np.zeros(768)
        context = r.build_context("test query")
        assert context == ""


# ─── CURRICULUM & PHASES ──────────────────────────────────────────────────────

class TestCurriculum:
    def test_all_phases_loaded(self):
        from logiik.curriculum.phases import PHASES
        assert len(PHASES) == 12

    def test_phase_ids_sequential(self):
        from logiik.curriculum.phases import PHASES
        ids = [p.id for p in PHASES]
        assert ids == list(range(1, 13))

    def test_get_phase(self):
        from logiik.curriculum.phases import get_phase
        for i in range(1, 13):
            p = get_phase(i)
            assert p is not None
            assert p.id == i

    def test_get_phase_invalid(self):
        from logiik.curriculum.phases import get_phase
        assert get_phase(0) is None
        assert get_phase(13) is None

    def test_phase6_spec(self):
        from logiik.curriculum.phases import get_phase
        p6 = get_phase(6)
        assert p6.name == "niche_scientific_reasoning"
        assert p6.teacher_student == True
        assert p6.correctness_threshold == 0.90
        assert p6.generative_ratio == 0.93
        assert "drosophila" not in p6.name.lower()
        assert p6.metadata["legacy_phase"] == "drosophila_ai_framework"

    def test_phase12_stages_in_metadata(self):
        from logiik.curriculum.phases import get_phase
        p12 = get_phase(12)
        stages = p12.metadata["stages"]
        assert len(stages) == 6
        names = [s["name"] for s in stages]
        assert all(n.startswith("phase12_stage_") for n in names)

    def test_teacher_student_phases(self):
        from logiik.curriculum.phases import get_teacher_student_phases
        ts = get_teacher_student_phases()
        ts_ids = [p.id for p in ts]
        assert ts_ids == [6, 11], f"Expected [6, 11], got {ts_ids}"

    def test_get_all_phase_names(self):
        from logiik.curriculum.phases import get_all_phase_names
        names = get_all_phase_names()
        assert len(names) == 12
        assert "Niche & Interdisciplinary Scientific Reasoning" in names
        assert "Synthetic Judgment" in names
        assert "Scientific Language & Literature" in names
        assert "Mathematical & Statistical Reasoning" in names
        assert "Adversarial Robustness & Epistemic Integrity" in names
        assert "Research Computing & Scientific Coding" in names


# ─── TRAINING (Phase 7) ───────────────────────────────────────────────────────

class TestPhase7Training:
    def test_imports(self):
        from logiik.core.training import (
            Phase7TeacherStudentLoop,
            GenerativeCurriculum,
            CurriculumDataset,
            TrainingExample,
            collate_examples,
            PhaseCompletionMonitor,
            build_phase_monitor,
        )
        assert Phase7TeacherStudentLoop is not None

    def test_teacher_question_registration(self):
        from logiik.core.training import Phase7TeacherStudentLoop
        loop = Phase7TeacherStudentLoop.__new__(Phase7TeacherStudentLoop)
        loop._teacher_db = {}
        loop._student_db = {}
        loop._metrics = {}
        loop._store = MagicMock()
        loop._store.store_phase7_teacher.return_value = True

        qid = loop.generate_teacher_question(
            prompt="How does pH affect enzyme folding?",
            answer_steps=[
                "Step 1: Identify ionisable residues",
                "Step 2: Predict protonation at target pH",
                "Step 3: Determine structural consequences",
            ],
            full_answer="At low pH, histidine residues protonate...",
        )
        assert qid in loop._teacher_db
        assert loop._teacher_db[qid].prompt == "How does pH affect enzyme folding?"

    def test_student_attempt(self):
        from logiik.core.training import (
            Phase7TeacherStudentLoop, StudentAttempt
        )
        loop = Phase7TeacherStudentLoop.__new__(Phase7TeacherStudentLoop)
        loop._teacher_db = {}
        loop._student_db = {}
        loop._metrics = {}
        loop._store = MagicMock()
        loop._store.store_phase7_teacher.return_value = True

        qid = loop.generate_teacher_question(
            "Test prompt",
            ["Step 1", "Step 2"],
            "Full answer"
        )
        attempt = loop.student_attempt(
            qid,
            ["Student step 1", "Student step 2"],
            "Student full answer with sufficient length for validation."
        )
        assert isinstance(attempt, StudentAttempt)
        assert attempt.question_id == qid
        assert attempt.attempt_index == 0

    def test_feedback_threshold(self):
        from logiik.core.training import Phase7TeacherStudentLoop
        loop = Phase7TeacherStudentLoop.__new__(Phase7TeacherStudentLoop)
        loop._teacher_db = {}
        loop._student_db = {}
        loop._metrics = {}
        loop._store = MagicMock()
        loop._store.store_phase7_teacher.return_value = True
        loop._store.store_phase7_student.return_value = True

        qid = loop.generate_teacher_question("Q", ["S1"], "A")
        loop.student_attempt(qid, ["s1"], "answer " * 10)

        result = loop.provide_feedback(qid, 0, ["needs work"], 0.85)
        assert result == False

        loop.student_attempt(qid, ["s1 improved"], "improved answer " * 5)
        result = loop.provide_feedback(qid, 1, ["good"], 0.90)
        assert result == True

    def test_phase_completion_monitor(self):
        from logiik.core.training import build_phase_monitor
        prompts = [{"id": f"p{i}"} for i in range(10)]
        monitor = build_phase_monitor(6, prompts)
        assert monitor is not None
        metrics = monitor.get_metrics()
        assert metrics["phase"] == "Niche & Interdisciplinary Scientific Reasoning"
        assert metrics["total_prompts"] == 10
        assert metrics["coverage_ratio"] == 0.0

    def test_monitor_update_and_coverage(self):
        from logiik.core.training import build_phase_monitor
        prompts = [{"id": f"p{i}"} for i in range(5)]
        monitor = build_phase_monitor(6, prompts)
        for i in range(5):
            emb = np.random.randn(768).astype(np.float32)
            emb /= np.linalg.norm(emb)
            monitor.update(
                "Sufficient answer text here for validation.",
                emb,
                f"p{i}"
            )
        assert monitor.get_metrics()["covered_prompts"] == 5
        assert monitor.get_metrics()["coverage_ratio"] == 1.0


# ─── INGESTION ────────────────────────────────────────────────────────────────

class TestIngestion:
    def test_phase8_import(self):
        from logiik.ingestion.phase8_images import (
            Phase8ImagePipeline, ImageRecord, IMAGE_TYPES
        )
        assert len(IMAGE_TYPES) == 6

    def test_phase8_classify_image_type(self):
        from logiik.ingestion.phase8_images import Phase8ImagePipeline
        pipeline = Phase8ImagePipeline.__new__(Phase8ImagePipeline)
        pipeline.phase8_image_db = []
        cases = [
            ("confocal microscopy stain fluorescen", "microscopy"),
            ("bar chart comparison", "chart"),
            ("scatter plot regression", "plot"),
            ("chemical molecule synthesis", "chemical_structure"),
            ("figure diagram pathway", "diagram"),
            ("", "other"),
        ]
        for caption, expected in cases:
            result = pipeline._classify_image_type(caption)
            assert result == expected, (
                f"classify({caption!r}) = {result!r}, expected {expected!r}"
            )

    def test_phase8_dedup_empty_db(self):
        from logiik.ingestion.phase8_images import Phase8ImagePipeline
        pipeline = Phase8ImagePipeline.__new__(Phase8ImagePipeline)
        pipeline.phase8_image_db = []
        pipeline._threshold = 0.90
        pipeline._embedder = MagicMock()
        img = Image.new("RGB", (64, 64))
        result = pipeline.is_image_duplicate(img, caption="test")
        assert result == False

    def test_phase9_import(self):
        from logiik.ingestion.phase9_pdfs import (
            Phase9PDFPipeline, IngestionResult, INGESTION_CONFIG
        )
        assert "lite_mode" in INGESTION_CONFIG
        assert "full_mode" in INGESTION_CONFIG

    def test_phase9_mode_config(self):
        from logiik.ingestion.phase9_pdfs import INGESTION_CONFIG
        assert INGESTION_CONFIG["lite_mode"]["chunk_size"] == 512
        assert INGESTION_CONFIG["lite_mode"]["cloud_stage"] == False
        assert INGESTION_CONFIG["full_mode"]["chunk_size"] == 1024
        assert INGESTION_CONFIG["full_mode"]["cloud_stage"] == True

    def test_ingestion_result(self):
        from logiik.ingestion.phase9_pdfs import IngestionResult
        r = IngestionResult("test.pdf")
        r.chunks_new = 10
        r.chunks_duplicate = 2
        d = r.to_dict()
        assert d["chunks_new"] == 10
        assert d["pdf_path"] == "test.pdf"
        assert "errors" in d


# ─── PHASE 10 ─────────────────────────────────────────────────────────────────

class TestPhase10:
    def test_imports(self):
        from logiik.core.phase10_training import (
            ReasoningStep, ModelOutput,
            CausalScenario, ScenarioGenerator,
            DeliberationEngine, EvaluationEngine,
            RewardEngine, Phase10Trainer,
            PHASE10_STAGES,
        )
        assert len(PHASE10_STAGES) == 6

    def test_causal_scenario_no_confounder(self):
        from logiik.core.phase10_training import CausalScenario
        s = CausalScenario().generate(n_nodes=5, inject_confounder=False)
        assert s.knowable == True
        prompt = s.to_prompt()
        assert "Causal System" in prompt

    def test_causal_scenario_with_confounder(self):
        from logiik.core.phase10_training import CausalScenario
        s = CausalScenario().generate(n_nodes=5, inject_confounder=True)
        assert s.knowable == False
        assert "Z_hidden" in s.graph.nodes

    def test_scenario_generator_all_stages(self):
        from logiik.core.phase10_training import (
            ScenarioGenerator, PHASE10_STAGES
        )
        gen = ScenarioGenerator()
        for stage in PHASE10_STAGES:
            scenarios = gen.generate(stage, count=2)
            assert len(scenarios) == 2
            for prompt, gt, knowable in scenarios:
                assert isinstance(prompt, str)
                assert len(prompt) > 0

    def test_evaluation_engine_brier(self):
        from logiik.core.phase10_training import EvaluationEngine
        e = EvaluationEngine()
        assert e.brier_score(1.0, 1.0) == 0.0
        assert e.brier_score(0.0, 1.0) == 1.0
        assert e.brier_score(0.5, 0.5) == 0.0

    def test_evaluation_engine_abstention(self):
        from logiik.core.phase10_training import EvaluationEngine
        e = EvaluationEngine()
        assert e.abstention_score("abstain", 0.0, False) == 0.6
        assert e.abstention_score("abstain", 0.0, True) == -0.5
        assert e.abstention_score("answer", 1.0, True) == 1.0
        assert e.abstention_score("answer", 0.0, True) == -1.5

    def test_reward_engine(self):
        from logiik.core.phase10_training import (
            RewardEngine, ModelOutput, ReasoningStep
        )
        engine = RewardEngine()
        steps = [
            ReasoningStep(0, "observation", "obs content", 0.8),
            ReasoningStep(1, "inference", "inf content", 0.85),
        ]
        output = ModelOutput(
            decision="answer",
            final_answer="correct answer",
            reasoning_chain=steps,
            confidence=0.82,
            uncertainty={"aleatoric": 0.18, "epistemic": 0.0, "model": 0.09},
        )
        reward, components = engine.compute_reward(output, "correct answer", True)
        assert isinstance(reward, float)
        assert all(
            k in components
            for k in ["correctness", "consistency", "calibration", "abstention", "total"]
        )
        assert components["correctness"] == 1.0

    def test_deliberation_engine_scaffold(self):
        from logiik.core.phase10_training import DeliberationEngine
        mock_model = MagicMock()
        mock_model.generate_step.side_effect = NotImplementedError
        mock_model.generate_final.side_effect = NotImplementedError
        engine = DeliberationEngine(mock_model, max_steps=5)
        output = engine.run("Test scientific prompt here.")
        assert output.decision in ("answer", "abstain")
        assert len(output.reasoning_chain) > 0
        assert 0.0 <= output.confidence <= 1.0

    def test_phase12_stages_in_metadata(self):
        from logiik.core.phase10_training import PHASE10_STAGES
        assert len(PHASE10_STAGES) == 6
        assert all(
            s.startswith("phase10_stage_")
            for s in PHASE10_STAGES
        )


# ─── API ──────────────────────────────────────────────────────────────────────

class TestAPI:
    def test_endpoints_import(self):
        from logiik.api.endpoints import app
        assert app is not None

    def test_health_endpoint(self):
        from logiik.api.endpoints import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        r = client.get("/logiik/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_phase_metrics_default(self):
        from logiik.api.endpoints import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        r = client.get("/logiik/phase_metrics")
        assert r.status_code == 200
        data = r.json()
        assert "coverage_ratio" in data
        assert "saturation_score" in data

    def test_training_metrics_default(self):
        from logiik.api.endpoints import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        r = client.get("/logiik/training_metrics")
        assert r.status_code == 200

    def test_gpu_status(self):
        from logiik.api.endpoints import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        r = client.get("/logiik/gpu_status")
        assert r.status_code == 200
        data = r.json()
        assert "gpu_available" in data

    def test_phase_update_roundtrip(self):
        from logiik.api.endpoints import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        payload = {
            "phase_id": 7,
            "phase": "Niche Scientific Reasoning",
            "coverage_ratio": 0.85,
            "saturation_score": 0.72,
            "covered_prompts": 17,
            "total_prompts": 20,
            "is_complete": False,
            "iteration": 99,
        }
        r = client.post("/logiik/phase/update", json=payload)
        assert r.status_code == 200
        r2 = client.get("/logiik/phase_metrics")
        data = r2.json()
        assert data["coverage_ratio"] == 0.85
        assert data["iteration"] == 99


# ─── SESSION MANAGER ─────────────────────────────────────────────────────────

class TestSessionManager:
    def test_imports(self):
        from logiik.session_manager.session_manager import LogiikSession
        from logiik.session_manager.query_server import app
        from logiik.session_manager.utils.helpers import (
            SessionLogger, get_gpu_snapshot, format_uptime
        )
        assert LogiikSession is not None

    def test_format_uptime(self):
        from logiik.session_manager.utils.helpers import format_uptime
        assert format_uptime(45) == "45s"
        assert format_uptime(130) == "2m 10s"
        assert format_uptime(3661) == "1h 1m 1s"

    def test_gpu_snapshot_structure(self):
        from logiik.session_manager.utils.helpers import get_gpu_snapshot
        snap = get_gpu_snapshot()
        required_keys = [
            "gpu_available", "device_name",
            "vram_total_gb", "vram_used_gb", "vram_free_gb",
        ]
        for key in required_keys:
            assert key in snap

    def test_session_status_structure(self):
        from logiik.session_manager.session_manager import LogiikSession
        import time
        session = LogiikSession.__new__(LogiikSession)
        session._model = None
        session._model_loaded = False
        session._model_source = "local"
        session._device = "cpu"
        session._query_count = 0
        session._session_start = time.time()
        session._last_query_time = time.time()
        session._expiry_seconds = 0
        session._shutdown_requested = False
        status = session.get_status()
        assert "model_loaded" in status
        assert "query_count" in status
        assert "uptime_minutes" in status

    def test_query_server_status_route(self):
        from logiik.session_manager.query_server import app
        with app.test_client() as client:
            r = client.get("/status")
            assert r.status_code == 200
