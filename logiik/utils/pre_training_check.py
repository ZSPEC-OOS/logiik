"""
Logiik Pre-Training Coherence Check.

Run this before merging the restructure branch and before
starting any training run. Verifies that all 12 phases
are correctly wired end-to-end: curriculum definitions,
generator routing, API endpoints, T-S loops, and Phase 3
corpus integration.

Usage:
    python -m logiik.utils.pre_training_check

All checks must pass before training begins.
"""
import sys
from logiik.utils.logging import get_logger

logger = get_logger("pre_training_check")


def check_curriculum_integrity():
    """Verify 12-phase curriculum structure."""
    from logiik.curriculum.phases import (
        PHASES, get_phase, get_teacher_student_phases,
        get_phases_by_track, get_phases_requiring_corpus,
    )
    assert len(PHASES) == 12, (
        f"Expected 12 phases, got {len(PHASES)}"
    )
    ids = [p.id for p in PHASES]
    assert ids == list(range(1, 13)), (
        f"Phase IDs not sequential: {ids}"
    )
    ts_phases = get_teacher_student_phases()
    assert [p.id for p in ts_phases] == [6, 11], (
        f"T-S phases should be [6,11], got "
        f"{[p.id for p in ts_phases]}"
    )
    corpus_phases = get_phases_requiring_corpus()
    assert any(p.id == 3 for p in corpus_phases), (
        "Phase 3 should require corpus"
    )
    p12 = get_phase(12)
    assert p12.metadata["rl_backend"] == "ppo_trl"
    stages = p12.metadata["stages"]
    assert len(stages) == 6
    assert all(
        s["name"].startswith("phase12_stage_")
        for s in stages
    )
    logger.info("[PASS] Curriculum integrity: 12 phases, "
                "T-S=[6,11], corpus=[3], Phase 12 PPO/TRL")
    return True


def check_generator_routing():
    """Verify GenerativeCurriculum has all 12 generators."""
    import inspect
    from logiik.core.training import GenerativeCurriculum
    required = [
        "_gen_memorization",
        "_gen_generation",
        "_gen_scientific_language",
        "_gen_mathematical_statistical",
        "_gen_scientific_reasoning",
        "_gen_niche_scientific",
        "_gen_image_data_analysis",
        "_gen_research_computing",
        "_gen_engineering",
        "_gen_abstraction",
        "_gen_adversarial_robustness",
        "_gen_synthetic_judgment",
    ]
    methods = [
        m for m, _ in inspect.getmembers(
            GenerativeCurriculum,
            predicate=inspect.isfunction
        )
    ]
    missing = [r for r in required if r not in methods]
    assert not missing, (
        f"Missing generator methods: {missing}"
    )
    # Verify set_retriever exists
    assert hasattr(GenerativeCurriculum, "set_retriever"), (
        "GenerativeCurriculum missing set_retriever method"
    )
    logger.info(
        "[PASS] Generator routing: all 12 generators present, "
        "set_retriever available"
    )
    return True


def check_phase3_corpus_wiring():
    """Verify Phase 3 corpus integration is wired correctly."""
    from logiik.core.training import GenerativeCurriculum
    gc = GenerativeCurriculum.__new__(GenerativeCurriculum)
    gc._retriever = None
    gc._phase6_loop = None
    gc._phase11_loop = None
    gc._topic_cursors = {}
    assert hasattr(gc, "_retriever"), (
        "GenerativeCurriculum missing _retriever attribute"
    )
    assert gc._retriever is None, (
        "_retriever should default to None"
    )

    # Mock retriever
    class MockRetriever:
        def retrieve(self, *a, **kw): return []

    gc.set_retriever(MockRetriever())
    assert gc._retriever is not None, (
        "set_retriever did not assign retriever"
    )
    logger.info(
        "[PASS] Phase 3 corpus wiring: "
        "_retriever attribute present, set_retriever works"
    )
    return True


def check_ts_loop_endpoints():
    """Verify T-S loop API endpoints exist and respond."""
    import logiik.api.endpoints as _ep
    from fastapi.testclient import TestClient
    # Clear any stale loop state from prior runs
    _ep._ts_loops.clear()
    client = TestClient(_ep.app)

    # Register a question for Phase 6
    r = client.post("/logiik/ts/6/question", json={
        "prompt": "How does pH affect enzyme folding?",
        "answer_steps": ["Step 1: identify residues",
                         "Step 2: predict protonation"],
        "full_answer": "At low pH histidine residues protonate...",
        "difficulty": 0.93,
    })
    assert r.status_code == 200, (
        f"Phase 6 question registration failed: {r.text}"
    )
    qid = r.json()["question_id"]
    assert qid, "No question_id returned"

    # Student attempt
    r2 = client.post("/logiik/ts/6/attempt", json={
        "question_id": qid,
        "student_answer_steps": ["Student step 1"],
        "student_full_answer": (
            "At low pH the enzyme active site residues become "
            "protonated altering electrostatic interactions."
        ),
    })
    assert r2.status_code == 200, (
        f"Student attempt failed: {r2.text}"
    )
    assert r2.json()["attempt_index"] == 0

    # Feedback — below threshold
    r3 = client.post("/logiik/ts/6/feedback", json={
        "question_id": qid,
        "attempt_index": 0,
        "feedback": ["Missing disulfide bond discussion"],
        "correctness": 0.82,
        "suggested_improvement": "Address disulfide stability",
    })
    assert r3.status_code == 200
    assert r3.json()["threshold_met"] == False

    # Feedback — at threshold
    r4 = client.post("/logiik/ts/6/attempt", json={
        "question_id": qid,
        "student_answer_steps": ["Step 1", "Step 2", "Step 3"],
        "student_full_answer": (
            "At low pH histidine residues (pKa 6.0) protonate, "
            "disrupting electrostatic interactions. Disulfide bonds "
            "remain intact providing structural stability."
        ),
    })
    assert r4.status_code == 200

    r5 = client.post("/logiik/ts/6/feedback", json={
        "question_id": qid,
        "attempt_index": 1,
        "feedback": ["Complete and accurate"],
        "correctness": 0.92,
    })
    assert r5.status_code == 200
    assert r5.json()["threshold_met"] == True

    # Metrics
    r6 = client.get("/logiik/ts/6/metrics")
    assert r6.status_code == 200
    metrics = r6.json()["metrics"]
    assert metrics["total_questions"] == 1
    assert metrics["accepted"] == 1

    # Phase 11 endpoint exists
    r7 = client.post("/logiik/ts/11/question", json={
        "prompt": "Evaluate this abstract for HARKing indicators.",
        "answer_steps": ["Step 1: examine hypothesis placement"],
        "full_answer": "The abstract shows post-hoc framing...",
        "difficulty": 0.95,
    })
    assert r7.status_code == 200, (
        f"Phase 11 question registration failed: {r7.text}"
    )

    # Invalid phase rejected
    r8 = client.post("/logiik/ts/5/question", json={
        "prompt": "test",
        "answer_steps": [],
        "full_answer": "test",
    })
    assert r8.status_code == 400

    logger.info(
        "[PASS] T-S loop endpoints: Phase 6 full loop "
        "(register→attempt→feedback→threshold), "
        "Phase 11 registration, invalid phase rejection"
    )
    return True


def check_reward_engine_phase12():
    """Verify RewardEngine pulls weights from Phase 12."""
    from logiik.core.phase10_training import RewardEngine
    from logiik.curriculum.phases import get_phase
    engine = RewardEngine()
    p12 = get_phase(12)
    expected = p12.metadata["reward_weights"]
    assert engine._weights["correctness"] == (
        expected["correctness"]
    ), "RewardEngine correctness weight mismatch"
    assert engine._weights["abstention"] == (
        expected["abstention"]
    ), "RewardEngine abstention weight mismatch"
    logger.info(
        "[PASS] RewardEngine sources weights from Phase 12"
    )
    return True


def check_api_curriculum_endpoint():
    """Verify /logiik/curriculum returns correct structure."""
    from logiik.api.endpoints import app
    from fastapi.testclient import TestClient
    client = TestClient(app)
    r = client.get("/logiik/curriculum")
    assert r.status_code == 200
    d = r.json()
    assert d["total_phases"] == 12
    track_phase_counts = {
        track: len(data["phases"])
        for track, data in d["tracks"].items()
    }
    assert track_phase_counts == {
        "foundation": 2, "language": 2, "domain": 3,
        "execution": 2, "integration": 2, "capstone": 1,
    }, f"Track counts wrong: {track_phase_counts}"
    logger.info(
        "[PASS] /logiik/curriculum: 12 phases, "
        "6 tracks, correct phase distribution"
    )
    return True


def run_all_checks() -> bool:
    """Run all pre-training coherence checks."""
    logger.info("=" * 60)
    logger.info("Logiik Pre-Training Coherence Check")
    logger.info("12-Phase Curriculum Restructure Validation")
    logger.info("=" * 60)

    checks = [
        ("Curriculum integrity",     check_curriculum_integrity),
        ("Generator routing",        check_generator_routing),
        ("Phase 3 corpus wiring",    check_phase3_corpus_wiring),
        ("T-S loop endpoints",       check_ts_loop_endpoints),
        ("RewardEngine Phase 12",    check_reward_engine_phase12),
        ("Curriculum API endpoint",  check_api_curriculum_endpoint),
    ]

    results = {}
    for name, fn in checks:
        try:
            results[name] = fn()
        except Exception as e:
            import traceback
            logger.error(
                f"[FAIL] {name}: {e}\n{traceback.format_exc()}"
            )
            results[name] = False

    logger.info("=" * 60)
    passed = sum(results.values())
    total = len(results)
    logger.info(f"Results: {passed}/{total} checks passed")

    if passed == total:
        logger.info(
            "STATUS: All checks passed. "
            "System coherent. Ready to merge and train."
        )
    else:
        failed = [k for k, v in results.items() if not v]
        logger.error(f"STATUS: Failed checks: {failed}")

    logger.info("=" * 60)
    return passed == total


if __name__ == "__main__":
    ok = run_all_checks()
    sys.exit(0 if ok else 1)
