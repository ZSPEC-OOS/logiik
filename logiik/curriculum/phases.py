"""
Logiik Curriculum Phase Definitions.

Single source of truth for all phase metadata, prompts,
generative ratios, and completion thresholds.

Phase numbering:
  1  — Memorization
  2  — Generation
  3  — Abstraction
  4  — Engineering Execution & Reliability
  5  — Coding Mastery
  6  — Scientific Reasoning & Experimental Design
  7  — Niche Scientific Reasoning          [NEW — replaces Drosophila]
  8  — Scientific Image Analysis
  9  — PDF / Textbook Ingestion
  10 — Synthetic Judgment

Legacy note:
  Phase 7 previously targeted Drosophila melanogaster genetics.
  That implementation is archived at:
  _legacy_backup/cognita/training/curriculum.py
  (_generate_drosophila_framework_examples method)
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class PhaseConfig:
    """
    Complete configuration for a single curriculum phase.

    Attributes:
        id:                   Phase number (1-indexed).
        name:                 Machine-readable phase name.
        display_name:         Human-readable phase name.
        generative_ratio:     Fraction of examples requiring generation
                              rather than selection (0–1).
        duration:             Fraction of total training budget allocated
                              to this phase. None = runs until completion
                              criterion met.
        description:          What the model learns in this phase.
        question_types:       Categories of questions generated.
        correctness_threshold: Minimum correctness score (0–1) required
                              before a student answer is accepted.
                              Used in teacher feedback loop.
        teacher_student:      True if phase uses iterative teacher
                              feedback loop.
        completion_criteria:  Dict describing how phase completion
                              is determined.
        metadata:             Arbitrary phase-specific config.
    """
    id: int
    name: str
    display_name: str
    generative_ratio: float
    description: str
    question_types: List[str]
    duration: Optional[float] = None
    correctness_threshold: float = 0.85
    teacher_student: bool = False
    completion_criteria: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ─── Phase definitions ────────────────────────────────────────────────────────

PHASES: List[PhaseConfig] = [

    PhaseConfig(
        id=1,
        name="memorization",
        display_name="Memorization",
        generative_ratio=0.10,
        duration=0.20,
        description=(
            "Model learns to reproduce teacher Q+A pairs accurately. "
            "Foundation for all subsequent phases."
        ),
        question_types=[
            "factual_recall",
            "definition",
            "identification",
        ],
        correctness_threshold=0.80,
        teacher_student=False,
        completion_criteria={
            "coverage_ratio": 0.95,
            "saturation_score": 0.90,
        },
    ),

    PhaseConfig(
        id=2,
        name="generation",
        display_name="Generation",
        generative_ratio=0.50,
        duration=0.20,
        description=(
            "Model generates original answers rather than selecting "
            "from provided options. Develops independent reasoning."
        ),
        question_types=[
            "open_ended",
            "explanation",
            "synthesis",
        ],
        correctness_threshold=0.82,
        teacher_student=False,
        completion_criteria={
            "coverage_ratio": 0.95,
            "saturation_score": 0.90,
        },
    ),

    PhaseConfig(
        id=3,
        name="abstraction",
        display_name="Abstraction",
        generative_ratio=0.80,
        duration=0.15,
        description=(
            "Cross-domain synthesis. Model connects concepts across "
            "disciplines and generalises beyond training examples."
        ),
        question_types=[
            "cross_domain_synthesis",
            "analogy",
            "generalisation",
        ],
        correctness_threshold=0.83,
        teacher_student=False,
        completion_criteria={
            "coverage_ratio": 0.95,
            "saturation_score": 0.90,
        },
    ),

    PhaseConfig(
        id=4,
        name="engineering_execution_reliability",
        display_name="Engineering Execution & Reliability",
        generative_ratio=0.85,
        duration=0.15,
        description=(
            "Implementation planning, test design, failure mode analysis, "
            "and reliability/security/performance trade-off reasoning."
        ),
        question_types=[
            "implementation_planning",
            "test_design",
            "failure_mode_analysis",
            "trade_off_reasoning",
        ],
        correctness_threshold=0.85,
        teacher_student=False,
        completion_criteria={
            "coverage_ratio": 0.95,
            "saturation_score": 0.90,
        },
    ),

    PhaseConfig(
        id=5,
        name="coding_mastery",
        display_name="Coding Mastery",
        generative_ratio=0.90,
        duration=0.12,
        description=(
            "Advanced coding tasks across common programming languages. "
            "Robust implementation, algorithm selection, debugging."
        ),
        question_types=[
            "algorithm_implementation",
            "debugging",
            "code_review",
            "language_comparison",
        ],
        correctness_threshold=0.87,
        teacher_student=False,
        completion_criteria={
            "coverage_ratio": 0.95,
            "saturation_score": 0.90,
        },
    ),

    PhaseConfig(
        id=6,
        name="scientific_reasoning_experimental_design",
        display_name="Scientific Reasoning & Experimental Design",
        generative_ratio=0.93,
        duration=0.10,
        description=(
            "Falsifiable hypothesis formation, experimental controls, "
            "confounder identification, uncertainty-aware conclusions."
        ),
        question_types=[
            "hypothesis_formation",
            "experimental_design",
            "confounder_identification",
            "statistical_interpretation",
            "uncertainty_quantification",
        ],
        correctness_threshold=0.88,
        teacher_student=False,
        completion_criteria={
            "coverage_ratio": 0.95,
            "saturation_score": 0.90,
        },
    ),

    PhaseConfig(
        id=7,
        name="niche_scientific_reasoning",
        display_name="Niche Scientific Reasoning",
        generative_ratio=0.94,
        duration=0.08,
        description=(
            "Deep reasoning on rare, interdisciplinary, or hypothetical "
            "scientific topics. Stepwise mechanistic reasoning with "
            "iterative teacher feedback until correctness >= 0.90."
        ),
        question_types=[
            "hypothetical_scenario",
            "multi_step_experimental_design",
            "interdisciplinary_synthesis",
            "mechanistic_reasoning",
            "rare_data_interpretation",
            "edge_case_analysis",
        ],
        correctness_threshold=0.90,   # Hard threshold — enforced in training loop
        teacher_student=True,          # Iterative feedback loop active
        completion_criteria={
            "coverage_ratio": 0.95,
            "saturation_score": 0.90,
            "max_iterations": 1000,
        },
        metadata={
            "legacy_phase": "drosophila_ai_framework",
            "legacy_file": "_legacy_backup/cognita/training/curriculum.py",
            "student_teacher_format": {
                "answer_steps": "List[str] — stepwise reasoning",
                "full_answer": "str — complete natural language answer",
                "feedback_fields": [
                    "correctness",
                    "feedback",
                    "suggested_improvement"
                ],
            },
        },
    ),

    PhaseConfig(
        id=8,
        name="scientific_image_analysis",
        display_name="Scientific Image Analysis",
        generative_ratio=0.94,
        duration=None,    # Runs until phase8_image_db saturated
        description=(
            "Extract, classify, embed, and deduplicate scientific images "
            "from PDFs (figures, diagrams, plots, microscopy, chemical "
            "structures). Feeds into Phase 9 to prevent duplicate ingestion."
        ),
        question_types=[
            "image_classification",
            "caption_interpretation",
            "figure_description",
            "diagram_reasoning",
        ],
        correctness_threshold=0.85,
        teacher_student=False,
        completion_criteria={
            "saturation_score": 0.90,
            "min_images_processed": 1,
        },
        metadata={
            "image_types": [
                "microscopy",
                "diagram",
                "plot",
                "chart",
                "chemical_structure",
                "other",
            ],
            "embedding_model": "BLIP-2",
            "deduplication_threshold": 0.90,
        },
    ),

    PhaseConfig(
        id=9,
        name="pdf_textbook_ingestion",
        display_name="PDF / Textbook Ingestion",
        generative_ratio=0.95,
        duration=None,    # Runs until ingestion queue empty
        description=(
            "Ingest text and images from PDFs and textbooks. "
            "Chunk, embed, deduplicate, and store in Pinecone + Firebase. "
            "Supports Lite Mode (<500 PDFs) and Full Mode (500-50k PDFs)."
        ),
        question_types=[
            "knowledge_extraction",
            "chunk_retrieval",
            "cross_modal_reasoning",
        ],
        correctness_threshold=0.85,
        teacher_student=False,
        completion_criteria={
            "queue_empty": True,
            "duplication_rate_below": 0.10,
        },
        metadata={
            "lite_mode": {
                "max_pdfs": 500,
                "chunk_size": 512,
                "overlap": 128,
                "batch_size": 5,
            },
            "full_mode": {
                "max_pdfs": 50000,
                "chunk_size": 1024,
                "overlap": 256,
                "batch_size": 20,
            },
            "embedding_model": "SPECTER2",
            "vector_backend": "pinecone",
            "text_backend": "firebase",
        },
    ),

    PhaseConfig(
        id=10,
        name="synthetic_judgment",
        display_name="Synthetic Judgment",
        generative_ratio=0.95,
        duration=None,    # Runs until all phase10 stages complete
        description=(
            "Train model to make scientifically valid decisions under "
            "uncertainty. Rewards decision quality, calibration, "
            "appropriate abstention, and logical consistency. "
            "Uses PPO via Hugging Face TRL."
        ),
        question_types=[
            "causal_reasoning",
            "consistency_check",
            "calibration",
            "abstention_decision",
            "belief_revision",
            "integrated_scientific_problem",
        ],
        correctness_threshold=0.90,
        teacher_student=False,
        completion_criteria={
            "all_stages_complete": True,
            "brier_score_below": 0.20,
            "abstention_precision_above": 0.80,
        },
        metadata={
            "rl_backend": "ppo_trl",
            "stages": [
                {"id": 1, "name": "phase10_stage_causal_reasoning"},
                {"id": 2, "name": "phase10_stage_consistency"},
                {"id": 3, "name": "phase10_stage_calibration"},
                {"id": 4, "name": "phase10_stage_abstention"},
                {"id": 5, "name": "phase10_stage_belief_revision"},
                {"id": 6, "name": "phase10_stage_integrated_scientific"},
            ],
            "reward_weights": {
                "correctness": 0.30,
                "consistency": 0.25,
                "calibration": 0.25,
                "abstention": 0.20,
            },
        },
    ),
]


# ─── Lookup helpers ───────────────────────────────────────────────────────────

def get_phase(phase_id: int) -> Optional[PhaseConfig]:
    """Return PhaseConfig by 1-indexed phase ID."""
    for phase in PHASES:
        if phase.id == phase_id:
            return phase
    return None


def get_phase_by_name(name: str) -> Optional[PhaseConfig]:
    """Return PhaseConfig by machine-readable name."""
    for phase in PHASES:
        if phase.name == name:
            return phase
    return None


def get_all_phase_names() -> List[str]:
    """Return ordered list of all phase display names."""
    return [p.display_name for p in PHASES]


def get_teacher_student_phases() -> List[PhaseConfig]:
    """Return phases that use the iterative teacher feedback loop."""
    return [p for p in PHASES if p.teacher_student]
