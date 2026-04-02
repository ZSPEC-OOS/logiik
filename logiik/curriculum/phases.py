"""
Logiik Curriculum Phase Definitions — Revised 12-Phase Structure.

Single source of truth for all phase metadata, prompts,
generative ratios, and completion thresholds.

Restructure summary vs. original 10-phase design:
  Added:   Phase 3  — Scientific Language & Literature
  Added:   Phase 4  — Mathematical & Statistical Reasoning
  Moved:   Phase 6  → Phase 5  (Scientific Reasoning)
  Moved:   Phase 7  → Phase 6  (Niche Scientific Reasoning)
  Moved:   Phase 8  → Phase 7  (Scientific Image Analysis)
  Replaced: Phase 5 → Phase 8  (Research Computing replaces
                                generic Coding Mastery)
  Moved:   Phase 4  → Phase 9  (Engineering Execution)
  Moved:   Phase 3  → Phase 10 (Abstraction, now grounded)
  Added:   Phase 11 — Adversarial Robustness & Epistemic Integrity
  Moved:   Phase 10 → Phase 12 (Synthetic Judgment, now grounded)
  Removed: Phase 9  — PDF ingestion demoted to background
                      infrastructure (not a curriculum phase)

Legacy note:
  Original phases archived at:
  _legacy_backup/cognita/training/curriculum.py
  Drosophila phase archived same location.
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
                              rather than selection (0-1).
        duration:             Fraction of total training budget. None
                              means phase runs until completion
                              criterion met.
        description:          What the model learns in this phase.
        question_types:       Categories of questions generated.
        correctness_threshold: Minimum correctness score before a
                              student answer is accepted.
        teacher_student:      True if phase uses iterative teacher
                              feedback loop.
        completion_criteria:  Dict describing phase completion logic.
        metadata:             Arbitrary phase-specific config.
        track:                Curriculum track this phase belongs to.
                              One of: foundation | language | domain |
                              execution | integration | capstone
    """
    id: int
    name: str
    display_name: str
    generative_ratio: float
    description: str
    question_types: List[str]
    track: str
    duration: Optional[float] = None
    correctness_threshold: float = 0.85
    teacher_student: bool = False
    completion_criteria: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ─── Phase definitions ────────────────────────────────────────────────────────

PHASES: List[PhaseConfig] = [

    # ── TRACK: FOUNDATION ────────────────────────────────────────────────────

    PhaseConfig(
        id=1,
        name="memorization",
        display_name="Memorization",
        track="foundation",
        generative_ratio=0.10,
        duration=0.10,
        description=(
            "Model learns to reproduce teacher Q+A pairs accurately. "
            "Establishes that questions have precise answers and that "
            "accuracy matters. Foundation for all subsequent phases."
        ),
        question_types=[
            "factual_recall",
            "definition",
            "identification",
            "term_to_concept_mapping",
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
        track="foundation",
        generative_ratio=0.50,
        duration=0.10,
        description=(
            "Model generates original answers rather than selecting "
            "from provided options. Develops independent response "
            "production beyond memorised patterns."
        ),
        question_types=[
            "open_ended",
            "explanation",
            "elaboration",
            "original_example_generation",
        ],
        correctness_threshold=0.82,
        teacher_student=False,
        completion_criteria={
            "coverage_ratio": 0.95,
            "saturation_score": 0.90,
        },
    ),

    # ── TRACK: LANGUAGE ──────────────────────────────────────────────────────

    PhaseConfig(
        id=3,
        name="scientific_language_literature",
        display_name="Scientific Language & Literature",
        track="language",
        generative_ratio=0.65,
        duration=0.10,
        description=(
            "Model learns the register, conventions, and logical "
            "structure of scientific writing. Covers abstract parsing, "
            "epistemic status classification, hedging language, "
            "citation conventions, and production of publication-grade "
            "scientific prose. Uses ingested PDF corpus as training "
            "data where available."
        ),
        question_types=[
            "abstract_parsing",
            "epistemic_status_classification",
            "claim_vs_evidence_identification",
            "hedging_language_interpretation",
            "methods_section_restatement",
            "statistical_language_interpretation",
            "scientific_argument_structure",
            "publication_register_prose_generation",
            "citation_convention_reasoning",
            "results_vs_conclusions_distinction",
        ],
        correctness_threshold=0.85,
        teacher_student=False,
        completion_criteria={
            "coverage_ratio": 0.95,
            "saturation_score": 0.88,
        },
        metadata={
            "uses_ingested_corpus": True,
            "corpus_query_phases": ["phase9"],
            "example_prompt_format": (
                "Classify each sentence in the following abstract "
                "excerpt as one of: [established_fact | "
                "experimental_result | hypothesis | speculation | "
                "methodological_claim]. Justify each classification."
            ),
            "register_targets": [
                "formal",
                "hedged",
                "precise",
                "citation_aware",
            ],
        },
    ),

    PhaseConfig(
        id=4,
        name="mathematical_statistical_reasoning",
        display_name="Mathematical & Statistical Reasoning",
        track="language",
        generative_ratio=0.70,
        duration=0.10,
        description=(
            "Model learns to interpret, evaluate, and reason about "
            "quantitative scientific evidence. Covers statistical "
            "output interpretation, experimental design evaluation "
            "for statistical validity, uncertainty propagation, "
            "data visualisation reasoning, Bayesian inference in "
            "scientific contexts, and detection of statistical errors "
            "in reported results."
        ),
        question_types=[
            "statistical_output_interpretation",
            "experimental_design_statistical_validity",
            "statistical_error_detection",
            "effect_size_vs_significance_distinction",
            "confidence_interval_interpretation",
            "bayesian_reasoning",
            "uncertainty_propagation",
            "data_visualisation_interpretation",
            "power_analysis_reasoning",
            "dimensional_analysis",
            "order_of_magnitude_estimation",
            "multiple_comparisons_reasoning",
        ],
        correctness_threshold=0.87,
        teacher_student=False,
        completion_criteria={
            "coverage_ratio": 0.95,
            "saturation_score": 0.88,
        },
        metadata={
            "latex_rendering_required": True,
            "example_prompt_format": (
                "A study reports: β=0.23, SE=0.19, p=0.23, n=31, "
                "R²=0.04. The authors conclude the treatment is "
                "effective. Evaluate this conclusion. What is wrong "
                "with this inference? What would you need to see to "
                "support the claim?"
            ),
            "domains": [
                "frequentist_statistics",
                "bayesian_statistics",
                "experimental_design",
                "data_visualisation",
                "numerical_methods",
            ],
        },
    ),

    # ── TRACK: DOMAIN ────────────────────────────────────────────────────────

    PhaseConfig(
        id=5,
        name="scientific_reasoning_experimental_design",
        display_name="Scientific Reasoning & Experimental Design",
        track="domain",
        generative_ratio=0.88,
        duration=0.09,
        description=(
            "Model develops rigorous scientific reasoning: falsifiable "
            "hypothesis formation, experimental controls, confounder "
            "identification, null hypothesis formulation, power "
            "analysis reasoning, pre-registration logic, and "
            "replication/reproducibility evaluation. Builds on "
            "statistical foundation from Phase 4."
        ),
        question_types=[
            "falsifiable_hypothesis_formation",
            "null_hypothesis_formulation",
            "experimental_control_design",
            "confounder_identification",
            "power_analysis_reasoning",
            "pre_registration_logic",
            "replication_reasoning",
            "reproducibility_evaluation",
            "uncertainty_aware_conclusions",
            "causal_vs_correlational_reasoning",
        ],
        correctness_threshold=0.88,
        teacher_student=False,
        completion_criteria={
            "coverage_ratio": 0.95,
            "saturation_score": 0.90,
        },
    ),

    PhaseConfig(
        id=6,
        name="niche_scientific_reasoning",
        display_name="Niche & Interdisciplinary Scientific Reasoning",
        track="domain",
        generative_ratio=0.93,
        duration=0.09,
        description=(
            "Deep reasoning on rare, interdisciplinary, or hypothetical "
            "scientific topics. Stepwise mechanistic reasoning with "
            "iterative teacher feedback until correctness >= 0.90. "
            "Covers biochemistry, molecular biology, genetics, "
            "neuroscience, ecology, materials science, climate science. "
            "Requires citing known molecular pathways or physical "
            "mechanisms — no generic placeholders."
        ),
        question_types=[
            "hypothetical_scenario",
            "multi_step_experimental_design",
            "interdisciplinary_synthesis",
            "mechanistic_reasoning",
            "rare_data_interpretation",
            "edge_case_analysis",
            "cross_disciplinary_integration",
            "molecular_pathway_reasoning",
            "physical_mechanism_reasoning",
        ],
        correctness_threshold=0.90,
        teacher_student=True,
        completion_criteria={
            "coverage_ratio": 0.95,
            "saturation_score": 0.90,
            "max_iterations": 1000,
        },
        metadata={
            "legacy_phase": "drosophila_ai_framework",
            "legacy_file": (
                "_legacy_backup/cognita/training/curriculum.py"
            ),
            "requires_mechanism_citation": True,
            "scientific_domains": [
                "biochemistry",
                "molecular_biology",
                "genetics",
                "neuroscience",
                "ecology",
                "materials_science",
                "climate_science",
                "biophysics",
                "cell_biology",
            ],
            "student_teacher_format": {
                "answer_steps": "List[str] — stepwise reasoning",
                "full_answer": "str — complete natural language answer",
                "feedback_fields": [
                    "correctness",
                    "feedback",
                    "suggested_improvement",
                ],
            },
        },
    ),

    PhaseConfig(
        id=7,
        name="scientific_image_data_analysis",
        display_name="Scientific Image & Data Analysis",
        track="domain",
        generative_ratio=0.93,
        duration=None,
        description=(
            "Model interprets and reasons about scientific figures, "
            "diagrams, plots, microscopy images, chemical structures, "
            "and experimental data tables. Includes figure limitation "
            "analysis and follow-up experiment suggestion. "
            "Uses Phase 8 image embeddings for retrieval."
        ),
        question_types=[
            "figure_interpretation",
            "figure_limitation_analysis",
            "follow_up_experiment_suggestion",
            "data_table_reasoning",
            "microscopy_interpretation",
            "chemical_structure_reasoning",
            "plot_critical_analysis",
            "image_classification",
            "caption_interpretation",
            "diagram_reasoning",
        ],
        correctness_threshold=0.88,
        teacher_student=False,
        completion_criteria={
            "saturation_score": 0.90,
            "min_images_processed": 1,
        },
        metadata={
            "embedding_model": "BLIP-2",
            "deduplication_threshold": 0.90,
            "image_types": [
                "microscopy",
                "diagram",
                "plot",
                "chart",
                "chemical_structure",
                "other",
            ],
        },
    ),

    # ── TRACK: EXECUTION ─────────────────────────────────────────────────────

    PhaseConfig(
        id=8,
        name="research_computing_scientific_coding",
        display_name="Research Computing & Scientific Coding",
        track="execution",
        generative_ratio=0.92,
        duration=0.09,
        description=(
            "Model learns research-specific coding: scientific data "
            "analysis pipelines, numerical methods, bioinformatics "
            "tooling, reproducible research software engineering, "
            "and GPU-accelerated scientific computing. Explicitly "
            "NOT generic coding mastery — every task is grounded "
            "in a scientific research context."
        ),
        question_types=[
            "scientific_data_pipeline",
            "statistical_analysis_implementation",
            "biological_data_format_parsing",
            "reproducible_analysis_pipeline",
            "numerical_method_implementation",
            "numerical_stability_reasoning",
            "vectorised_scientific_computation",
            "gpu_accelerated_computing",
            "bioinformatics_tool_interfacing",
            "molecular_structure_manipulation",
            "scientific_unit_testing",
            "research_logging_versioning",
            "scientific_bug_diagnosis",
            "containerisation_reproducibility",
        ],
        correctness_threshold=0.90,
        teacher_student=False,
        completion_criteria={
            "coverage_ratio": 0.95,
            "saturation_score": 0.90,
        },
        metadata={
            "primary_languages": ["python", "r", "bash"],
            "key_libraries": [
                "numpy", "scipy", "pandas", "matplotlib",
                "scikit-learn", "pytorch", "biopython",
                "tidyverse", "ggplot2",
            ],
            "replaces_legacy_phase": "coding_mastery",
            "example_prompt_format": (
                "You have a pandas DataFrame with columns: "
                "[sample_id, treatment, timepoint, expression_value]. "
                "Some expression_value entries are negative "
                "(instrument artifact). Write code to: "
                "(1) identify and flag artifacts, "
                "(2) compute per-treatment mean expression at each "
                "timepoint excluding artifacts, "
                "(3) perform a paired t-test between treatment groups "
                "at the final timepoint, "
                "(4) produce a publication-ready results summary "
                "with effect size and confidence interval."
            ),
        },
    ),

    PhaseConfig(
        id=9,
        name="engineering_execution_reliability",
        display_name="Engineering Execution & Reliability",
        track="execution",
        generative_ratio=0.90,
        duration=0.08,
        description=(
            "Implementation planning, test design, failure mode "
            "analysis, and reliability/security/performance trade-off "
            "reasoning. Contextualised for research infrastructure: "
            "pipeline reliability, data integrity, experiment "
            "reproducibility systems, and scientific computing "
            "failure modes including silent numerical errors and "
            "non-deterministic results."
        ),
        question_types=[
            "implementation_planning",
            "test_design",
            "failure_mode_analysis",
            "trade_off_reasoning",
            "research_pipeline_reliability",
            "data_integrity_verification",
            "experiment_reproducibility_systems",
            "silent_numerical_error_detection",
            "security_data_governance",
            "performance_profiling",
        ],
        correctness_threshold=0.88,
        teacher_student=False,
        completion_criteria={
            "coverage_ratio": 0.95,
            "saturation_score": 0.90,
        },
        metadata={
            "research_context": True,
            "legacy_phase_id": 4,
        },
    ),

    # ── TRACK: INTEGRATION ───────────────────────────────────────────────────

    PhaseConfig(
        id=10,
        name="abstraction_cross_domain_synthesis",
        display_name="Abstraction & Cross-Domain Synthesis",
        track="integration",
        generative_ratio=0.94,
        duration=0.07,
        description=(
            "Cross-domain synthesis now grounded by genuine domain "
            "knowledge from Phases 3-9. Model connects mechanisms "
            "across disciplines using precise domain language from "
            "both fields, identifies methodological transfer "
            "opportunities, synthesises conflicting findings, and "
            "generates novel hypotheses at disciplinary boundaries "
            "with explicit mechanistic grounding."
        ),
        question_types=[
            "cross_discipline_mechanism_connection",
            "methodological_transfer",
            "conflicting_findings_synthesis",
            "boundary_hypothesis_generation",
            "analogical_reasoning_with_mechanism",
            "interdisciplinary_experimental_design",
            "knowledge_gap_identification",
        ],
        correctness_threshold=0.90,
        teacher_student=False,
        completion_criteria={
            "coverage_ratio": 0.95,
            "saturation_score": 0.90,
        },
        metadata={
            "requires_domain_grounding": True,
            "minimum_prior_phases_complete": 9,
            "legacy_phase_id": 3,
            "note": (
                "Relocated from Phase 3 to Phase 10. At Phase 3 "
                "the model had no domain depth — connections would "
                "have been superficial. At Phase 10 they are "
                "substantive."
            ),
        },
    ),

    PhaseConfig(
        id=11,
        name="adversarial_robustness_epistemic_integrity",
        display_name="Adversarial Robustness & Epistemic Integrity",
        track="integration",
        generative_ratio=0.95,
        duration=None,
        description=(
            "Model is trained to resist confident but incorrect "
            "scientific claims, detect methodological flaws, identify "
            "data fabrication indicators, maintain appropriate "
            "uncertainty under pressure to overclaim, and distinguish "
            "rhetorical from evidential arguments. Critical for "
            "research use where misleading inputs are plausible."
        ),
        question_types=[
            "methodological_flaw_identification",
            "data_fabrication_detection",
            "harking_detection",
            "citation_manipulation_detection",
            "overgeneralisation_identification",
            "inappropriate_extrapolation_detection",
            "false_premise_identification",
            "reproducibility_risk_assessment",
            "rhetorical_vs_evidential_argument",
            "confidence_resistance",
            "uncertainty_maintenance_under_pressure",
            "flawed_assumption_exposure",
        ],
        correctness_threshold=0.91,
        teacher_student=True,
        completion_criteria={
            "coverage_ratio": 0.95,
            "saturation_score": 0.91,
            "max_iterations": 1000,
        },
        metadata={
            "adversarial_inputs": True,
            "example_prompt_format": (
                "A researcher presents the following finding with "
                "high confidence: [plausible but methodologically "
                "flawed abstract]. They ask you to help design a "
                "follow-up study assuming this finding is correct. "
                "Evaluate whether the finding warrants this confidence "
                "before proceeding."
            ),
            "fabrication_indicators": [
                "too_clean_results",
                "impossible_standard_deviations",
                "suspicious_digit_patterns",
                "terminal_digit_preference",
                "implausible_effect_sizes",
            ],
        },
    ),

    # ── TRACK: CAPSTONE ──────────────────────────────────────────────────────

    PhaseConfig(
        id=12,
        name="synthetic_judgment",
        display_name="Synthetic Judgment",
        track="capstone",
        generative_ratio=0.95,
        duration=None,
        description=(
            "Model makes scientifically valid decisions under "
            "uncertainty. Rewarded for decision quality, calibration, "
            "appropriate abstention, and logical consistency. "
            "Now properly grounded by Phases 3-11: uses real "
            "statistical language, domain-specific mechanisms, "
            "adversarial robustness, and belief revision under "
            "conditions where prior beliefs may have been based "
            "on flawed science."
        ),
        question_types=[
            "causal_reasoning",
            "consistency_check",
            "calibrated_probability_estimation",
            "abstention_decision",
            "belief_revision",
            "integrated_scientific_problem",
            "flawed_prior_belief_revision",
            "adversarial_scenario_judgment",
        ],
        correctness_threshold=0.92,
        teacher_student=False,
        completion_criteria={
            "all_stages_complete": True,
            "brier_score_below": 0.20,
            "abstention_precision_above": 0.80,
        },
        metadata={
            "rl_backend": "ppo_trl",
            "grounded_by_phases": [3, 4, 5, 6, 7, 8, 9, 10, 11],
            "stages": [
                {"id": 1, "name": "phase12_stage_causal_reasoning"},
                {"id": 2, "name": "phase12_stage_consistency"},
                {"id": 3, "name": "phase12_stage_calibration"},
                {"id": 4, "name": "phase12_stage_abstention"},
                {"id": 5, "name": "phase12_stage_belief_revision"},
                {"id": 6, "name": "phase12_stage_integrated_scientific"},
            ],
            "reward_weights": {
                "correctness": 0.30,
                "consistency": 0.25,
                "calibration": 0.25,
                "abstention": 0.20,
            },
            "legacy_phase_id": 10,
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


def get_phases_by_track(track: str) -> List[PhaseConfig]:
    """
    Return all phases belonging to a curriculum track.
    Tracks: foundation | language | domain | execution |
            integration | capstone
    """
    return [p for p in PHASES if p.track == track]


def get_all_phase_names() -> List[str]:
    """Return ordered list of all phase display names."""
    return [p.display_name for p in PHASES]


def get_teacher_student_phases() -> List[PhaseConfig]:
    """Return phases that use iterative teacher feedback loop."""
    return [p for p in PHASES if p.teacher_student]


def get_phases_requiring_corpus() -> List[PhaseConfig]:
    """Return phases that query the ingested PDF corpus."""
    return [
        p for p in PHASES
        if p.metadata.get("uses_ingested_corpus", False)
    ]
