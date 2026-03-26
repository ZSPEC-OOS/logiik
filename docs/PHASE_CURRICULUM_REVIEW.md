# Curriculum Phase Readiness Review (March 26, 2026)

## Question Reviewed
Whether the current 5-phase curriculum is sufficient to reliably produce an AI that can achieve:
- **Phase 4: Coding Mastery**
- **Phase 5: Drosophila AI Framework**

## Executive Determination
**Partially sufficient, but not robust enough for consistent success.**  
The current progression (Memorization → Generation → Abstraction → Coding Mastery → Drosophila) has a strong conceptual arc, but it skips two high-risk capability bridges that are typically required in practice:

1. **Production-grade software execution bridge** between abstraction and coding mastery.
2. **Scientific method + bioscience grounding bridge** between coding mastery and Drosophila specialization.

## Phase-by-Phase Review

### Phase 1 — Memorization
**What works**
- Builds factual substrate and response format compliance.
- Supports teacher-student distillation and broad concept ingestion.

**Gaps**
- Limited transfer unless retrieval and grounding behavior is explicitly reinforced.
- Memorized patterns can overfit shallow answer templates.

**Verdict**
- Necessary, but insufficient on its own for downstream engineering/science objectives.

---

### Phase 2 — Generation
**What works**
- Introduces open-ended answer production.
- Improves fluency and robustness beyond multiple-choice framing.

**Gaps**
- “Generate” prompts do not force verifiable correctness or tool-mediated checks.
- Hallucination risk grows if evaluation is primarily style/coherence based.

**Verdict**
- Good expansion phase, but needs stronger correctness instrumentation.

---

### Phase 3 — Abstraction
**What works**
- Encourages cross-domain synthesis and conceptual transfer.
- Prepares for non-trivial reasoning.

**Gaps**
- Still too far from the operational demands of production coding:
  - debugging loops
  - multi-file edits
  - test-driven iteration
  - failure recovery
- Still too far from bioscience specialization demands:
  - hypothesis quality control
  - causal/mechanistic reasoning under uncertainty
  - evidence/citation discipline

**Verdict**
- Strong but broad. Needs specialization bridges before Phase 4 and Phase 5.

---

### Phase 4 — Coding Mastery (current)
**Strengths**
- Topic list covers language breadth and key engineering concepts.

**Risk**
- Breadth-first language exposure can produce shallow competency without iterative build-test-debug workflows and architecture-level reasoning checkpoints.

**Verdict**
- Ambitious and appropriate as a target phase, but currently under-supported by prerequisite operational training.

---

### Phase 5 — Drosophila AI Framework (current)
**Strengths**
- Domain scope is advanced and well-articulated (genetics, neural wiring, toolchain, evaluation).

**Risk**
- Jump from generic coding to deep biological framework design is too large unless the model is first trained for:
  - scientific evidence handling
  - mechanistic/causal validation
  - uncertainty calibration and contradiction resolution

**Verdict**
- Correct end-goal, but needs a dedicated pre-specialization science phase.

## Recommendation: Add Two Bridge Phases

### New Phase 4 (Bridge A): Engineering Execution & Reliability
**Purpose**
- Convert conceptual coding ability into reliable software delivery behavior.

**Core competencies**
- Spec decomposition, planning, and milestone execution.
- Multi-file refactors and dependency-aware edits.
- Test-first or test-guided development loops.
- Static analysis, security checks, and regression control.
- Tool use discipline (linters, tests, profilers, CI-like gates).
- Debugging under failing tests/log traces.

**Exit criteria**
- Sustained pass rate on hidden tests.
- Lower defect injection over iterative tasks.
- Demonstrated ability to recover from failed attempts.

---

### New Phase 6 (Bridge B, after Coding Mastery): Scientific Reasoning & Experimental Design
**Purpose**
- Prepare for Drosophila framework specialization with scientific rigor.

**Core competencies**
- Hypothesis formulation with falsifiable predictions.
- Causal graph reasoning and confounder identification.
- Experimental design quality (controls, replicates, endpoints).
- Evidence grading, contradiction handling, and citation-grounded answers.
- Uncertainty quantification and confidence calibration.
- Structured genotype → pathway → circuit → behavior reasoning templates.

**Exit criteria**
- Mechanistic reasoning benchmark pass thresholds.
- Citation-grounded answer compliance.
- Reduced unsupported claims in science tasks.

## Proposed Revised Sequence
1. Memorization  
2. Generation  
3. Abstraction  
4. **Engineering Execution & Reliability** (new)  
5. Coding Mastery (existing, repositioned)  
6. **Scientific Reasoning & Experimental Design** (new)  
7. Drosophila AI Framework (existing, repositioned)

## Final Conclusion
If the objective is **reliable** achievement of current Phases 4 and 5, the present 5-phase structure is **not fully sufficient**.  
You should add the two bridge phases above before expecting consistent performance at the coding mastery and Drosophila framework levels.
