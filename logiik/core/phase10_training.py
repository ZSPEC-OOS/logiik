"""
Logiik Phase 10 — Synthetic Judgment.
"""
import random
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import networkx as nx

from logiik.curriculum.phases import get_phase
from logiik.utils.logging import get_logger, log_event

logger = get_logger("core.phase10_training")

PHASE10_STAGES = [
    "phase10_stage_causal_reasoning",
    "phase10_stage_consistency",
    "phase10_stage_calibration",
    "phase10_stage_abstention",
    "phase10_stage_belief_revision",
    "phase10_stage_integrated_scientific",
]

@dataclass
class ReasoningStep:
    step_id: int
    type: str
    content: str
    confidence: float
    dependencies: List[int] = field(default_factory=list)
    evidence_for: List[str] = field(default_factory=list)
    evidence_against: List[str] = field(default_factory=list)
    contradiction_score: float = 0.0

@dataclass
class ModelOutput:
    decision: str
    final_answer: Optional[str]
    reasoning_chain: List[ReasoningStep]
    confidence: float
    uncertainty: Dict[str, float]
    missing_information: List[str] = field(default_factory=list)

class CausalScenario:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.ground_truth: Optional[str] = None
        self.knowable: bool = True
        self.question: str = ""

    def generate(self, n_nodes: int = 5, edge_prob: float = 0.3, inject_confounder: bool = False) -> "CausalScenario":
        nodes = [f"X{i}" for i in range(n_nodes)]
        self.graph.clear()
        self.graph.add_nodes_from(nodes)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if random.random() < edge_prob:
                    self.graph.add_edge(nodes[i], nodes[j])
        if inject_confounder:
            self._inject_confounder()
        self._generate_question()
        return self

    def _inject_confounder(self):
        nodes = list(self.graph.nodes)
        if len(nodes) < 2:
            return
        n1, n2 = random.sample(nodes, 2)
        self.graph.add_node("Z_hidden")
        self.graph.add_edge("Z_hidden", n1)
        self.graph.add_edge("Z_hidden", n2)
        self.knowable = False
        logger.debug(f"Confounder injected: Z_hidden -> {n1}, Z_hidden -> {n2}")

    def _generate_question(self):
        nodes = [n for n in self.graph.nodes if not n.startswith("Z_")]
        if not nodes:
            self.question = "What can be inferred from this system?"
            return
        q_types = [
            ("intervention", f"What happens to {random.choice(nodes)} if {random.choice(nodes)} is removed?"),
            ("counterfactual", f"What if {random.choice(nodes)} had been doubled?"),
            ("confounder", f"Is the relationship between {random.choice(nodes)} and {random.choice(nodes)} causal or confounded?"),
            ("mechanism", f"What is the causal pathway from {nodes[0]} to {nodes[-1]}?"),
        ]
        _, self.question = random.choice(q_types)
        self.ground_truth = f"Derived from graph structure: {self.question}"

    def to_prompt(self) -> str:
        edges = list(self.graph.edges)
        observable = [(u, v) for u, v in edges if not u.startswith("Z_") and not v.startswith("Z_")]
        edge_desc = ", ".join([f"{u} -> {v}" for u, v in observable]) or "no observable relationships"
        return (
            f"Causal System:\n"
            f"Observable variables: {[n for n in self.graph.nodes if not n.startswith('Z_')]}\n"
            f"Observed relationships: {edge_desc}\n\n"
            f"Question: {self.question}"
        )


class ScenarioGenerator:
    SCIENTIFIC_PROBLEMS = [
        {
            "prompt": (
                "Given:\n"
                "- Enzyme E increases when gene X is active\n"
                "- Knockout of X reduces E by 80%\n"
                "- Drug D increases E without affecting X expression\n\n"
                "Question: What is the most likely mechanism by which Drug D increases E?"
            ),
            "ground_truth": (
                "Drug D acts downstream of X or via an independent pathway to increase E. "
                "Possible mechanisms: post-translational stabilisation of E, activation of a "
                "parallel transcription factor, or inhibition of E degradation."
            ),
            "knowable": True,
        },
        {
            "prompt": (
                "Given:\n"
                "- Protein P misfolds at pH < 5.0\n"
                "- Histidine residues (pKa ~6.0) are present in the active site\n"
                "- Disulfide bonds are intact across all pH values\n\n"
                "Question: What is the molecular mechanism of pH-dependent misfolding?"
            ),
            "ground_truth": (
                "At pH < 5.0, histidine residues become protonated (positively charged), "
                "disrupting electrostatic interactions in the active site and destabilising the "
                "native fold. Disulfide bonds provide structural stability but cannot compensate "
                "for electrostatic disruption."
            ),
            "knowable": True,
        },
        {
            "prompt": (
                "Experiment: Treatment A increases cell proliferation by 40% in vitro. "
                "Treatment B decreases apoptosis by 30% in vitro. Combined A+B shows only "
                "35% proliferation increase.\n\n"
                "Question: Is the combination effect additive, synergistic, or antagonistic? "
                "What mechanism could explain this?"
            ),
            "ground_truth": (
                "The combination is antagonistic — expected additive effect would be ~55-60% "
                "increase but only 35% is observed. Possible mechanism: A and B share a "
                "downstream effector, or B's anti-apoptotic effect is partially dependent on "
                "the same pathway A activates."
            ),
            "knowable": True,
        },
        {
            "prompt": (
                "A study reports a correlation of r=0.85 between biomarker B and disease "
                "severity D in a cohort of n=45 patients. No control group was included.\n\n"
                "Question: Can you conclude that B causes D? What additional experiments are needed?"
            ),
            "ground_truth": (
                "Correlation does not imply causation. The study cannot establish causality "
                "without: (1) a control group, (2) temporal precedence of B before D, "
                "(3) ruling out confounders, (4) experimental manipulation of B. The small "
                "n=45 also limits statistical power."
            ),
            "knowable": True,
        },
    ]

    def generate(self, stage: str, count: int = 1) -> List[Tuple[str, str, bool]]:
        scenarios = []
        for _ in range(count):
            scenario = self._generate_one(stage)
            scenarios.append(scenario)
        return scenarios

    def _generate_one(self, stage: str) -> Tuple[str, str, bool]:
        if stage == "phase10_stage_causal_reasoning":
            s = CausalScenario().generate(n_nodes=random.randint(4, 7), inject_confounder=False)
            return s.to_prompt(), s.ground_truth or "", True

        elif stage == "phase10_stage_consistency":
            s = CausalScenario().generate(n_nodes=5)
            prompt = (
                s.to_prompt() +
                "\n\nAdditional claim: X0 has no effect on any other variable.\n\n"
                "Question: Is this additional claim consistent with the observed relationships?"
            )
            return prompt, "Contradiction exists — evaluate consistency.", True

        elif stage == "phase10_stage_calibration":
            prob = round(random.uniform(0.1, 0.9), 2)
            outcome = random.choice([0, 1])
            prompt = (
                f"Based on the following evidence summary, estimate the probability that "
                f"the hypothesis is true.\n"
                f"Evidence strength: {prob:.0%} of studies support it.\n"
                f"Sample sizes: mostly n<100.\n"
                f"Replication: 2 of 5 independent replications succeeded.\n\n"
                f"Question: What probability do you assign to this hypothesis being true?"
            )
            return prompt, str(prob * 0.6), True

        elif stage == "phase10_stage_abstention":
            prompt = (
                "Given:\n"
                "- Variable Y was measured at time T\n"
                "- No baseline measurement exists\n"
                "- Confounders were not recorded\n"
                "- Sample size: n=3\n\n"
                "Question: What is the causal effect of intervention X on outcome Y?"
            )
            return prompt, "ABSTAIN", False

        elif stage == "phase10_stage_belief_revision":
            prompt = (
                "Initial evidence: Drug D reduces tumour size in 70% of cases (n=20).\n\n"
                "New evidence: A larger trial (n=200) shows Drug D reduces tumour size in 45% of cases.\n\n"
                "Further evidence: Meta-analysis of 10 trials (n=2000) shows 48% response rate.\n\n"
                "Question: What is your revised estimate of Drug D's efficacy and what does "
                "this revision tell us about the initial study?"
            )
            return (
                prompt,
                "~48% efficacy. Initial study likely overestimated due to small n and "
                "possible publication bias.",
                True
            )

        else:
            problem = random.choice(self.SCIENTIFIC_PROBLEMS)
            return (problem["prompt"], problem["ground_truth"], problem["knowable"])


class DeliberationEngine:
    def __init__(self, model, max_steps: int = 20):
        self._model = model
        self._max_steps = max_steps

    def run(self, prompt: str) -> ModelOutput:
        state: List[ReasoningStep] = []
        for step_id in range(self._max_steps):
            step = self._generate_step(prompt, state, step_id)
            state.append(step)
            if self._should_halt(state):
                logger.debug(f"Halting at step {step_id}: confidence={step.confidence:.2f}")
                break
        return self._finalize(prompt, state)

    def _generate_step(self, prompt: str, state: List[ReasoningStep], step_id: int) -> ReasoningStep:
        try:
            return self._model.generate_step(prompt, state, step_id)
        except (AttributeError, NotImplementedError):
            step_types = ["observation", "inference", "intervention", "counterfactual"]
            return ReasoningStep(
                step_id=step_id,
                type=step_types[step_id % len(step_types)],
                content=f"Reasoning step {step_id}: analysing {prompt[:50]}",
                confidence=min(0.5 + step_id * 0.08, 0.95),
                dependencies=list(range(step_id)),
                evidence_for=[],
                evidence_against=[],
                contradiction_score=0.0,
            )

    def _should_halt(self, state: List[ReasoningStep]) -> bool:
        if not state:
            return False
        last = state[-1]
        return (
            last.confidence > 0.90
            or len(state) >= self._max_steps
            or getattr(last, "abstain_signal", False)
        )

    def _finalize(self, prompt: str, state: List[ReasoningStep]) -> ModelOutput:
        if not state:
            return ModelOutput(
                decision="abstain",
                final_answer=None,
                reasoning_chain=[],
                confidence=0.0,
                uncertainty={"aleatoric": 1.0, "epistemic": 1.0, "model": 1.0},
                missing_information=["No reasoning steps generated."],
            )
        last = state[-1]
        avg_confidence = float(np.mean([s.confidence for s in state]))
        total_contradiction = sum(s.contradiction_score for s in state)
        if avg_confidence < 0.40 or total_contradiction > 0.60:
            decision = "abstain"
            final_answer = None
        else:
            decision = "answer"
            try:
                final_answer = self._model.generate_final(state, prompt)
            except (AttributeError, NotImplementedError):
                final_answer = last.content
        return ModelOutput(
            decision=decision,
            final_answer=final_answer,
            reasoning_chain=state,
            confidence=avg_confidence,
            uncertainty={
                "aleatoric": round(1.0 - avg_confidence, 3),
                "epistemic": round(total_contradiction, 3),
                "model": round(max(0, 0.5 - avg_confidence * 0.5), 3),
            },
            missing_information=[],
        )


class EvaluationEngine:
    @staticmethod
    def evaluate_prediction(prediction: Optional[str], ground_truth: str) -> float:
        if prediction is None:
            return 0.0
        pred_clean = prediction.strip().lower()
        gt_clean = ground_truth.strip().lower()
        if pred_clean == gt_clean:
            return 1.0
        pred_words = set(pred_clean.split())
        gt_words = set(gt_clean.split())
        if not gt_words:
            return 0.0
        overlap = len(pred_words & gt_words) / len(gt_words)
        return round(min(overlap, 0.8), 3)

    @staticmethod
    def brier_score(prob: float, outcome: float) -> float:
        return round((prob - outcome) ** 2, 4)

    @staticmethod
    def abstention_score(decision: str, correct: float, knowable: bool) -> float:
        if decision == "abstain":
            return +0.6 if not knowable else -0.5
        else:
            return +1.0 if correct > 0.5 else -1.5

    @staticmethod
    def consistency_score(reasoning_chain: List[ReasoningStep]) -> float:
        if not reasoning_chain:
            return 0.0
        total_contradiction = sum(s.contradiction_score for s in reasoning_chain)
        return round(max(0.0, 1.0 - total_contradiction), 4)

    @staticmethod
    def inject_contradiction_scores(reasoning_chain: List[ReasoningStep]) -> List[ReasoningStep]:
        for i, step in enumerate(reasoning_chain):
            score = 0.0
            prior_evidence_for = []
            for prior in reasoning_chain[:i]:
                prior_evidence_for.extend(prior.evidence_for)
            if step.evidence_against and prior_evidence_for:
                contradicted = [e for e in step.evidence_against if e in prior_evidence_for]
                if contradicted:
                    score = min(len(contradicted) / max(len(prior_evidence_for), 1), 1.0)
            step.contradiction_score = round(score, 3)
        return reasoning_chain


class RewardEngine:
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        phase10 = get_phase(10)
        default_weights = (
            phase10.metadata.get("reward_weights", {})
            if phase10 else {}
        )
        self._weights = weights or {
            "correctness": default_weights.get("correctness", 0.30),
            "consistency": default_weights.get("consistency", 0.25),
            "calibration": default_weights.get("calibration", 0.25),
            "abstention":  default_weights.get("abstention",  0.20),
        }
        self._eval = EvaluationEngine()
        logger.info(f"RewardEngine initialised: weights={self._weights}")

    def compute_reward(
        self,
        output: ModelOutput,
        ground_truth: str,
        knowable: bool,
    ) -> Tuple[float, Dict[str, float]]:
        correctness = self._eval.evaluate_prediction(output.final_answer, ground_truth)
        calibration = -self._eval.brier_score(output.confidence, correctness)
        abstention = self._eval.abstention_score(output.decision, correctness, knowable)
        consistency = self._eval.consistency_score(
            self._eval.inject_contradiction_scores(output.reasoning_chain)
        )
        brevity_penalty = 0.01 * len(output.reasoning_chain)
        total = (
            self._weights["correctness"]  * correctness
            + self._weights["consistency"] * consistency
            + self._weights["calibration"] * calibration
            + self._weights["abstention"]  * abstention
            - brevity_penalty
        )
        total = round(total, 4)
        components = {
            "correctness": round(correctness, 4),
            "consistency": round(consistency, 4),
            "calibration": round(calibration, 4),
            "abstention":  round(abstention, 4),
            "brevity_penalty": round(brevity_penalty, 4),
            "total": total,
        }
        log_event("core.phase10_training", f"Reward computed: {components}", level="debug")
        return total, components


class Phase10Trainer:
    def __init__(
        self,
        model,
        tokenizer,
        reward_engine: Optional[RewardEngine] = None,
        ppo_config: Optional[Dict] = None,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._reward_engine = reward_engine or RewardEngine()
        self._scenario_gen = ScenarioGenerator()
        self._deliberation = DeliberationEngine(model)
        self._ppo_config = ppo_config or {}
        self._current_stage_index = 0
        self._stage_metrics: Dict[str, List[Dict]] = {stage: [] for stage in PHASE10_STAGES}
        self._ppo_trainer = None
        logger.info("Phase10Trainer initialised.")

    def setup_ppo(self):
        try:
            from trl import PPOTrainer, PPOConfig
            from trl import AutoModelForCausalLMWithValueHead
            config = PPOConfig(
                model_name=getattr(self._model.config, "_name_or_path", "logiik_model"),
                learning_rate=self._ppo_config.get("learning_rate", 1.41e-5),
                batch_size=self._ppo_config.get("batch_size", 16),
                mini_batch_size=self._ppo_config.get("mini_batch_size", 4),
                gradient_accumulation_steps=self._ppo_config.get("gradient_accumulation_steps", 4),
                optimize_cuda_cache=True,
                early_stopping=False,
                target_kl=self._ppo_config.get("target_kl", 0.1),
                ppo_epochs=self._ppo_config.get("ppo_epochs", 4),
                seed=42,
                init_kl_coef=0.2,
                adap_kl_ctrl=True,
            )
            self._ppo_trainer = PPOTrainer(
                config=config,
                model=self._model,
                ref_model=None,
                tokenizer=self._tokenizer,
            )
            logger.info("PPOTrainer initialised successfully.")
        except ImportError:
            logger.warning("trl not installed. PPO training unavailable. Run: pip install trl>=0.8.6")
            self._ppo_trainer = None
        except Exception as e:
            logger.error(f"PPOTrainer setup failed: {e}")
            self._ppo_trainer = None

    def train_stage(self, stage: str, n_scenarios: int = 50) -> Dict:
        if stage not in PHASE10_STAGES:
            raise ValueError(f"Unknown stage: {stage}. Valid: {PHASE10_STAGES}")
        logger.info(f"Phase 10 training: stage={stage}, n_scenarios={n_scenarios}")
        scenarios = self._scenario_gen.generate(stage, count=n_scenarios)
        stage_rewards = []
        stage_components = []
        for prompt, ground_truth, knowable in scenarios:
            output = self._deliberation.run(prompt)
            reward, components = self._reward_engine.compute_reward(output, ground_truth, knowable)
            stage_rewards.append(reward)
            stage_components.append(components)
            if self._ppo_trainer is not None:
                self._ppo_step(prompt, output, reward)
        metrics = self._aggregate_metrics(stage, stage_rewards, stage_components)
        self._stage_metrics[stage].append(metrics)
        log_event(
            "core.phase10_training",
            f"Stage complete: {stage} | avg_reward={metrics['avg_reward']:.4f} | "
            f"correctness={metrics['correctness_rate']:.3f}",
            level="info"
        )
        return metrics

    def train_all_stages(self, n_scenarios_per_stage: int = 50) -> Dict[str, Dict]:
        all_metrics = {}
        for stage in PHASE10_STAGES:
            metrics = self.train_stage(stage, n_scenarios_per_stage)
            all_metrics[stage] = metrics
            logger.info(f"Completed {stage}: avg_reward={metrics['avg_reward']:.4f}")
        logger.info("All Phase 10 stages complete.")
        return all_metrics

    def get_metrics(self) -> Dict:
        return self._stage_metrics

    def _ppo_step(self, prompt: str, output: ModelOutput, reward: float):
        if self._ppo_trainer is None:
            return
        try:
            query_tensors = self._tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512,
            )["input_ids"]
            response_text = (
                output.final_answer
                if output.final_answer
                else "I cannot determine the answer from the given information."
            )
            response_tensors = self._tokenizer(
                response_text, return_tensors="pt", truncation=True, max_length=256,
            )["input_ids"]
            reward_tensor = torch.tensor([reward], dtype=torch.float32)
            self._ppo_trainer.step(
                [query_tensors.squeeze(0)],
                [response_tensors.squeeze(0)],
                [reward_tensor],
            )
        except Exception as e:
            logger.warning(f"PPO step failed: {e}")

    def _aggregate_metrics(self, stage: str, rewards: List[float], components: List[Dict]) -> Dict:
        if not rewards:
            return {"stage": stage, "avg_reward": 0.0}
        abstentions = sum(1 for c in components if c.get("abstention", 0) > 0)
        return {
            "stage": stage,
            "n_scenarios": len(rewards),
            "avg_reward": round(float(np.mean(rewards)), 4),
            "min_reward": round(float(np.min(rewards)), 4),
            "max_reward": round(float(np.max(rewards)), 4),
            "correctness_rate": round(float(np.mean([c["correctness"] for c in components])), 4),
            "consistency_avg": round(float(np.mean([c["consistency"] for c in components])), 4),
            "calibration_avg": round(float(np.mean([c["calibration"] for c in components])), 4),
            "abstention_rate": round(abstentions / len(rewards), 4),
        }
