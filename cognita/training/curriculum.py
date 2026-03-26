"""
Curriculum Engine - Implements the Question + 5-10 Answers training structure
with progressive learning and generative capability development.
"""
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from cognita.core.teacher_interface import TrainingExample


@dataclass
class ProcessedExample:
    """Tokenized and processed training example."""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    teacher_logits: Optional[torch.Tensor]  # Soft targets from teacher
    weight: float  # Importance weight based on difficulty


class CurriculumDataset(Dataset):
    """
    Dataset implementing the Q+A training structure:
    - Question + 5-10 possible answers
    - Soft labels from teacher for knowledge distillation
    - Progressive difficulty weighting
    """

    def __init__(
        self,
        examples: List[TrainingExample],
        tokenizer,
        max_length: int = 512,
        generative_ratio: float = 0.3  # 30% examples for generative training
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.generative_ratio = generative_ratio

        # Separate examples for memorization vs generation
        split_point = int(len(examples) * (1 - generative_ratio))
        self.memorization_examples = examples[:split_point]
        self.generative_examples = examples[split_point:]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> ProcessedExample:
        example = self.examples[idx]

        # Build question prefix and full prompt separately so we can mask the prefix
        question_prefix = self._format_question_prefix(example)
        formatted_text = self._format_training_prompt(example)

        # Tokenize full sequence
        encoding = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Measure question prefix length (without padding/special tokens)
        prefix_ids = self.tokenizer(
            question_prefix,
            add_special_tokens=False,
            return_tensors="pt"
        )["input_ids"]
        prefix_len = prefix_ids.shape[1]

        # Create labels: mask padding AND question prefix tokens
        # Loss is computed only on answer tokens
        labels = encoding["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[:, :prefix_len] = -100

        # Generate soft teacher targets (simulated here, would come from teacher API)
        teacher_logits = self._simulate_teacher_logits(example)

        # Weight by difficulty
        weight = 1.0 + example.difficulty  # Harder examples get more weight

        return ProcessedExample(
            input_ids=encoding["input_ids"].squeeze(0),
            attention_mask=encoding["attention_mask"].squeeze(0),
            labels=labels.squeeze(0),
            teacher_logits=teacher_logits,
            weight=weight
        )

    def _format_question_prefix(self, example: TrainingExample) -> str:
        """Return only the question/context prefix — these tokens are masked from the loss."""
        prefix = f"Question: {example.question}\n\nPossible Answers:\n"
        for i, answer in enumerate(example.answers):
            prefix += f"{i + 1}. {answer}\n"
        prefix += "\nBest Answer: "
        return prefix

    def _format_training_prompt(self, example: TrainingExample) -> str:
        """
        Format: Question + enumerated answers + correct answer indicator.
        Loss is computed only on the answer text and explanation (not the prefix).
        """
        prompt = f"Question: {example.question}\n\nPossible Answers:\n"

        for i, answer in enumerate(example.answers):
            marker = " [CORRECT]" if i in example.correct_indices else ""
            prompt += f"{i + 1}. {answer}{marker}\n"

        prompt += f"\nBest Answer: {example.answers[example.correct_indices[0]]}"
        prompt += f"\nExplanation: {example.explanation}"

        return prompt

    def _simulate_teacher_logits(self, example: TrainingExample) -> torch.Tensor:
        """
        Create soft target distribution over vocabulary
        representing teacher's knowledge.
        """
        vocab_size = self.tokenizer.vocab_size
        logits = torch.zeros(vocab_size)

        # Boost probability for correct answer tokens
        correct_answer = example.answers[example.correct_indices[0]]
        tokens = self.tokenizer.encode(correct_answer)

        for token_id in tokens:
            if token_id < vocab_size:
                logits[token_id] = 2.0  # Higher logit for correct tokens

        return logits


class GenerativeCurriculum:
    """
    Advanced curriculum that transitions from memorization to generation:
    Phase 1: Learn from teacher's Q+A structure (memorization)
    Phase 2: Generate novel answers given questions (generation)
    Phase 3: Synthesize knowledge across domains (abstraction)
    """

    def __init__(
        self,
        teacher_orchestrator,
        tokenizer,
        phase_ratios: Tuple[float, float, float] = (0.4, 0.4, 0.2)
    ):
        self.teacher = teacher_orchestrator
        self.tokenizer = tokenizer
        self.phase_ratios = phase_ratios
        self.current_phase = 0
        self.phase_names = ["Memorization", "Generation", "Abstraction"]

    def generate_phase_batch(self, batch_size: int) -> CurriculumDataset:
        """Generate training batch appropriate for current learning phase."""
        if self.current_phase == 0:
            # Phase 1: Standard Q+A with clear correct answers
            topics = self._get_foundational_topics()
            examples = self.teacher.generate_curriculum_batch(
                topics,
                examples_per_topic=batch_size // len(topics),
                difficulty_progression=True
            )

        elif self.current_phase == 1:
            # Phase 2: Open-ended questions requiring generation
            topics = self._get_intermediate_topics()
            examples = self._generate_generative_examples(topics, batch_size)

        else:
            # Phase 3: Cross-domain synthesis
            examples = self._generate_abstraction_examples(batch_size)

        return CurriculumDataset(
            examples,
            self.tokenizer,
            generative_ratio=self.phase_ratios[self.current_phase]
        )

    def _generate_generative_examples(
        self, topics: List[str], count: int
    ) -> List[TrainingExample]:
        """
        Generate examples where student must create original answers,
        not just select from provided options.
        """
        examples = []

        for topic in topics:
            # Get base question
            base = self.teacher.teachers[0].generate_training_example(topic, num_answers=3)

            # Modify to require generation: provide question + context, not answers
            modified = TrainingExample(
                question=base.question + "\n(Provide an original answer based on your understanding)",
                answers=["[GENERATE]"],  # Signal for generative mode
                correct_indices=[0],
                difficulty=min(base.difficulty * 1.2, 1.0),  # Harder, capped at 1.0
                domain=base.domain,
                explanation="Generate original answer showing understanding"
            )
            examples.append(modified)

        return examples[:count]

    def advance_phase(self):
        """Move to next training phase."""
        if self.current_phase < 2:
            self.current_phase += 1
            print(
                f"Advanced to Phase {self.current_phase + 1}: "
                f"{self.phase_names[self.current_phase]}"
            )
            return True
        return False

    def _get_foundational_topics(self) -> List[str]:
        return ["basic_reasoning", "fact_recall", "pattern_matching", "classification"]

    def _get_intermediate_topics(self) -> List[str]:
        return ["creative_writing", "problem_solving", "analysis", "synthesis"]

    def _generate_abstraction_examples(self, count: int) -> List[TrainingExample]:
        """Generate cross-domain synthesis problems."""
        topics = self._get_foundational_topics() + self._get_intermediate_topics()
        examples = []

        for i in range(count):
            topic = topics[i % len(topics)]
            base = self.teacher.teachers[0].generate_training_example(
                topic, difficulty=0.8, num_answers=5
            )

            cross_domain = TrainingExample(
                question=(
                    f"Connecting {base.domain} with other fields: {base.question}\n"
                    "(Draw on knowledge from multiple domains)"
                ),
                answers=base.answers,
                correct_indices=base.correct_indices,
                difficulty=0.9,
                domain=f"cross_domain_{base.domain}",
                explanation=f"Cross-domain synthesis: {base.explanation}"
            )
            examples.append(cross_domain)

        return examples
