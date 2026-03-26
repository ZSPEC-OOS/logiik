"""
Curriculum Engine - Implements the Question + 5-10 Answers training structure
with progressive learning and generative capability development.
"""
import math
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass


def collate_examples(batch):
    """Custom collate for ProcessedExample — stacks fixed-size tensors,
    drops variable-length teacher_logits to avoid shape mismatch errors."""
    return ProcessedExample(
        input_ids=torch.stack([b.input_ids for b in batch]),
        attention_mask=torch.stack([b.attention_mask for b in batch]),
        labels=torch.stack([b.labels for b in batch]),
        teacher_logits=None,  # variable length per example — skip stacking
        weight=sum(b.weight for b in batch) / len(batch),
    )

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
        generative_ratio: float = 0.3,
        val_ratio: float = 0.1,        # fraction of (q, ai) pairs held out for validation
        _items: Optional[List[Tuple["TrainingExample", int]]] = None,  # internal use
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.generative_ratio = generative_ratio

        if _items is not None:
            # Used by val_dataset — receives a pre-split item list directly
            self.items = _items
            self.examples = examples
            self._val_dataset = None
            return

        # Expand each TrainingExample into one item per answer so the model
        # learns P(a_i | q) independently for every answer in the set.
        all_items: List[Tuple[TrainingExample, int]] = []
        for ex in examples:
            for ans_idx in range(len(ex.answers)):
                all_items.append((ex, ans_idx))

        # Deterministic train/val split (last val_ratio fraction held out)
        split = int(len(all_items) * (1 - val_ratio))
        self.items = all_items[:split]
        val_items = all_items[split:]

        # Expose validation set as a separate Dataset instance
        self._val_dataset: Optional["CurriculumDataset"] = CurriculumDataset(
            examples, tokenizer, max_length, generative_ratio, _items=val_items
        )

        # Retain original examples list for phase splitting
        self.examples = examples
        split_point = int(len(examples) * (1 - generative_ratio))
        self.memorization_examples = examples[:split_point]
        self.generative_examples = examples[split_point:]

    @property
    def val_dataset(self) -> Optional["CurriculumDataset"]:
        return self._val_dataset

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx) -> ProcessedExample:
        example, ans_idx = self.items[idx]

        # Build question prefix and full prompt separately so we can mask the prefix
        question_prefix = self._format_question_prefix(example)
        formatted_text = self._format_training_prompt(example, ans_idx)

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

        # Generate soft teacher targets for this specific answer
        teacher_logits = self._simulate_teacher_logits(example, ans_idx)

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
        """Return only the question prefix — these tokens are masked from the loss."""
        return f"Question: {example.question}\nAnswer:"

    def _format_training_prompt(self, example: TrainingExample, ans_idx: int) -> str:
        """
        Format a single (q, a_i) pair.
        Loss is computed only on the answer tokens (after 'Answer:').
        """
        return f"Question: {example.question}\nAnswer: {example.answers[ans_idx]}"

    def _simulate_teacher_logits(self, example: TrainingExample, ans_idx: int) -> torch.Tensor:
        """
        Create a per-position soft target distribution over vocabulary for
        the answer tokens of this specific (q, a_i) pair.
        """
        vocab_size = self.tokenizer.vocab_size
        answer_tokens = self.tokenizer.encode(
            example.answers[ans_idx], add_special_tokens=False
        )
        seq_len = len(answer_tokens)
        logits = torch.zeros(seq_len, vocab_size)

        for pos, token_id in enumerate(answer_tokens):
            if token_id < vocab_size:
                logits[pos, token_id] = 2.0  # Peak at the correct next token

        return logits


class GenerativeCurriculum:
    """
    Advanced curriculum that transitions from memorization to generation:
    Phase 1: Learn from teacher's Q+A structure (memorization)
    Phase 2: Generate novel answers given questions (generation)
    Phase 3: Synthesize knowledge across domains (abstraction)
    Phase 4: Demonstrate complete coding understanding across common languages
    Phase 5: Build specialized scientific framework competency (Drosophila genetics focus)
    """

    def __init__(
        self,
        teacher_orchestrator,
        tokenizer,
        topics_description: str = "",
        phase_ratios: Tuple[float, float, float, float, float] = (0.25, 0.25, 0.2, 0.15, 0.15),
        phase_topics: Optional[Dict[str, List[str]]] = None,
        topics_per_session: int = 5,
    ):
        self.teacher = teacher_orchestrator
        self.tokenizer = tokenizer
        self.topics_description = topics_description
        self.phase_ratios = phase_ratios
        self.current_phase = 0
        self.phase_names = [
            "Memorization",
            "Generation",
            "Abstraction",
            "Coding Mastery",
            "Drosophila AI Framework",
        ]
        # Keyed by lowercase phase name; each value is an ordered topic list
        self.phase_topics: Dict[str, List[str]] = phase_topics or {}
        self.topics_per_session = topics_per_session
        # Per-phase cursor: tracks which topic to serve next in each phase
        self._topic_cursors: Dict[str, int] = {
            name.lower(): 0 for name in self.phase_names
        }

    def generate_phase_batch(
        self,
        batch_size: int,
        question_bank=None,   # optional QuestionBank for deduplication
    ) -> CurriculumDataset:
        """Generate training batch for current phase, with optional dedup."""
        if self.current_phase == 0:
            topics = self._get_topics()
            examples = self.teacher.generate_curriculum_batch(
                topics,
                examples_per_topic=max(1, batch_size // len(topics)),
                difficulty_progression=True
            )
        elif self.current_phase == 1:
            topics = self._get_topics()
            examples = self._generate_generative_examples(topics, batch_size)
        elif self.current_phase == 2:
            examples = self._generate_abstraction_examples(batch_size)
        elif self.current_phase == 3:
            examples = self._generate_coding_mastery_examples(batch_size)
        else:
            examples = self._generate_drosophila_framework_examples(batch_size)

        # Deduplicate via Question Bank when provided
        if question_bank is not None and examples:
            filtered = [
                ex for ex in examples
                if question_bank.check_and_log(ex.question, self.topics_description)
            ]
            # Always keep at least one example to avoid an empty dataset
            examples = filtered if filtered else examples[:1]

        return CurriculumDataset(
            examples,
            self.tokenizer,
            generative_ratio=self.phase_ratios[self.current_phase]
        )

    def _get_topics(self) -> List[str]:
        """
        Return the next slice of topics for the current phase, cycling through
        the phase_topics list.  Falls back to topics_description or a generic
        default when no structured topic list is configured.
        """
        phase_key = self.phase_names[self.current_phase].lower()
        topics = self.phase_topics.get(phase_key, [])

        if not topics:
            # Fallback: treat the free-form description as a single topic
            return [self.topics_description] if self.topics_description.strip() else ["general_knowledge"]

        n = len(topics)
        start = self._topic_cursors[phase_key]
        # Wrap-around slice so we always return topics_per_session items
        selected = [topics[(start + i) % n] for i in range(self.topics_per_session)]
        # Advance cursor for the next batch (wraps so all topics are eventually covered)
        self._topic_cursors[phase_key] = (start + self.topics_per_session) % n
        return selected

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
        if self.current_phase < len(self.phase_names) - 1:
            self.current_phase += 1
            print(
                f"Advanced to Phase {self.current_phase + 1}: "
                f"{self.phase_names[self.current_phase]}"
            )
            return True
        return False

    def _generate_abstraction_examples(self, count: int) -> List[TrainingExample]:
        """Generate cross-domain synthesis problems, cycling through abstraction topics."""
        topics = self._get_topics()
        examples = []

        for i in range(count):
            topic = topics[i % len(topics)]
            base = self.teacher.teachers[0].generate_training_example(
                topic, difficulty=0.8, num_answers=5
            )
            cross_domain = TrainingExample(
                question=(
                    f"Connecting ideas across fields: {base.question}\n"
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

    def _generate_coding_mastery_examples(self, count: int) -> List[TrainingExample]:
        """Generate advanced coding tasks spanning common programming languages."""
        topics = self._get_topics()
        examples = []

        for i in range(count):
            topic = topics[i % len(topics)]
            base = self.teacher.teachers[0].generate_training_example(
                topic, difficulty=0.9, num_answers=5
            )
            coding_mastery = TrainingExample(
                question=(
                    f"Coding mastery challenge: {base.question}\n"
                    "(Demonstrate robust implementation choices across common programming languages)"
                ),
                answers=base.answers,
                correct_indices=base.correct_indices,
                difficulty=0.95,
                domain=f"coding_mastery_{base.domain}",
                explanation=f"Coding mastery synthesis: {base.explanation}"
            )
            examples.append(coding_mastery)

        return examples

    def _generate_drosophila_framework_examples(self, count: int) -> List[TrainingExample]:
        """
        Generate expert-level interdisciplinary prompts for a specialized
        Drosophila melanogaster genetics AI framework, centered on axon
        guidance and neural wiring with first-principles integration.
        """
        topics = self._get_topics()
        examples = []

        for i in range(count):
            topic = topics[i % len(topics)]
            base = self.teacher.teachers[0].generate_training_example(
                topic, difficulty=0.95, num_answers=5
            )
            drosophila_framework = TrainingExample(
                question=(
                    f"Drosophila framework design challenge: {base.question}\n"
                    "(Trace genotype → molecular pathway → circuit wiring → behavior, "
                    "with chemistry, microbiology, and neuroscience integration)"
                ),
                answers=base.answers,
                correct_indices=base.correct_indices,
                difficulty=0.98,
                domain=f"drosophila_framework_{base.domain}",
                explanation=(
                    "Specialized AI framework reasoning for Drosophila genetics "
                    f"and axon guidance: {base.explanation}"
                ),
            )
            examples.append(drosophila_framework)

        return examples
