"""
Teacher AI Interface - Connects to Kimi K2.5 as the teacher model to provide
structured training data: questions + 5-10 possible answers
"""
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

import openai


@dataclass
class TrainingExample:
    """Structured training example from teacher."""
    question: str
    answers: List[str]       # 5-10 possible answers
    correct_indices: List[int]  # Which answers are correct
    difficulty: float         # 0.0 to 1.0
    domain: str               # Subject area
    explanation: str          # Why these answers are valid


class TeacherInterface(ABC):
    """Abstract base for teacher AI connections."""

    @abstractmethod
    def generate_training_example(
        self,
        topic: str,
        difficulty: float = 0.5,
        num_answers: int = 5
    ) -> TrainingExample:
        """Generate a question with multiple possible answers."""
        pass

    @abstractmethod
    def evaluate_answer(
        self,
        question: str,
        student_answer: str,
        reference_answers: List[str]
    ) -> Dict[str, float]:
        """Evaluate student's answer quality."""
        pass


class KimiK2Teacher(TeacherInterface):
    """Kimi K2.5 as teacher model (OpenAI-compatible API)."""

    def __init__(
        self,
        api_key: str,           # Your Kimi API key
        base_url: str,          # e.g. https://api.moonshot.cn/v1
        model: str,             # e.g. kimi-k2-5
    ):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate_training_example(
        self,
        topic: str,
        difficulty: float = 0.5,
        num_answers: int = 5
    ) -> TrainingExample:
        """Generate structured training example with question + multiple answers."""
        prompt = f"""Generate a training example for an AI student learning about: {topic}

Difficulty level: {difficulty * 100:.0f}%

Provide:
1. A clear, specific question
2. {num_answers} possible answers (mix of correct and plausible incorrect ones)
3. Indicate which answers are correct (can be multiple)
4. Provide explanation for learning

Format as JSON:
{{
    "question": "...",
    "answers": ["...", "...", ...],
    "correct_indices": [0, 2],
    "explanation": "...",
    "domain": "..."
}}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a teacher AI creating training data for a student AI. Be precise and educational."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=1,
            response_format={"type": "json_object"}
        )

        data = json.loads(response.choices[0].message.content)

        return TrainingExample(
            question=data["question"],
            answers=data["answers"],
            correct_indices=data["correct_indices"],
            difficulty=difficulty,
            domain=data.get("domain", topic),
            explanation=data["explanation"]
        )

    def evaluate_answer(
        self,
        question: str,
        student_answer: str,
        reference_answers: List[str]
    ) -> Dict[str, float]:
        """Evaluate student answer against references."""
        prompt = f"""Evaluate this student answer:

Question: {question}
Reference Answers: {json.dumps(reference_answers)}
Student Answer: {student_answer}

Rate on scale 0-1 for:
- accuracy (factual correctness)
- completeness (covers key points)
- originality (not just copying)
- coherence (well structured)

Return JSON with scores and brief feedback."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)


class TeacherOrchestrator:
    """Manages teacher and curriculum progression."""

    def __init__(self, primary_teacher: TeacherInterface):
        self.teachers = [primary_teacher]
        self.curriculum_progress = {}
        self.generated_examples = []

    def generate_curriculum_batch(
        self,
        topics: List[str],
        examples_per_topic: int = 10,
        difficulty_progression: bool = True
    ) -> List[TrainingExample]:
        """Generate a structured curriculum with progressive difficulty."""
        batch = []

        for topic in topics:
            for i in range(examples_per_topic):
                difficulty = (i / examples_per_topic) if difficulty_progression else 0.5
                example = self.teachers[0].generate_training_example(
                    topic=topic,
                    difficulty=difficulty,
                    num_answers=min(5 + int(difficulty * 5), 10)  # 5-10 answers based on difficulty
                )
                batch.append(example)
                self.generated_examples.append(example)

        return batch

    def get_learning_statistics(self) -> Dict:
        """Analyze generated training data."""
        if not self.generated_examples:
            return {}

        domains = {}
        difficulties = []

        for ex in self.generated_examples:
            domains[ex.domain] = domains.get(ex.domain, 0) + 1
            difficulties.append(ex.difficulty)

        return {
            "total_examples": len(self.generated_examples),
            "domains": domains,
            "avg_difficulty": sum(difficulties) / len(difficulties),
            "difficulty_distribution": {
                "easy": sum(1 for d in difficulties if d < 0.33),
                "medium": sum(1 for d in difficulties if 0.33 <= d < 0.66),
                "hard": sum(1 for d in difficulties if d >= 0.66)
            }
        }
