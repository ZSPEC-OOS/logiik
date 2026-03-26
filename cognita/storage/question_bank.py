"""
Question Bank — persistent deduplication system for training questions.

Flow:
  NEW QUESTION GENERATED
       ↓
  [Check Question Bank] → EXISTS? → YES → Log to Toss Log → DISCARD
       ↓
       NO
       ↓
  Transmit to NERO → Log to Question Bank

When toss_count >= repeat_threshold, training is marked complete.
"""
import json
import re
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace for comparison."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


class QuestionBank:
    """
    Persistent log of all unique questions seen during training.
    Provides real-time duplicate detection and a Toss Log.
    Saves to JSON files under knowledge_base_path/logs/.
    """

    def __init__(self, base_path: Path):
        self.logs_path = Path(base_path) / "logs"
        self.logs_path.mkdir(parents=True, exist_ok=True)

        self._bank_file = self.logs_path / "question_bank.json"
        self._toss_file = self.logs_path / "toss_log.json"

        # normalized_text → entry dict
        self._bank: Dict[str, dict] = {}
        # list of toss entries
        self._toss: List[dict] = []

        self._load()

    # ── Persistence ──────────────────────────────────────────────

    def _load(self):
        if self._bank_file.exists():
            with open(self._bank_file) as f:
                for entry in json.load(f):
                    self._bank[entry["normalized"]] = entry
        if self._toss_file.exists():
            with open(self._toss_file) as f:
                self._toss = json.load(f)

    def _save(self):
        with open(self._bank_file, "w") as f:
            json.dump(list(self._bank.values()), f, indent=2)
        with open(self._toss_file, "w") as f:
            json.dump(self._toss, f, indent=2)

    # ── Core check ───────────────────────────────────────────────

    def check_and_log(self, question: str, topic: str = "") -> bool:
        """
        Check whether this question has been seen before.

        Returns True  → question is new; added to Question Bank.
        Returns False → duplicate; added to Toss Log, should be discarded.
        """
        norm = _normalize(question)

        if norm in self._bank:
            original = self._bank[norm]
            self._toss.append({
                "question":          question,
                "timestamp":         time.time(),
                "matched_id":        original["id"],
                "matched_question":  original["question"],
            })
            self._save()
            return False

        self._bank[norm] = {
            "id":         str(uuid.uuid4())[:8],
            "question":   question,
            "normalized": norm,
            "topic":      topic,
            "timestamp":  time.time(),
        }
        self._save()
        return True

    # ── Properties ───────────────────────────────────────────────

    @property
    def bank_count(self) -> int:
        return len(self._bank)

    @property
    def toss_count(self) -> int:
        return len(self._toss)

    # ── Accessors ────────────────────────────────────────────────

    def get_bank(self) -> List[dict]:
        """Return Question Bank entries (normalized key excluded)."""
        return [
            {k: v for k, v in e.items() if k != "normalized"}
            for e in self._bank.values()
        ]

    def get_toss_log(self) -> List[dict]:
        return list(self._toss)

    def generate_report(self, topics_description: str) -> dict:
        """Build final training completion report."""
        domains: Dict[str, int] = {}
        for entry in self._bank.values():
            d = entry.get("topic", "unknown")
            domains[d] = domains.get(d, 0) + 1

        return {
            "status":            "complete",
            "questions_asked":   self.bank_count,
            "questions_tossed":  self.toss_count,
            "topics_description": topics_description,
            "topics_covered":    domains,
            "generated_at":      time.time(),
        }

    def reset(self):
        """Clear bank and toss log for a fresh training session."""
        self._bank = {}
        self._toss = []
        self._save()
