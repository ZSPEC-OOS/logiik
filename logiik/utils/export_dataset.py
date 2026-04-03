"""
Logiik Dataset Exporter.

Exports banked TrainingExample objects to JSONL files ready for
GPU training. One file per phase, plus a combined all-phases file.

Usage:
    python -m logiik.utils.export_dataset --output_dir ./training_data

Output structure:
    training_data/
        phase_01_memorization.jsonl
        phase_02_generation.jsonl
        ...
        phase_12_synthetic_judgment.jsonl
        all_phases.jsonl
        export_summary.json

JSONL format (one JSON object per line):
    {
        "prompt":       "<question text>",
        "completion":   "<correct answer text>",
        "phase_id":     1,
        "phase_name":   "memorization",
        "domain":       "scientific_language_corpus",
        "difficulty":   0.80
    }

This format is directly compatible with:
    - HuggingFace TRL SFTTrainer
    - OpenAI fine-tuning API
    - Axolotl
    - LLaMA-Factory
"""
import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional

from logiik.curriculum.phases import PHASES, get_phase
from logiik.utils.logging import get_logger

logger = get_logger("export_dataset")


def _example_to_record(example, phase_id: int, phase_name: str) -> Dict:
    """Convert a TrainingExample to a flat JSONL record."""
    # Build the completion: use the first correct answer
    correct_idx = example.correct_indices[0] if example.correct_indices else 0
    if example.answers and example.answers[correct_idx] != "[GENERATE]":
        completion = example.answers[correct_idx]
    else:
        # Generative example — completion is the explanation as a guide
        completion = example.explanation

    return {
        "prompt": example.question,
        "completion": completion,
        "phase_id": phase_id,
        "phase_name": phase_name,
        "domain": example.domain,
        "difficulty": round(example.difficulty, 4),
    }


def export_from_bank(
    bank: Dict[int, List],
    output_dir: str = "./training_data",
    min_examples_per_phase: int = 1,
) -> Dict:
    """
    Export a bank of TrainingExamples to JSONL files.

    Args:
        bank:          Dict mapping phase_id (1-12) → List[TrainingExample]
        output_dir:    Directory to write files into (created if missing)
        min_examples_per_phase: Skip phases with fewer examples than this

    Returns:
        Summary dict with counts per phase and total.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(out.resolve()),
        "phases": {},
        "total_examples": 0,
    }

    all_records = []

    for phase in PHASES:
        pid = phase.id
        examples = bank.get(pid, [])

        if len(examples) < min_examples_per_phase:
            logger.warning(
                f"Phase {pid} ({phase.name}): only {len(examples)} examples "
                f"— skipping (min={min_examples_per_phase})"
            )
            summary["phases"][pid] = {
                "name": phase.name, "count": 0, "skipped": True
            }
            continue

        records = [_example_to_record(ex, pid, phase.name) for ex in examples]
        all_records.extend(records)

        fname = out / f"phase_{pid:02d}_{phase.name}.jsonl"
        with open(fname, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        summary["phases"][pid] = {
            "name": phase.name,
            "count": len(records),
            "file": fname.name,
            "skipped": False,
        }
        summary["total_examples"] += len(records)
        logger.info(
            f"Phase {pid:2d} ({phase.name:30s}): {len(records):6,} examples → {fname.name}"
        )

    # Combined file
    all_path = out / "all_phases.jsonl"
    with open(all_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Summary JSON
    summary_path = out / "export_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 50)
    logger.info(f"Export complete: {summary['total_examples']:,} total examples")
    logger.info(f"Output: {out.resolve()}")
    logger.info("=" * 50)

    return summary


def export_from_text_store(
    output_dir: str = "./training_data",
) -> Dict:
    """
    Export by loading banked examples directly from TextStore (Firebase).
    Use this if you ran generation with Firebase enabled.
    """
    from logiik.storage.text_store import TextStore
    store = TextStore()

    bank: Dict[int, List] = {}
    for phase in PHASES:
        records = store.load_banked_examples(phase.id)
        if records:
            bank[phase.id] = records
            logger.info(f"Loaded {len(records)} examples from store for phase {phase.id}")

    return export_from_bank(bank, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export Logiik banked Q&A to JSONL training files"
    )
    parser.add_argument(
        "--output_dir", default="./training_data",
        help="Directory to write JSONL files (default: ./training_data)"
    )
    parser.add_argument(
        "--source", choices=["store", "demo"], default="store",
        help="store = load from TextStore/Firebase, demo = generate sample data"
    )
    parser.add_argument(
        "--min_examples", type=int, default=10,
        help="Skip phases with fewer than this many examples (default: 10)"
    )
    args = parser.parse_args()

    if args.source == "demo":
        # Demo mode: generate a small batch per phase to test export
        logger.info("Demo mode: generating sample examples for each phase...")
        from logiik.core.training import GenerativeCurriculum
        from logiik.curriculum.phases import PHASES

        gc = GenerativeCurriculum(
            topics_description="scientific concepts",
            topics_per_session=3,
        )
        bank = {}
        for phase in PHASES:
            gc.current_phase = phase.id - 1
            batch = gc.generate_phase_batch(batch_size=5)
            bank[phase.id] = batch.examples
            logger.info(f"Phase {phase.id}: generated {len(batch.examples)} demo examples")

        export_from_bank(bank, args.output_dir, min_examples_per_phase=1)
    else:
        export_from_text_store(args.output_dir)
