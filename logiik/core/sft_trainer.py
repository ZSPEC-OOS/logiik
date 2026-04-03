"""
Logiik SFT (Supervised Fine-Tuning) Trainer — Phases 1–11.

Loads exported JSONL files and fine-tunes a causal language model
on the banked Q&A pairs using HuggingFace TRL SFTTrainer.

Phase 12 uses PPO (see phase10_training.py). This trainer handles
all earlier phases.

Usage:
    python -m logiik.core.sft_trainer \
        --data_dir  ./training_data \
        --model_name  mistralai/Mistral-7B-v0.1 \
        --output_dir  ./checkpoints \
        --phases  1 2 3 4 5 6 7 8 9 10 11

Requirements:
    pip install trl transformers datasets accelerate bitsandbytes
    (bitsandbytes only needed for 4-bit quantisation on consumer GPUs)
"""
import argparse
import json
from pathlib import Path
from typing import List, Optional

from logiik.utils.logging import get_logger

logger = get_logger("sft_trainer")


# ─── Dataset helpers ──────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_phases(data_dir: str, phase_ids: List[int]) -> List[dict]:
    """Load JSONL records for the specified phases."""
    data_dir = Path(data_dir)
    all_records = []
    for pid in sorted(phase_ids):
        matches = sorted(data_dir.glob(f"phase_{pid:02d}_*.jsonl"))
        if not matches:
            logger.warning(f"No JSONL file found for phase {pid} in {data_dir}")
            continue
        records = load_jsonl(str(matches[0]))
        logger.info(f"Phase {pid:2d}: loaded {len(records):,} examples from {matches[0].name}")
        all_records.extend(records)
    logger.info(f"Total examples loaded: {len(all_records):,}")
    return all_records


def records_to_hf_dataset(records: List[dict]):
    """Convert JSONL records to a HuggingFace Dataset."""
    from datasets import Dataset

    # SFTTrainer expects a 'text' field with the full prompt+completion
    # formatted as a chat/instruction template.
    def _format(rec):
        return {
            "text": (
                f"### Question\n{rec['prompt']}\n\n"
                f"### Answer\n{rec['completion']}"
            ),
            "phase_id": rec.get("phase_id", 0),
            "domain": rec.get("domain", ""),
            "difficulty": rec.get("difficulty", 0.0),
        }

    formatted = [_format(r) for r in records]
    return Dataset.from_list(formatted)


# ─── Trainer ──────────────────────────────────────────────────────────────────

def run_sft(
    data_dir: str,
    model_name: str,
    output_dir: str,
    phase_ids: Optional[List[int]] = None,
    max_seq_length: int = 1024,
    num_train_epochs: int = 3,
    per_device_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    use_4bit: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    resume_from_checkpoint: Optional[str] = None,
):
    """
    Fine-tune a causal LM on Logiik curriculum data.

    Args:
        data_dir:          Directory containing phase_*.jsonl files
        model_name:        HuggingFace model ID or local path
        output_dir:        Where to save checkpoints and final model
        phase_ids:         Which phases to train on (default: 1–11)
        max_seq_length:    Max token length per example
        num_train_epochs:  Training epochs
        per_device_batch_size: Batch size per GPU
        gradient_accumulation_steps: Effective batch = batch * accum
        learning_rate:     AdamW learning rate
        use_4bit:          Load model in 4-bit (QLoRA) — saves VRAM
        lora_r:            LoRA rank
        lora_alpha:        LoRA alpha scaling
        resume_from_checkpoint: Path to resume from
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from trl import SFTTrainer, SFTConfig
    from peft import LoraConfig, get_peft_model

    if phase_ids is None:
        phase_ids = list(range(1, 12))  # Phases 1–11

    logger.info(f"Loading data for phases: {phase_ids}")
    records = load_phases(data_dir, phase_ids)
    if not records:
        raise ValueError(f"No training data found in {data_dir}")

    dataset = records_to_hf_dataset(records)
    logger.info(f"Dataset ready: {len(dataset)} examples")

    # ── Load tokenizer ────────────────────────────────────────────────────────
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info(f"Loading model: {model_name} (4bit={use_4bit})")
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    # ── LoRA config ───────────────────────────────────────────────────────────
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── Training args ─────────────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=25,
        save_strategy="epoch",
        save_total_limit=3,
        report_to="none",
        max_seq_length=max_seq_length,
        dataset_text_field="text",
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=training_args,
    )

    logger.info("Starting SFT training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    logger.info(f"Saving final model to {output_dir}/final")
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    logger.info("Training complete.")


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SFT fine-tuning on Logiik curriculum data (Phases 1–11)"
    )
    parser.add_argument("--data_dir", required=True,
                        help="Directory with phase_*.jsonl files")
    parser.add_argument("--model_name", required=True,
                        help="HuggingFace model ID, e.g. mistralai/Mistral-7B-v0.1")
    parser.add_argument("--output_dir", default="./checkpoints",
                        help="Output directory for checkpoints")
    parser.add_argument("--phases", nargs="+", type=int,
                        default=list(range(1, 12)),
                        help="Phase IDs to train on (default: 1-11)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--no_4bit", action="store_true",
                        help="Disable 4-bit quantisation (needs more VRAM)")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--resume", default=None,
                        help="Checkpoint path to resume from")
    args = parser.parse_args()

    run_sft(
        data_dir=args.data_dir,
        model_name=args.model_name,
        output_dir=args.output_dir,
        phase_ids=args.phases,
        max_seq_length=args.max_seq_length,
        num_train_epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        use_4bit=not args.no_4bit,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        resume_from_checkpoint=args.resume,
    )
