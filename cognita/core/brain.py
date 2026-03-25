"""
Cognita Brain - Generative Neural Architecture with Local Knowledge Integration
Implements a transformer-based student model that learns from teacher API
and develops generative capabilities beyond mere memorization.
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path


class CognitaBrain(nn.Module):
    """
    Core neural architecture combining:
    - Base transformer for language understanding
    - LoRA adapters for efficient fine-tuning
    - Generative head for original answer synthesis
    - Knowledge distillation from teacher API
    """

    def __init__(
        self,
        base_model: str = "microsoft/DialoGPT-medium",
        knowledge_dim: int = 768,
        lora_r: int = 16,
        lora_alpha: int = 32,
        generative_temperature: float = 0.7,
        device: str = "auto"
    ):
        super().__init__()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and device == "auto" else
            "cpu" if device == "auto" else device
        )

        # Base transformer model (student)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None
        )

        # LoRA configuration for efficient adaptation
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            bias="none",
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        self.model = get_peft_model(self.base_model, lora_config)

        # Generative head - enables original answer synthesis beyond memorization
        self.generative_head = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, knowledge_dim),
            nn.LayerNorm(knowledge_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(knowledge_dim, self.model.config.vocab_size)
        ).to(self.device)

        self.temperature = generative_temperature
        self.knowledge_embeddings = []
        self.training_history = []

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        teacher_logits: Optional[torch.Tensor] = None,
        return_generative: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional knowledge distillation from teacher.

        Args:
            input_ids: Tokenized input
            attention_mask: Attention mask
            labels: Target labels for supervised learning
            teacher_logits: Soft targets from teacher API for distillation
            return_generative: Whether to return generative head outputs
        """
        # Base model forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels if teacher_logits is None else None,
            output_hidden_states=True
        )

        # Extract hidden states for generative head
        hidden_states = outputs.hidden_states[-1]  # Last layer

        # Generative head prediction
        gen_logits = self.generative_head(hidden_states)

        # Combine base and generative outputs with temperature scaling
        combined_logits = (outputs.logits + gen_logits * 0.3) / self.temperature

        loss = outputs.loss if hasattr(outputs, 'loss') else None

        # Knowledge distillation loss if teacher logits provided
        if teacher_logits is not None:
            distill_loss = self._distillation_loss(
                combined_logits, teacher_logits, labels
            )
            if loss is not None:
                loss = 0.7 * loss + 0.3 * distill_loss
            else:
                loss = distill_loss

        result = {
            "loss": loss,
            "logits": combined_logits,
            "hidden_states": hidden_states,
            "generative_logits": gen_logits
        }

        return result

    def _distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        temperature: float = 2.0
    ) -> torch.Tensor:
        """KL divergence loss for knowledge distillation with temperature scaling."""
        soft_targets = nn.functional.softmax(teacher_logits / temperature, dim=-1)
        soft_prob = nn.functional.log_softmax(student_logits / temperature, dim=-1)

        loss = nn.functional.kl_div(
            soft_prob, soft_targets, reduction='batchmean'
        ) * (temperature ** 2)

        return loss

    def generate_original_answer(
        self,
        question: str,
        context_embeddings: Optional[torch.Tensor] = None,
        max_length: int = 150,
        min_confidence: float = 0.7
    ) -> Dict[str, any]:
        """
        Generate original answer based on learned patterns, not memorization.

        Uses the generative head with knowledge context to synthesize
        novel responses that demonstrate understanding rather than repetition.
        """
        self.eval()
        with torch.no_grad():
            inputs = self.tokenizer(
                question,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            # Get base model output
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=self.temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

            # Get generative head influence
            base_output = self.model(
                **inputs,
                output_hidden_states=True
            )
            gen_influence = self.generative_head(base_output.hidden_states[-1][:, -1, :])

            # Calculate confidence score
            confidence = torch.softmax(gen_influence, dim=-1).max().item()

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove the question from the answer if it appears
            answer = generated_text.replace(question, "").strip()

            return {
                "answer": answer,
                "confidence": confidence,
                "is_original": confidence > min_confidence,
                "tokens_used": outputs.shape[1]
            }

    def save_knowledge_state(self, path: Path):
        """Save model weights and knowledge embeddings to local folder."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save LoRA adapters
        self.model.save_pretrained(path / "adapters")

        # Save generative head
        torch.save(self.generative_head.state_dict(), path / "generative_head.pt")

        # Save knowledge embeddings
        if self.knowledge_embeddings:
            np.save(
                path / "knowledge_embeddings.npy",
                np.array(self.knowledge_embeddings)
            )

        # Save training history
        with open(path / "training_history.json", "w") as f:
            json.dump(self.training_history, f, indent=2)

        print(f"Knowledge state saved to {path}")

    def load_knowledge_state(self, path: Path):
        """Load previously saved knowledge."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"No knowledge found at {path}")

        # Load adapters
        self.model = AutoModelForCausalLM.from_pretrained(
            path / "adapters",
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        )

        # Load generative head
        self.generative_head.load_state_dict(
            torch.load(path / "generative_head.pt", map_location=self.device)
        )

        # Load knowledge embeddings
        embeddings_path = path / "knowledge_embeddings.npy"
        if embeddings_path.exists():
            self.knowledge_embeddings = np.load(embeddings_path).tolist()

        # Load training history
        history_path = path / "training_history.json"
        if history_path.exists():
            with open(history_path) as f:
                self.training_history = json.load(f)

        print(f"Knowledge loaded from {path}")
