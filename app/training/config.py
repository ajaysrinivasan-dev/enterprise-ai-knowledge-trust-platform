"""Validated training configuration."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    """Explicit LoRA/QLoRA settings with conservative defaults."""

    base_model: str = "sshleifer/tiny-gpt2"
    method: str = "lora"
    max_length: int = 128
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    epochs: float = 1.0
    learning_rate: float = 2e-4
    seed: int = 42
    output_dir: str = "artifacts/training"
    max_steps: int = 20
    min_free_vram_gb: float = 4.0

    def validate(self) -> None:
        """Reject unsafe or unsupported training settings before model loading."""

        if self.method not in {"lora", "qlora"}:
            raise ValueError("method must be 'lora' or 'qlora'")
        if self.max_length <= 0 or self.batch_size <= 0 or self.gradient_accumulation_steps <= 0:
            raise ValueError("sequence and batch settings must be positive")
        if self.epochs <= 0 or self.learning_rate <= 0 or self.max_steps <= 0:
            raise ValueError("training limits must be positive")
        if not self.base_model.strip():
            raise ValueError("base_model is required")
        if Path(self.output_dir).is_absolute():
            raise ValueError("output_dir must be relative to the approved artifacts directory")
