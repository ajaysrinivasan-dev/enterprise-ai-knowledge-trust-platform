"""Opt-in Hugging Face/PEFT LoRA and QLoRA training workflow.

Imports are intentionally lazy: the default application and tests do not download
models or require CUDA.
"""

import json
import random
import shutil
from pathlib import Path
from typing import Any

from app.training.artifacts import AdapterMetadata, save_metadata
from app.training.config import TrainingConfig
from app.training.dataset import IntentSplits, validate_training_ready
from app.training.preflight import collect_preflight


def _prompt(text: str) -> str:
    return f"Classify the intent. Return one label only.\nInput: {text}\nLabel:"


def train_adapter(
    splits: IntentSplits,
    config: TrainingConfig,
    *,
    run_id: str,
    confirm: bool = False,
) -> AdapterMetadata:
    """Run bounded adapter training only after explicit confirmation."""

    config.validate()
    validate_training_ready(splits)
    if not confirm:
        raise ValueError("training requires explicit confirmation")
    report = collect_preflight(config.output_dir)
    if not report.optional_packages["transformers"] or not report.optional_packages["peft"]:
        raise RuntimeError("LoRA training requires installed transformers and peft")
    if config.method == "qlora" and (
        not report.cuda_available or not report.optional_packages["bitsandbytes"]
    ):
        raise RuntimeError("QLoRA requires CUDA and the bitsandbytes package")
    if config.method == "qlora" and report.free_vram_gb is not None:
        if report.free_vram_gb < config.min_free_vram_gb:
            raise RuntimeError("available GPU memory is below the configured QLoRA minimum")
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError("training dependencies are incomplete") from exc
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    output = Path(config.output_dir).resolve() / run_id
    output.mkdir(parents=True, exist_ok=False)
    try:
        quantization = None
        model_kwargs: dict[str, Any] = {}
        if config.method == "qlora":
            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            model_kwargs = {"quantization_config": quantization, "device_map": "auto"}
        tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is None:
                raise RuntimeError("selected tokenizer has neither a pad token nor an EOS token")
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(config.base_model, **model_kwargs)
        target_modules = _target_modules(model)
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora)
        encoded = _encode_supervised(tokenizer, splits.train, config.max_length)
        dataset = _SimpleDataset(encoded)
        args = TrainingArguments(
            output_dir=str(output),
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            num_train_epochs=config.epochs,
            learning_rate=config.learning_rate,
            max_steps=config.max_steps,
            seed=config.seed,
            save_strategy="no",
            report_to=[],
            remove_unused_columns=False,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dataset,
        )
        trainer.train()
        model.save_pretrained(output)
        tokenizer.save_pretrained(output / "tokenizer")
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    metadata = AdapterMetadata(
        run_id=run_id,
        base_model=config.base_model,
        method=config.method,
        dataset_fingerprint=splits.fingerprint,
        split_seed=splits.seed,
        status="completed",
        output_dir=str(output),
        tokenizer_path=str(output / "tokenizer"),
        adapter_path=str(output),
    )
    save_metadata(output, metadata)
    (output / "training_config.json").write_text(
        json.dumps(config.__dict__, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metadata


def _target_modules(model: Any) -> list[str]:
    """Select supported attention projections and reject unknown architectures."""

    names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    candidates = (
        ("c_attn",),
        ("q_proj", "k_proj", "v_proj", "o_proj"),
        ("query", "key", "value"),
    )
    for candidate in candidates:
        selected = [name for name in candidate if name in names]
        if selected:
            return selected
    raise RuntimeError(
        "selected causal language model has no supported LoRA target modules "
        "(expected c_attn or q/k/v/o projections)"
    )


def load_adapter_predictor(metadata: AdapterMetadata):
    """Load a completed adapter through PEFT without deserializing pickle objects."""

    try:
        from peft import AutoPeftModelForCausalLM, PeftConfig
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("adapter inference requires peft and transformers") from exc
    adapter_config = PeftConfig.from_pretrained(metadata.adapter_path, local_files_only=True)
    if adapter_config.base_model_name_or_path != metadata.base_model:
        raise RuntimeError("adapter base model does not match its recorded base model")
    tokenizer = AutoTokenizer.from_pretrained(metadata.tokenizer_path, local_files_only=True)
    if tokenizer.pad_token is None:
        raise RuntimeError("saved tokenizer has no padding token")
    model = AutoPeftModelForCausalLM.from_pretrained(
        metadata.adapter_path,
        is_trainable=False,
        local_files_only=True,
    )

    def predict(text: str) -> str:
        inputs = tokenizer(_prompt(text), return_tensors="pt")
        generated = model.generate(**inputs, max_new_tokens=8)
        output = tokenizer.decode(generated[0][inputs["input_ids"].shape[1] :])
        label = output.strip().split()[0] if output.strip() else ""
        return (
            label
            if label in {"question", "search", "sql_report", "email_draft", "verify"}
            else "unknown"
        )

    return predict


class _SimpleDataset:
    """Minimal torch dataset avoiding a hard runtime dependency on datasets."""

    def __init__(self, encoded: dict[str, list[list[int]]]) -> None:
        self.encoded = encoded

    def __len__(self) -> int:
        return len(self.encoded["input_ids"])

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return {key: values[index] for key, values in self.encoded.items()}


def _encode_supervised(tokenizer, rows: tuple[dict[str, str], ...], max_length: int) -> dict:
    """Mask prompt tokens so causal loss is calculated only on intent labels."""

    input_ids: list[list[int]] = []
    attention_mask: list[list[int]] = []
    labels: list[list[int]] = []
    for row in rows:
        prompt = tokenizer(_prompt(row["text"]), add_special_tokens=True)
        full = tokenizer(
            _prompt(row["text"]) + " " + row["intent"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        prompt_length = min(len(prompt["input_ids"]), max_length)
        input_ids.append(full["input_ids"])
        attention_mask.append(full["attention_mask"])
        labels.append(
            [
                -100 if index < prompt_length else token
                for index, token in enumerate(full["input_ids"])
            ]
        )
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
