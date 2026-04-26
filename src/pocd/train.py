from __future__ import annotations

from dataclasses import asdict, dataclass
import inspect
import json
import math
from pathlib import Path

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from .dataset import build_records, tokenize_records


@dataclass
class TrainConfig:
    model_name: str
    variant: str
    train_config_name: str = "depth-3ext-NatLang"
    train_split: str = "train"
    eval_config_name: str = "depth-3ext-NatLang"
    eval_split: str = "dev"
    train_max_examples: int | None = None
    eval_max_examples: int | None = 2048
    output_dir: str = "artifacts/run"
    max_length: int = 1024
    learning_rate: float = 2e-4
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    seed: int = 0
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    load_in_4bit: bool = True
    trust_remote_code: bool = True
    study_tag: str = "default"
    model_tag: str = "base"
    notes: str | None = None


class CausalLMDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad = max_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * pad)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * pad)
            batch["labels"].append(feature["labels"] + [-100] * pad)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def _load_tokenizer(model_name: str, trust_remote_code: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _load_model(config: TrainConfig):
    quantization_config = None
    if config.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def train_model(config: TrainConfig) -> dict:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = _load_tokenizer(config.model_name, trust_remote_code=config.trust_remote_code)

    train_records = build_records(
        config_name=config.train_config_name,
        split=config.train_split,
        variant=config.variant,
        max_examples=config.train_max_examples,
        seed=config.seed,
    )
    eval_records = build_records(
        config_name=config.eval_config_name,
        split=config.eval_split,
        variant=config.variant,
        max_examples=config.eval_max_examples,
        seed=config.seed,
    )

    train_dataset = tokenize_records(train_records, tokenizer, max_length=config.max_length)
    eval_dataset = tokenize_records(eval_records, tokenizer, max_length=config.max_length)

    model = _load_model(config)
    collator = CausalLMDataCollator(tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        do_eval=len(eval_dataset) > 0,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        eval_strategy="steps" if len(eval_dataset) > 0 else "no",
        save_strategy="steps",
        save_total_limit=1,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to=[],
        remove_unused_columns=False,
        optim="paged_adamw_8bit" if config.load_in_4bit else "adamw_torch",
        seed=config.seed,
        dataloader_num_workers=0,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset if len(eval_dataset) > 0 else None,
        "data_collator": collator,
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    train_output = trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "config": asdict(config),
        "train_records": len(train_records),
        "eval_records": len(eval_records),
        "train_tokenized": len(train_dataset),
        "eval_tokenized": len(eval_dataset),
        "effective_batch_size": (
            config.per_device_train_batch_size * config.gradient_accumulation_steps
        ),
        "train_steps_estimate": max(
            1,
            math.ceil(
                len(train_dataset)
                / max(1, config.per_device_train_batch_size * config.gradient_accumulation_steps)
            ),
        ),
        "train_metrics": train_output.metrics,
    }
    (output_dir / "train_metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata
