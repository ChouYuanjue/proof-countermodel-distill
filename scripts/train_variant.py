#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from pocd.train import TrainConfig, train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--variant",
        choices=["answer_only", "proof_only", "proco", "proco_chain", "proco_witness", "proco_no_refute"],
        required=True,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-config-name", default="depth-3ext-NatLang")
    parser.add_argument("--eval-config-name", default="depth-3ext-NatLang")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="dev")
    parser.add_argument("--train-max-examples", type=int)
    parser.add_argument("--eval-max-examples", type=int, default=2048)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--study-tag", default="default")
    parser.add_argument("--model-tag", default="base")
    parser.add_argument("--notes")
    args = parser.parse_args()

    config = TrainConfig(
        model_name=args.model_name,
        variant=args.variant,
        output_dir=args.output_dir,
        train_config_name=args.train_config_name,
        eval_config_name=args.eval_config_name,
        train_split=args.train_split,
        eval_split=args.eval_split,
        train_max_examples=args.train_max_examples,
        eval_max_examples=args.eval_max_examples,
        max_length=args.max_length,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        seed=args.seed,
        load_in_4bit=not args.no_4bit,
        study_tag=args.study_tag,
        model_tag=args.model_tag,
        notes=args.notes,
    )
    metadata = train_model(config)
    print(metadata)


if __name__ == "__main__":
    main()
