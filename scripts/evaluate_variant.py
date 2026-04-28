#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from pocd.eval import EvalConfig, evaluate_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--variant",
        choices=["answer_only", "proof_only", "proco", "proco_chain", "proco_witness", "proco_no_refute"],
        required=True,
    )
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--config-name", default="depth-3ext-NatLang")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--adapter-path")
    parser.add_argument("--train-metadata-path")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--study-tag", default="default")
    parser.add_argument("--model-tag", default="base")
    parser.add_argument("--notes")
    parser.add_argument("--progress-interval-batches", type=int, default=50)
    parser.add_argument("--support-deletion", action="store_true")
    parser.add_argument("--mutation-max-source-examples", type=int)
    parser.add_argument("--mutation-delete-kinds", default="fact,rule")
    args = parser.parse_args()

    config = EvalConfig(
        model_name=args.model_name,
        variant=args.variant,
        output_path=args.output_path,
        config_name=args.config_name,
        split=args.split,
        adapter_path=args.adapter_path,
        max_examples=args.max_examples,
        batch_size=args.batch_size,
        max_prompt_length=args.max_prompt_length,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        data_seed=args.data_seed,
        load_in_4bit=not args.no_4bit,
        study_tag=args.study_tag,
        model_tag=args.model_tag,
        train_metadata_path=args.train_metadata_path,
        notes=args.notes,
        progress_interval_batches=args.progress_interval_batches,
        support_deletion=args.support_deletion,
        mutation_max_source_examples=args.mutation_max_source_examples,
        mutation_delete_kinds=args.mutation_delete_kinds,
    )
    summary = evaluate_model(config)
    print(summary)


if __name__ == "__main__":
    main()
