#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from pocd.dataset import build_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="depth-3ext-NatLang")
    parser.add_argument("--split", default="train")
    parser.add_argument("--variant", choices=["answer_only", "proof_only", "proco"], default="proco")
    parser.add_argument("--max-examples", type=int, default=3)
    args = parser.parse_args()

    records = build_records(
        config_name=args.config_name,
        split=args.split,
        variant=args.variant,
        max_examples=args.max_examples,
    )
    for idx, record in enumerate(records):
        print(f"\n=== Example {idx} ===")
        print(record["prompt"])
        print(record["target"])


if __name__ == "__main__":
    main()
