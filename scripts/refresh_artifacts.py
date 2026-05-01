#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def compile_paper() -> None:
    paper_dir = ROOT / "paper"
    commands = [
        ["pdflatex", "-interaction=nonstopmode", "main.tex"],
        ["bibtex", "main"],
        ["pdflatex", "-interaction=nonstopmode", "main.tex"],
        ["pdflatex", "-interaction=nonstopmode", "main.tex"],
    ]
    for command in commands:
        print("$", " ".join(command), flush=True)
        subprocess.run(command, cwd=paper_dir, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-paper", action="store_true")
    args = parser.parse_args()

    run(["python", "scripts/summarize_results.py"])
    run(["python", "scripts/summarize_unknown_behavior.py"])
    run(["python", "scripts/summarize_abstention_evidence.py"])
    run(["python", "scripts/summarize_abstention_audit.py"])
    run(["python", "scripts/summarize_decoding_cost.py"])
    run(["python", "scripts/summarize_error_profile.py"])
    run(["python", "scripts/summarize_support_deletion.py"])
    run(["python", "scripts/plot_results.py"])
    run(["python", "scripts/export_latex_tables.py"])
    run(["python", "scripts/summarize_compute.py"])
    run(["python", "scripts/check_paper_claims.py"])
    if args.compile_paper:
        compile_paper()


if __name__ == "__main__":
    main()
