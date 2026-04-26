#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
TRAIN_RUNTIME_RE = re.compile(r"'train_runtime':\s*([0-9.]+)")


def _artifact_name(path: Path) -> str:
    return path.name


def _infer_from_name(path: Path) -> tuple[str | None, str | None]:
    name = path.name
    if name.startswith("maintrack_"):
        return "maintrack", "qwen7b"
    if name.startswith("ablation_"):
        return "ablation", "qwen7b"
    if name.startswith("smoke_"):
        return "smoke", "qwen7b"
    return None, None


def _preference(path: Path) -> tuple[int, str]:
    name = path.parent.name
    if name.startswith("maintrack_"):
        return (0, name)
    if name.startswith("ablation_"):
        return (1, name)
    if name.startswith("smoke_"):
        return (2, name)
    return (3, name)


def _find_runtime(metadata: dict, artifact_dir: Path) -> float | None:
    train_metrics = metadata.get("train_metrics") or {}
    runtime = train_metrics.get("train_runtime")
    if runtime is not None:
        return float(runtime)

    for log_dir in [ROOT / "logs" / "maintrack"]:
        log_path = log_dir / f"{_artifact_name(artifact_dir)}_train.log"
        if log_path.exists():
            match = TRAIN_RUNTIME_RE.search(log_path.read_text())
            if match:
                return float(match.group(1))
    return None


def collect_rows(artifacts_dir: Path) -> list[dict]:
    rows: list[dict] = []
    metadata_paths = sorted(
        artifacts_dir.glob("*/train_metadata.json"),
        key=_preference,
    )
    seen_realpaths: set[Path] = set()
    for metadata_path in metadata_paths:
        artifact_dir = metadata_path.parent
        realpath = artifact_dir.resolve()
        if realpath in seen_realpaths:
            continue
        seen_realpaths.add(realpath)
        metadata = json.loads(metadata_path.read_text())
        config = metadata["config"]
        runtime = _find_runtime(metadata, artifact_dir)
        inferred_study, inferred_model = _infer_from_name(artifact_dir)
        rows.append(
            {
                "artifact_dir": str(artifact_dir),
                "study_tag": config.get("study_tag") or inferred_study,
                "model_tag": config.get("model_tag") or inferred_model,
                "variant": config.get("variant"),
                "seed": config.get("seed"),
                "train_config_name": config.get("train_config_name"),
                "train_examples": metadata.get("train_records"),
                "train_steps_estimate": metadata.get("train_steps_estimate"),
                "effective_batch_size": metadata.get("effective_batch_size"),
                "train_runtime_seconds": runtime,
                "train_runtime_hours": None if runtime is None else runtime / 3600.0,
                "train_samples_per_second": (metadata.get("train_metrics") or {}).get("train_samples_per_second"),
                "train_steps_per_second": (metadata.get("train_metrics") or {}).get("train_steps_per_second"),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "artifact_dir",
                "study_tag",
                "model_tag",
                "variant",
                "seed",
                "train_config_name",
                "train_examples",
                "train_steps_estimate",
                "effective_batch_size",
                "train_runtime_seconds",
                "train_runtime_hours",
                "train_samples_per_second",
                "train_steps_per_second",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--output-csv", default="results/compute_summary.csv")
    args = parser.parse_args()

    rows = collect_rows(ROOT / args.artifacts_dir)
    write_csv(ROOT / args.output_csv, rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
