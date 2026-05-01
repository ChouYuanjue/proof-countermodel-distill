#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


VARIANT_ORDER = {
    "answer_only": 0,
    "proof_only": 1,
    "proco_chain": 2,
    "proco_witness": 3,
    "proco_no_refute": 4,
    "proco": 5,
}
VARIANT_LABELS = {
    "answer_only": "answer-only",
    "proof_only": "proof-only",
    "proco_chain": "\\model{}-chain",
    "proco_witness": "\\model{}-witness",
    "proco_no_refute": "\\model{}-no-refute",
    "proco": "\\model{}",
}
EVAL_GROUP_ORDER = {
    "in_domain": 0,
    "depth_ood": 1,
    "template_transfer": 2,
    "domain_transfer": 3,
}
EVAL_GROUP_LABELS = {
    "in_domain": "In-domain",
    "depth_ood": "Depth-OOD",
    "template_transfer": "Template Transfer",
    "domain_transfer": "Domain Transfer",
}
METRIC_KEYS = [
    "accuracy",
    "macro_f1",
    "unknown_f1",
    "faithfulness_rate",
    "joint_accuracy",
    "valid_format_rate",
]


def _parse_int(value: str | int | float | None) -> int | None:
    if value in {None, "", "None"}:
        return None
    return int(float(value))


def _eval_scope_from_max_examples(max_examples: str | int | float | None) -> str:
    parsed = _parse_int(max_examples)
    return "full" if parsed is None else f"subset_{parsed}"


def _eval_scope_sort_key(scope: str) -> tuple[int, int]:
    if scope.startswith("subset_"):
        return (0, _parse_int(scope.split("_", 1)[1]) or 0)
    return (1, math.inf)


def _infer_eval_group(train_config_name: str | None, eval_config_name: str) -> str:
    if train_config_name and train_config_name == eval_config_name:
        return "in_domain"
    if eval_config_name == "depth-5":
        return "depth_ood"
    if eval_config_name == "birds-electricity":
        return "domain_transfer"
    return "template_transfer"


def _infer_legacy_metadata(path: Path, payload: dict) -> dict | None:
    stem = path.stem
    suffixes = ["_dev", "_test"]
    split = None
    for suffix in suffixes:
        if stem.endswith(suffix):
            split = suffix[1:]
            stem = stem[: -len(suffix)]
            break
    if split is None or "_" not in stem:
        return None

    study, remainder = stem.split("_", 1)
    variant = None
    tail = ""
    for candidate in ["answer_only", "proof_only", "proco_chain", "proco_witness", "proco_no_refute", "proco"]:
        if remainder == candidate:
            variant = candidate
            break
        prefix = f"{candidate}_"
        if remainder.startswith(prefix):
            variant = candidate
            tail = remainder[len(prefix) :]
            break
    if variant is None:
        return None

    config = payload.get("config", {})
    summary = payload["summary"]
    train_config_name = None
    eval_config_name = summary.get("config_name") or config.get("config_name") or "depth-3ext-NatLang"
    model_tag = tail or "base"
    eval_max_examples = _parse_int(config.get("max_examples"))

    return {
        "study": study,
        "model_tag": model_tag,
        "variant": variant,
        "seed": config.get("seed", 0),
        "train_config_name": train_config_name,
        "train_split": None,
        "train_examples": None,
        "eval_config_name": eval_config_name,
        "eval_split": summary.get("split") or config.get("split") or split,
        "eval_group": _infer_eval_group(train_config_name, eval_config_name),
        "eval_max_examples": eval_max_examples,
        "eval_scope": _eval_scope_from_max_examples(eval_max_examples),
    }


def _extract_metadata(path: Path, payload: dict) -> dict | None:
    summary = payload.get("summary")
    if summary is None:
        return None

    metadata = payload.get("metadata", {})
    train_metadata = metadata.get("train_metadata") or {}
    train_config = train_metadata.get("config") or {}
    config = payload.get("config", {})

    variant = summary.get("variant") or config.get("variant")
    if variant is None:
        legacy = _infer_legacy_metadata(path, payload)
        return legacy

    study = metadata.get("study_tag") or config.get("study_tag")
    model_tag = metadata.get("model_tag") or config.get("model_tag")
    if not study or not model_tag:
        legacy = _infer_legacy_metadata(path, payload) or {}
        study = study or legacy.get("study")
        model_tag = model_tag or legacy.get("model_tag")

    train_config_name = metadata.get("train_config_name") or train_config.get("train_config_name")
    train_split = metadata.get("train_split") or train_config.get("train_split")
    train_examples = metadata.get("train_records") or train_metadata.get("train_records")
    if train_examples is None:
        train_examples = train_config.get("train_max_examples")
    eval_config_name = summary.get("config_name") or config.get("config_name")
    eval_split = summary.get("split") or config.get("split") or "dev"
    eval_max_examples = metadata.get("eval_max_examples")
    if eval_max_examples is None:
        eval_max_examples = config.get("max_examples")
    eval_max_examples = _parse_int(eval_max_examples)
    eval_scope = metadata.get("eval_scope") or _eval_scope_from_max_examples(eval_max_examples)
    eval_group = summary.get("eval_group") or metadata.get("eval_group") or _infer_eval_group(
        train_config_name=train_config_name,
        eval_config_name=eval_config_name,
    )

    return {
        "study": study or "default",
        "model_tag": model_tag or "base",
        "variant": variant,
        "seed": config.get("seed", train_config.get("seed", 0)),
        "train_config_name": train_config_name,
        "train_split": train_split,
        "train_examples": train_examples,
        "eval_config_name": eval_config_name,
        "eval_split": eval_split,
        "eval_group": eval_group,
        "eval_max_examples": eval_max_examples,
        "eval_scope": eval_scope,
    }


def _format_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}"


def _format_mean_std(mean_value: float, std_value: float, runs: int) -> str:
    if runs <= 1:
        return _format_pct(mean_value)
    return f"{mean_value * 100:.1f} ± {std_value * 100:.1f}"


def _row_sort_key(row: dict) -> tuple:
    train_examples = row.get("train_examples")
    train_examples_key = math.inf if train_examples in {None, ""} else int(train_examples)
    return (
        row["study"],
        row["model_tag"],
        train_examples_key,
        EVAL_GROUP_ORDER.get(row["eval_group"], 99),
        row["eval_config_name"],
        row["eval_split"],
        _eval_scope_sort_key(row["eval_scope"]),
        VARIANT_ORDER.get(row["variant"], 99),
        int(row.get("seed", 0)),
    )


def load_rows(results_dir: Path) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    per_class_rows: list[dict] = []

    for path in sorted(results_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        if "summary" not in payload:
            continue

        metadata = _extract_metadata(path, payload)
        if metadata is None:
            continue

        summary = payload["summary"]
        row = {
            **metadata,
            "path": str(path),
            "examples": summary["examples"],
            "accuracy": summary["accuracy"],
            "macro_f1": summary["macro_f1"],
            "unknown_f1": summary["per_class"]["Unknown"]["f1"],
            "faithfulness_rate": summary["faithfulness_rate"],
            "joint_accuracy": summary["joint_accuracy"],
            "valid_format_rate": summary["valid_format_rate"],
        }
        rows.append(row)

        for label, metrics in summary["per_class"].items():
            per_class_rows.append(
                {
                    **metadata,
                    "label": label,
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                }
            )

    rows.sort(key=_row_sort_key)
    per_class_rows.sort(
        key=lambda row: (
            row["study"],
            row["model_tag"],
            math.inf if row.get("train_examples") is None else int(row["train_examples"]),
            EVAL_GROUP_ORDER.get(row["eval_group"], 99),
            row["eval_config_name"],
            row["eval_split"],
            _eval_scope_sort_key(row["eval_scope"]),
            VARIANT_ORDER.get(row["variant"], 99),
            row["label"],
            int(row.get("seed", 0)),
        )
    )
    return rows, per_class_rows


def aggregate_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["study"],
            row["model_tag"],
            row["variant"],
            row["train_config_name"],
            row["train_split"],
            row["train_examples"],
            row["eval_group"],
            row["eval_config_name"],
            row["eval_split"],
            row["eval_max_examples"],
            row["eval_scope"],
        )
        grouped[key].append(row)

    aggregated: list[dict] = []
    for key, bucket in grouped.items():
        (
            study,
            model_tag,
            variant,
            train_config_name,
            train_split,
            train_examples,
            eval_group,
            eval_config_name,
            eval_split,
            eval_max_examples,
            eval_scope,
        ) = key
        row = {
            "study": study,
            "model_tag": model_tag,
            "variant": variant,
            "train_config_name": train_config_name,
            "train_split": train_split,
            "train_examples": train_examples,
            "eval_group": eval_group,
            "eval_config_name": eval_config_name,
            "eval_split": eval_split,
            "eval_max_examples": eval_max_examples,
            "eval_scope": eval_scope,
            "runs": len(bucket),
            "examples": bucket[0]["examples"],
            "seeds": ",".join(str(item["seed"]) for item in sorted(bucket, key=lambda item: int(item["seed"]))),
        }
        for metric in METRIC_KEYS:
            values = [float(item[metric]) for item in bucket]
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        aggregated.append(row)

    aggregated.sort(
        key=lambda row: (
            row["study"],
            row["model_tag"],
            math.inf if row.get("train_examples") is None else int(row["train_examples"]),
            EVAL_GROUP_ORDER.get(row["eval_group"], 99),
            row["eval_config_name"],
            row["eval_split"],
            _eval_scope_sort_key(row["eval_scope"]),
            VARIANT_ORDER.get(row["variant"], 99),
        )
    )
    return aggregated


def aggregate_per_class_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["study"],
            row["model_tag"],
            row["variant"],
            row["train_config_name"],
            row["train_split"],
            row["train_examples"],
            row["eval_group"],
            row["eval_config_name"],
            row["eval_split"],
            row["eval_max_examples"],
            row["eval_scope"],
            row["label"],
        )
        grouped[key].append(row)

    aggregated: list[dict] = []
    for key, bucket in grouped.items():
        (
            study,
            model_tag,
            variant,
            train_config_name,
            train_split,
            train_examples,
            eval_group,
            eval_config_name,
            eval_split,
            eval_max_examples,
            eval_scope,
            label,
        ) = key
        aggregated.append(
            {
                "study": study,
                "model_tag": model_tag,
                "variant": variant,
                "train_config_name": train_config_name,
                "train_split": train_split,
                "train_examples": train_examples,
                "eval_group": eval_group,
                "eval_config_name": eval_config_name,
                "eval_split": eval_split,
                "eval_max_examples": eval_max_examples,
                "eval_scope": eval_scope,
                "label": label,
                "runs": len(bucket),
                "seeds": ",".join(str(item["seed"]) for item in sorted(bucket, key=lambda item: int(item["seed"]))),
                "precision_mean": mean(float(item["precision"]) for item in bucket),
                "precision_std": pstdev(float(item["precision"]) for item in bucket) if len(bucket) > 1 else 0.0,
                "recall_mean": mean(float(item["recall"]) for item in bucket),
                "recall_std": pstdev(float(item["recall"]) for item in bucket) if len(bucket) > 1 else 0.0,
                "f1_mean": mean(float(item["f1"]) for item in bucket),
                "f1_std": pstdev(float(item["f1"]) for item in bucket) if len(bucket) > 1 else 0.0,
            }
        )

    aggregated.sort(
        key=lambda row: (
            row["study"],
            row["model_tag"],
            math.inf if row.get("train_examples") is None else int(row["train_examples"]),
            EVAL_GROUP_ORDER.get(row["eval_group"], 99),
            row["eval_config_name"],
            row["eval_split"],
            _eval_scope_sort_key(row["eval_scope"]),
            VARIANT_ORDER.get(row["variant"], 99),
            row["label"],
        )
    )
    return aggregated


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_markdown(raw_rows: list[dict], aggregated_rows: list[dict]) -> str:
    grouped: dict[tuple[str, str, int | None, str, str, str], list[dict]] = defaultdict(list)
    for row in aggregated_rows:
        grouped[
            (
                row["study"],
                row["model_tag"],
                row["train_examples"],
                row["eval_config_name"],
                row["eval_split"],
                row["eval_scope"],
            )
        ].append(row)

    lines = ["# Result Summary", "", "## Aggregated Results", ""]
    for key in sorted(
        grouped.keys(),
        key=lambda item: (
            item[0],
            item[1],
            math.inf if item[2] is None else int(item[2]),
            item[3],
            item[4],
            _eval_scope_sort_key(item[5]),
        ),
    ):
        study, model_tag, train_examples, eval_config_name, eval_split, eval_scope = key
        group_rows = grouped[key]
        eval_group = group_rows[0]["eval_group"]
        train_display = "full" if train_examples is None else str(train_examples)
        lines.append(
            f"### {study} / {model_tag} / train={train_display} / {eval_config_name}/{eval_split} / "
            f"{EVAL_GROUP_LABELS.get(eval_group, eval_group)} / scope={eval_scope}"
        )
        lines.append("")
        lines.append("| Variant | Runs | Acc. | Macro-F1 | Unknown F1 | Faithfulness | Joint | Valid |")
        lines.append("|---------|------|------|----------|------------|--------------|-------|-------|")
        for row in sorted(group_rows, key=lambda item: VARIANT_ORDER.get(item["variant"], 99)):
            lines.append(
                "| "
                + " | ".join(
                    [
                        VARIANT_LABELS.get(row["variant"], row["variant"]),
                        str(row["runs"]),
                        _format_mean_std(row["accuracy_mean"], row["accuracy_std"], row["runs"]),
                        _format_mean_std(row["macro_f1_mean"], row["macro_f1_std"], row["runs"]),
                        _format_mean_std(row["unknown_f1_mean"], row["unknown_f1_std"], row["runs"]),
                        _format_mean_std(
                            row["faithfulness_rate_mean"],
                            row["faithfulness_rate_std"],
                            row["runs"],
                        ),
                        _format_mean_std(row["joint_accuracy_mean"], row["joint_accuracy_std"], row["runs"]),
                        _format_mean_std(
                            row["valid_format_rate_mean"],
                            row["valid_format_rate_std"],
                            row["runs"],
                        ),
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.append("## Raw Runs")
    lines.append("")
    lines.append("| Study | Model | Train | Eval | Scope | Variant | Seed | Acc. | Faithfulness | Joint |")
    lines.append("|-------|-------|-------|------|-------|---------|------|------|--------------|-------|")
    for row in raw_rows:
        train_display = "full" if row["train_examples"] is None else str(row["train_examples"])
        lines.append(
            "| "
            + " | ".join(
                [
                    row["study"],
                    row["model_tag"],
                    train_display,
                    f"{row['eval_config_name']}/{row['eval_split']}",
                    row["eval_scope"],
                    VARIANT_LABELS.get(row["variant"], row["variant"]),
                    str(row["seed"]),
                    _format_pct(row["accuracy"]),
                    _format_pct(row["faithfulness_rate"]),
                    _format_pct(row["joint_accuracy"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-csv", default="results/summary_metrics.csv")
    parser.add_argument("--aggregate-csv", default="results/summary_metrics_agg.csv")
    parser.add_argument("--per-class-csv", default="results/per_class_metrics.csv")
    parser.add_argument("--per-class-aggregate-csv", default="results/per_class_metrics_agg.csv")
    parser.add_argument("--output-md", default="results/summary_metrics.md")
    args = parser.parse_args()

    rows, per_class_rows = load_rows(Path(args.results_dir))
    if not rows:
        raise SystemExit(f"No parsable result JSON files found under {args.results_dir}")

    aggregated_rows = aggregate_rows(rows)
    aggregated_per_class_rows = aggregate_per_class_rows(per_class_rows)

    write_csv(
        Path(args.output_csv),
        rows,
        [
            "study",
            "model_tag",
            "variant",
            "seed",
            "train_config_name",
            "train_split",
            "train_examples",
            "eval_group",
            "eval_config_name",
            "eval_split",
            "eval_max_examples",
            "eval_scope",
            "examples",
            "accuracy",
            "macro_f1",
            "unknown_f1",
            "faithfulness_rate",
            "joint_accuracy",
            "valid_format_rate",
            "path",
        ],
    )
    write_csv(
        Path(args.aggregate_csv),
        aggregated_rows,
        [
            "study",
            "model_tag",
            "variant",
            "train_config_name",
            "train_split",
            "train_examples",
            "eval_group",
            "eval_config_name",
            "eval_split",
            "eval_max_examples",
            "eval_scope",
            "runs",
            "seeds",
            "examples",
            "accuracy_mean",
            "accuracy_std",
            "macro_f1_mean",
            "macro_f1_std",
            "unknown_f1_mean",
            "unknown_f1_std",
            "faithfulness_rate_mean",
            "faithfulness_rate_std",
            "joint_accuracy_mean",
            "joint_accuracy_std",
            "valid_format_rate_mean",
            "valid_format_rate_std",
        ],
    )
    write_csv(
        Path(args.per_class_csv),
        per_class_rows,
        [
            "study",
            "model_tag",
            "variant",
            "seed",
            "train_config_name",
            "train_split",
            "train_examples",
            "eval_group",
            "eval_config_name",
            "eval_split",
            "eval_max_examples",
            "eval_scope",
            "label",
            "precision",
            "recall",
            "f1",
        ],
    )
    write_csv(
        Path(args.per_class_aggregate_csv),
        aggregated_per_class_rows,
        [
            "study",
            "model_tag",
            "variant",
            "train_config_name",
            "train_split",
            "train_examples",
            "eval_group",
            "eval_config_name",
            "eval_split",
            "eval_max_examples",
            "eval_scope",
            "label",
            "runs",
            "seeds",
            "precision_mean",
            "precision_std",
            "recall_mean",
            "recall_std",
            "f1_mean",
            "f1_std",
        ],
    )

    markdown = build_markdown(rows, aggregated_rows)
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
