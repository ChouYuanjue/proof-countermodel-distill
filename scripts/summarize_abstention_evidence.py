#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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
DATASET_LABELS = {
    "depth-3ext-NatLang": "ID",
    "depth-5": "Depth-OOD",
    "NatLang": "NatLang Transfer",
    "depth-3": "Depth-3 Transfer",
    "birds-electricity": "Birds-Electricity",
}


def _parse_int(value: object) -> int | None:
    if value in {None, "", "None"}:
        return None
    return int(float(value))


def _eval_scope(max_examples: object) -> str:
    parsed = _parse_int(max_examples)
    return "full" if parsed is None else f"subset_{parsed}"


def _metadata(path: Path, payload: dict) -> dict:
    summary = payload["summary"]
    config = payload.get("config", {})
    metadata = payload.get("metadata", {})
    train_metadata = metadata.get("train_metadata") or {}
    train_config = train_metadata.get("config") or {}

    train_examples = metadata.get("train_records") or train_metadata.get("train_records")
    if train_examples is None:
        train_examples = train_config.get("train_max_examples")
    eval_max_examples = metadata.get("eval_max_examples")
    if eval_max_examples is None:
        eval_max_examples = config.get("max_examples")

    return {
        "study": metadata.get("study_tag") or config.get("study_tag") or path.stem.split("_", 1)[0],
        "model_tag": metadata.get("model_tag") or config.get("model_tag") or "base",
        "variant": summary.get("variant") or config.get("variant"),
        "seed": int(config.get("seed", train_config.get("seed", 0))),
        "train_examples": _parse_int(train_examples),
        "eval_config_name": summary.get("config_name") or config.get("config_name"),
        "eval_split": summary.get("split") or config.get("split") or "test",
        "eval_max_examples": _parse_int(eval_max_examples),
        "eval_scope": metadata.get("eval_scope") or _eval_scope(eval_max_examples),
        "path": str(path),
    }


def _summarize_file(path: Path) -> dict | None:
    payload = json.loads(path.read_text())
    if "summary" not in payload or "predictions" not in payload:
        return None
    meta = _metadata(path, payload)
    if not meta["variant"] or not meta["eval_config_name"]:
        return None

    predictions = payload["predictions"]
    pred_unknown = [item for item in predictions if item["pred_label"] == "Unknown"]
    gold_unknown = [item for item in predictions if item["gold_label"] == "Unknown"]
    faithful_unknown = [
        item
        for item in gold_unknown
        if item["pred_label"] == "Unknown" and bool(item.get("faithful"))
    ]
    unfaithful_unknown = [
        item
        for item in gold_unknown
        if item["pred_label"] == "Unknown" and not bool(item.get("faithful"))
    ]
    overcommit = [item for item in gold_unknown if item["pred_label"] != "Unknown"]
    malformed_unknown = [
        item
        for item in unfaithful_unknown
        if (item.get("parsed") or {}).get("mode") != "ABSTAIN"
    ]

    correct_pred_unknown = [item for item in pred_unknown if item["gold_label"] == "Unknown"]
    faithful_pred_unknown = [item for item in pred_unknown if bool(item.get("faithful"))]

    return {
        **meta,
        "examples": len(predictions),
        "gold_unknown": len(gold_unknown),
        "predicted_unknown": len(pred_unknown),
        "correct_predicted_unknown": len(correct_pred_unknown),
        "faithful_predicted_unknown": len(faithful_pred_unknown),
        "faithful_gold_unknown": len(faithful_unknown),
        "unfaithful_gold_unknown": len(unfaithful_unknown),
        "malformed_gold_unknown": len(malformed_unknown),
        "overcommit_gold_unknown": len(overcommit),
        "unknown_precision": len(correct_pred_unknown) / max(1, len(pred_unknown)),
        "unknown_recall": len(correct_pred_unknown) / max(1, len(gold_unknown)),
        "faithful_abstention_precision": len(faithful_pred_unknown) / max(1, len(pred_unknown)),
        "faithful_unknown_rate": len(faithful_unknown) / max(1, len(gold_unknown)),
        "generic_unknown_rate": len(unfaithful_unknown) / max(1, len(gold_unknown)),
        "overcommit_rate": len(overcommit) / max(1, len(gold_unknown)),
    }


def _aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["study"],
            row["model_tag"],
            row["variant"],
            row["train_examples"],
            row["eval_config_name"],
            row["eval_split"],
            row["eval_scope"],
        )
        grouped[key].append(row)

    metric_keys = [
        "predicted_unknown",
        "correct_predicted_unknown",
        "faithful_predicted_unknown",
        "faithful_gold_unknown",
        "unfaithful_gold_unknown",
        "malformed_gold_unknown",
        "overcommit_gold_unknown",
        "unknown_precision",
        "unknown_recall",
        "faithful_abstention_precision",
        "faithful_unknown_rate",
        "generic_unknown_rate",
        "overcommit_rate",
    ]
    out = []
    for key, bucket in grouped.items():
        study, model_tag, variant, train_examples, eval_config_name, eval_split, eval_scope = key
        row = {
            "study": study,
            "model_tag": model_tag,
            "variant": variant,
            "train_examples": train_examples,
            "eval_config_name": eval_config_name,
            "eval_split": eval_split,
            "eval_scope": eval_scope,
            "runs": len(bucket),
            "gold_unknown": bucket[0]["gold_unknown"],
        }
        for metric in metric_keys:
            values = [item[metric] for item in bucket]
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        out.append(row)

    out.sort(
        key=lambda row: (
            row["study"],
            10**12 if row["train_examples"] is None else row["train_examples"],
            row["eval_config_name"],
            VARIANT_ORDER.get(row["variant"], 99),
        )
    )
    return out


def _fmt_count(row: dict, metric: str) -> str:
    mean_value = row[f"{metric}_mean"]
    std_value = row[f"{metric}_std"]
    if row["runs"] > 1:
        return f"{mean_value:.1f} $\\pm$ {std_value:.1f}"
    return f"{mean_value:.1f}"


def _fmt_pct(row: dict, metric: str) -> str:
    mean_value = row[f"{metric}_mean"] * 100
    std_value = row[f"{metric}_std"] * 100
    if row["runs"] > 1:
        return f"{mean_value:.1f} $\\pm$ {std_value:.1f}"
    return f"{mean_value:.1f}"


def _main_rows(agg_rows: list[dict]) -> list[dict]:
    return [
        row
        for row in agg_rows
        if row["study"] == "maintrack"
        and row["model_tag"] == "qwen7b"
        and row["train_examples"] == 4096
        and row["eval_config_name"] in {"depth-3ext-NatLang", "depth-5"}
        and row["eval_split"] == "test"
        and row["eval_scope"] == "subset_4000"
        and row["variant"] in {"proof_only", "proco"}
    ]


def _write_markdown(path: Path, agg_rows: list[dict]) -> None:
    rows = _main_rows(agg_rows)
    lines = [
        "# Abstention Evidence Summary",
        "",
        "This table focuses on gold-unknown examples in the fixed 7B 4k seeded suite.",
        "",
        "| Domain | Variant | Runs | Pred. Unknown | Verifier-accepted Unknown | Generic Unknown | Over-Commit | Verifier-accepted Unknown Rate |",
        "|--------|---------|------|---------------|---------------------------|-----------------|-------------|--------------------------------|",
    ]
    for row in rows:
        lines.append(
            " | ".join(
                [
                    f"| {DATASET_LABELS.get(row['eval_config_name'], row['eval_config_name'])}",
                    VARIANT_LABELS.get(row["variant"], row["variant"]),
                    str(row["runs"]),
                    _fmt_count(row, "predicted_unknown"),
                    _fmt_count(row, "faithful_gold_unknown"),
                    _fmt_count(row, "unfaithful_gold_unknown"),
                    _fmt_count(row, "overcommit_gold_unknown"),
                    _fmt_pct(row, "faithful_unknown_rate") + " |",
                ]
            )
        )
    lines.append("")
    lines.append(
        "Generic Unknown means the gold label is `Unknown` and the model also predicts "
        "`Unknown`, but the generated abstention witness is not verifier-accepted."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _write_latex(path: Path, agg_rows: list[dict]) -> None:
    rows = _main_rows(agg_rows)
    lines = [
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Domain & Variant & Verifier-accepted & Generic & Over-Commit & Verifier-accepted Rate \\\\",
        "\\midrule",
    ]
    current_dataset = None
    for row in rows:
        if current_dataset is not None and current_dataset != row["eval_config_name"]:
            lines.append("\\midrule")
        current_dataset = row["eval_config_name"]
        lines.append(
            " & ".join(
                [
                    DATASET_LABELS.get(row["eval_config_name"], row["eval_config_name"]),
                    VARIANT_LABELS.get(row["variant"], row["variant"]),
                    _fmt_count(row, "faithful_gold_unknown"),
                    _fmt_count(row, "unfaithful_gold_unknown"),
                    _fmt_count(row, "overcommit_gold_unknown"),
                    _fmt_pct(row, "faithful_unknown_rate"),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--csv-output", default="results/abstention_evidence.csv")
    parser.add_argument("--agg-csv-output", default="results/abstention_evidence_agg.csv")
    parser.add_argument("--md-output", default="results/abstention_evidence.md")
    parser.add_argument("--latex-output", default="paper/generated/abstention_evidence_table.tex")
    args = parser.parse_args()

    rows = []
    for path in sorted(Path(args.results_dir).glob("*.json")):
        row = _summarize_file(path)
        if row is not None:
            rows.append(row)

    csv_path = Path(args.csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    agg_rows = _aggregate(rows)
    agg_path = Path(args.agg_csv_output)
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    agg_fieldnames = list(agg_rows[0].keys()) if agg_rows else []
    with agg_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=agg_fieldnames)
        writer.writeheader()
        writer.writerows(agg_rows)

    _write_markdown(Path(args.md_output), agg_rows)
    _write_latex(Path(args.latex_output), agg_rows)
    print(args.md_output)
    print(args.latex_output)


if __name__ == "__main__":
    main()
