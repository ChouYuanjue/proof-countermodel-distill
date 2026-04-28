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
    "proco": 2,
}
VARIANT_LABELS = {
    "answer_only": "answer-only",
    "proof_only": "proof-only",
    "proco": "ProCo",
}
MODEL_LABELS = {
    "qwen7b": "Qwen2.5-7B",
    "mistral7b": "Mistral-7B",
}
KIND_LABELS = {
    "fact": "Deleted Fact",
    "rule": "Deleted Rule",
}


def _parse_int(value: str | int | None) -> int | None:
    if value in {None, "", "None"}:
        return None
    return int(float(value))


def _infer_train_examples(payload: dict) -> int | None:
    metadata = payload.get("metadata", {})
    train_metadata = metadata.get("train_metadata") or {}
    train_config = train_metadata.get("config") or {}
    return _parse_int(
        metadata.get("train_records")
        or metadata.get("train_max_examples")
        or train_metadata.get("train_records")
        or train_config.get("train_max_examples")
    )


def _eval_scope(payload: dict) -> str:
    metadata = payload.get("metadata", {})
    config = payload.get("config", {})
    max_examples = _parse_int(metadata.get("eval_max_examples") or config.get("max_examples"))
    return "full" if max_examples is None else f"subset_{max_examples}"


def _summarize_bucket(predictions: list[dict]) -> dict:
    predicted_unknown = [item for item in predictions if item["pred_label"] == "Unknown"]
    faithful_unknown = [
        item
        for item in predictions
        if item["pred_label"] == "Unknown" and item.get("faithful")
    ]
    overcommit = [item for item in predictions if item["pred_label"] != "Unknown"]
    return {
        "examples": len(predictions),
        "predicted_unknown": len(predicted_unknown),
        "faithful_unknown": len(faithful_unknown),
        "overcommit": len(overcommit),
        "predicted_unknown_rate": len(predicted_unknown) / max(1, len(predictions)),
        "faithful_unknown_rate": len(faithful_unknown) / max(1, len(predictions)),
        "overcommit_rate": len(overcommit) / max(1, len(predictions)),
    }


def load_rows(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(results_dir.glob("mutation_*support-deletion*.json")):
        payload = json.loads(path.read_text())
        summary = payload.get("summary", {})
        config = payload.get("config", {})
        metadata = payload.get("metadata", {})
        if summary.get("eval_group") != "support_deletion":
            continue
        base = {
            "path": str(path),
            "study": metadata.get("study_tag") or config.get("study_tag") or "mutation",
            "model_tag": metadata.get("model_tag") or config.get("model_tag") or "base",
            "variant": summary.get("variant") or config.get("variant"),
            "seed": int(config.get("seed", 0)),
            "train_examples": _infer_train_examples(payload),
            "eval_config_name": summary.get("config_name", "support-deletion"),
            "eval_split": summary.get("split") or config.get("split") or "test",
            "eval_scope": _eval_scope(payload),
        }
        predictions = payload["predictions"]
        for group_name, bucket in [
            ("overall", predictions),
            *[
                (
                    f"deleted_{kind}",
                    [
                        item
                        for item in predictions
                        if (item.get("mutation_metadata") or {}).get("deleted_kind") == kind
                    ],
                )
                for kind in ["fact", "rule"]
            ],
            *[
                (
                    f"source_{label.lower()}",
                    [
                        item
                        for item in predictions
                        if (item.get("mutation_metadata") or {}).get("source_answer") == label
                    ],
                )
                for label in ["True", "False"]
            ],
        ]:
            if not bucket:
                continue
            rows.append({**base, "group": group_name, **_summarize_bucket(bucket)})
    rows.sort(
        key=lambda row: (
            row["study"],
            row["model_tag"],
            row["train_examples"] or -1,
            row["eval_scope"],
            row["group"],
            VARIANT_ORDER.get(row["variant"], 99),
            row["seed"],
        )
    )
    return rows


def aggregate_rows(rows: list[dict]) -> list[dict]:
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
            row["group"],
        )
        grouped[key].append(row)

    aggregated: list[dict] = []
    for key, bucket in grouped.items():
        (
            study,
            model_tag,
            variant,
            train_examples,
            eval_config_name,
            eval_split,
            eval_scope,
            group,
        ) = key
        row = {
            "study": study,
            "model_tag": model_tag,
            "variant": variant,
            "train_examples": train_examples,
            "eval_config_name": eval_config_name,
            "eval_split": eval_split,
            "eval_scope": eval_scope,
            "group": group,
            "runs": len(bucket),
            "seeds": ",".join(str(item["seed"]) for item in sorted(bucket, key=lambda item: item["seed"])),
            "examples": bucket[0]["examples"],
        }
        for metric in [
            "predicted_unknown",
            "faithful_unknown",
            "overcommit",
            "predicted_unknown_rate",
            "faithful_unknown_rate",
            "overcommit_rate",
        ]:
            values = [float(item[metric]) for item in bucket]
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        aggregated.append(row)

    aggregated.sort(
        key=lambda row: (
            row["study"],
            {"qwen7b": 0, "mistral7b": 1}.get(row["model_tag"], 99),
            row["train_examples"] or -1,
            row["eval_scope"],
            row["group"],
            VARIANT_ORDER.get(row["variant"], 99),
        )
    )
    return aggregated


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt_count(row: dict, key: str) -> str:
    mean_key = f"{key}_mean"
    std_key = f"{key}_std"
    if row["runs"] > 1:
        return f"{row[mean_key]:.1f} $\\pm$ {row[std_key]:.1f}"
    return f"{row[mean_key]:.1f}"


def _fmt_pct(row: dict, key: str) -> str:
    mean_key = f"{key}_mean"
    std_key = f"{key}_std"
    if row["runs"] > 1:
        return f"{row[mean_key] * 100:.1f} $\\pm$ {row[std_key] * 100:.1f}"
    return f"{row[mean_key] * 100:.1f}"


def write_latex(path: Path, rows: list[dict]) -> None:
    picked = [
        row
        for row in rows
        if row["study"] == "mutation"
        and row["model_tag"] == "qwen7b"
        and row["train_examples"] == 4096
        and row["eval_scope"] == "subset_4000"
        and row["group"] in {"deleted_fact", "deleted_rule"}
        and row["variant"] in {"answer_only", "proof_only", "proco"}
    ]
    picked.sort(
        key=lambda row: (
            {"deleted_fact": 0, "deleted_rule": 1}.get(row["group"], 99),
            VARIANT_ORDER.get(row["variant"], 99),
        )
    )
    lines = [
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Deleted Support & Variant & Runs & Pred. Unknown & Faithful Unknown & Faithful Rate \\\\",
        "\\midrule",
    ]
    current_group = None
    for row in picked:
        if current_group is not None and current_group != row["group"]:
            lines.append("\\midrule")
        current_group = row["group"]
        kind = row["group"].replace("deleted_", "")
        lines.append(
            " & ".join(
                [
                    KIND_LABELS.get(kind, kind),
                    VARIANT_LABELS.get(row["variant"], row["variant"]),
                    str(row["runs"]),
                    _fmt_count(row, "predicted_unknown"),
                    _fmt_count(row, "faithful_unknown"),
                    _fmt_pct(row, "faithful_unknown_rate"),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def build_markdown(rows: list[dict]) -> str:
    lines = ["# Support-Deletion Mutation Summary", ""]
    for row in rows:
        train = "full" if row["train_examples"] is None else str(row["train_examples"])
        lines.append(
            "- "
            f"{row['study']} / {MODEL_LABELS.get(row['model_tag'], row['model_tag'])} / "
            f"{VARIANT_LABELS.get(row['variant'], row['variant'])} / train={train} / "
            f"{row['group']}: pred_unknown={row['predicted_unknown_mean']:.1f}, "
            f"faithful={row['faithful_unknown_mean']:.1f}, "
            f"faithful_rate={row['faithful_unknown_rate_mean'] * 100:.1f}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-csv", default="results/support_deletion.csv")
    parser.add_argument("--aggregate-csv", default="results/support_deletion_agg.csv")
    parser.add_argument("--output-md", default="results/support_deletion.md")
    parser.add_argument("--latex-output", default="paper/generated/support_deletion_breakdown_table.tex")
    args = parser.parse_args()

    rows = load_rows(Path(args.results_dir))
    aggregated = aggregate_rows(rows)

    raw_fields = [
        "study",
        "model_tag",
        "variant",
        "seed",
        "train_examples",
        "eval_config_name",
        "eval_split",
        "eval_scope",
        "group",
        "examples",
        "predicted_unknown",
        "faithful_unknown",
        "overcommit",
        "predicted_unknown_rate",
        "faithful_unknown_rate",
        "overcommit_rate",
        "path",
    ]
    agg_fields = [
        "study",
        "model_tag",
        "variant",
        "train_examples",
        "eval_config_name",
        "eval_split",
        "eval_scope",
        "group",
        "runs",
        "seeds",
        "examples",
        "predicted_unknown_mean",
        "predicted_unknown_std",
        "faithful_unknown_mean",
        "faithful_unknown_std",
        "overcommit_mean",
        "overcommit_std",
        "predicted_unknown_rate_mean",
        "predicted_unknown_rate_std",
        "faithful_unknown_rate_mean",
        "faithful_unknown_rate_std",
        "overcommit_rate_mean",
        "overcommit_rate_std",
    ]
    write_csv(Path(args.output_csv), rows, raw_fields)
    write_csv(Path(args.aggregate_csv), aggregated, agg_fields)
    Path(args.output_md).write_text(build_markdown(aggregated))
    write_latex(Path(args.latex_output), aggregated)
    print(args.output_md)
    print(args.latex_output)


if __name__ == "__main__":
    main()
