#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


VARIANT_ORDER = {
    "answer_only": 0,
    "proof_only": 1,
    "proco_chain": 2,
    "proco_witness": 3,
    "proco": 4,
}
VARIANT_LABELS = {
    "answer_only": "answer-only",
    "proof_only": "proof-only",
    "proco_chain": "ProCo-chain",
    "proco_witness": "ProCo-witness",
    "proco": "ProCo",
}
DATASET_LABELS = {
    "depth-3ext-NatLang": "ID",
    "depth-5": "Depth-OOD",
    "NatLang": "NatLang Transfer",
    "depth-3": "Depth-3 Transfer",
    "birds-electricity": "Birds-Electricity",
}


def _parse_int(value: str | None) -> int | None:
    if value in {None, "", "None"}:
        return None
    return int(float(value))


def _parse_scope(row: dict) -> str:
    eval_scope = row.get("eval_scope")
    if eval_scope:
        return eval_scope
    eval_max_examples = _parse_int(row.get("eval_max_examples"))
    return "full" if eval_max_examples is None else f"subset_{eval_max_examples}"


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = dict(row)
            parsed["train_examples"] = _parse_int(parsed.get("train_examples"))
            parsed["eval_max_examples"] = _parse_int(parsed.get("eval_max_examples"))
            parsed["eval_scope"] = _parse_scope(parsed)
            parsed["runs"] = int(parsed["runs"])
            for key in [
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
            ]:
                parsed[key] = float(parsed[key])
            rows.append(parsed)
    return rows


def load_unknown_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = dict(row)
            parsed["train_examples"] = _parse_int(parsed.get("train_examples"))
            parsed["eval_max_examples"] = _parse_int(parsed.get("eval_max_examples"))
            parsed["eval_scope"] = _parse_scope(parsed)
            parsed["runs"] = int(parsed["runs"])
            parsed["gold_unknown"] = int(float(parsed["gold_unknown"]))
            for key in [
                "predicted_unknown_mean",
                "predicted_unknown_std",
                "faithful_unknown_mean",
                "faithful_unknown_std",
                "overcommit_mean",
                "overcommit_std",
            ]:
                parsed[key] = float(parsed[key])
            rows.append(parsed)
    return rows


def _fmt(row: dict, key: str) -> str:
    mean_key = f"{key}_mean"
    std_key = f"{key}_std"
    if row["runs"] > 1:
        return f"{row[mean_key] * 100:.1f} $\\pm$ {row[std_key] * 100:.1f}"
    return f"{row[mean_key] * 100:.1f}"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _row_matches(row: dict, *, study: str, train_examples: int | None, datasets: set[str], variants: set[str]) -> bool:
    return (
        row["study"] == study
        and row["train_examples"] == train_examples
        and row["eval_config_name"] in datasets
        and row["variant"] in variants
    )


def build_seed_table(rows: list[dict]) -> str:
    picked = [
        row
        for row in rows
        if _row_matches(
            row,
            study="maintrack",
            train_examples=4096,
            datasets={"depth-3ext-NatLang", "depth-5"},
            variants={"answer_only", "proof_only", "proco"},
        )
        and row["eval_split"] == "test"
        and row["eval_scope"] == "subset_4000"
    ]
    picked.sort(key=lambda row: (row["eval_config_name"], VARIANT_ORDER.get(row["variant"], 99)))

    lines = [
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Domain & Variant & Runs & Acc. & Unknown F1 & Faithful & Joint \\\\",
        "\\midrule",
    ]
    current_dataset = None
    for row in picked:
        if current_dataset is not None and current_dataset != row["eval_config_name"]:
            lines.append("\\midrule")
        current_dataset = row["eval_config_name"]
        lines.append(
            " & ".join(
                [
                    DATASET_LABELS.get(row["eval_config_name"], row["eval_config_name"]),
                    VARIANT_LABELS.get(row["variant"], row["variant"]),
                    str(row["runs"]),
                    _fmt(row, "accuracy"),
                    _fmt(row, "unknown_f1"),
                    _fmt(row, "faithfulness_rate"),
                    _fmt(row, "joint_accuracy"),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def build_scaling_table(rows: list[dict]) -> str:
    picked = [
        row
        for row in rows
        if row["study"] == "maintrack"
        and row["variant"] in {"answer_only", "proof_only", "proco"}
        and row["eval_config_name"] in {"depth-3ext-NatLang", "depth-5"}
        and row["eval_split"] == "test"
        and row["train_examples"] is not None
        and row["eval_scope"] == "subset_4000"
    ]
    picked.sort(
        key=lambda row: (
            row["train_examples"],
            row["eval_config_name"],
            VARIANT_ORDER.get(row["variant"], 99),
        )
    )

    lines = [
        "\\begin{tabular}{rllccc}",
        "\\toprule",
        "Train & Domain & Variant & Acc. & Faithful & Joint \\\\",
        "\\midrule",
    ]
    last_train = None
    last_dataset = None
    for row in picked:
        if last_train is not None and last_train != row["train_examples"]:
            lines.append("\\midrule")
            last_dataset = None
        elif last_dataset is not None and last_dataset != row["eval_config_name"]:
            lines.append("\\cmidrule(lr){2-6}")
        last_train = row["train_examples"]
        last_dataset = row["eval_config_name"]
        lines.append(
            " & ".join(
                [
                    str(row["train_examples"]),
                    DATASET_LABELS.get(row["eval_config_name"], row["eval_config_name"]),
                    VARIANT_LABELS.get(row["variant"], row["variant"]),
                    _fmt(row, "accuracy"),
                    _fmt(row, "faithfulness_rate"),
                    _fmt(row, "joint_accuracy"),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def build_ablation_table(rows: list[dict]) -> str:
    picked = [
        row
        for row in rows
        if row["train_examples"] == 4096
        and row["eval_config_name"] in {"depth-3ext-NatLang", "depth-5"}
        and row["eval_split"] == "test"
        and row["eval_scope"] == "subset_4000"
        and (
            (row["study"] == "maintrack" and row["variant"] in {"proof_only", "proco"})
            or (row["study"] == "ablation" and row["variant"] in {"proco_chain", "proco_witness"})
        )
    ]
    picked.sort(key=lambda row: (row["eval_config_name"], VARIANT_ORDER.get(row["variant"], 99)))

    lines = [
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Domain & Variant & Runs & Acc. & Faithful & Joint \\\\",
        "\\midrule",
    ]
    current_dataset = None
    for row in picked:
        if current_dataset is not None and current_dataset != row["eval_config_name"]:
            lines.append("\\midrule")
        current_dataset = row["eval_config_name"]
        lines.append(
            " & ".join(
                [
                    DATASET_LABELS.get(row["eval_config_name"], row["eval_config_name"]),
                    VARIANT_LABELS.get(row["variant"], row["variant"]),
                    str(row["runs"]),
                    _fmt(row, "accuracy"),
                    _fmt(row, "faithfulness_rate"),
                    _fmt(row, "joint_accuracy"),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def build_transfer_table(rows: list[dict]) -> str:
    picked = [
        row
        for row in rows
        if row["study"] == "maintrack"
        and row["train_examples"] in {32768, None}
        and row["variant"] in {"answer_only", "proof_only", "proco"}
        and row["eval_config_name"] in {"depth-3ext-NatLang", "depth-5", "NatLang", "depth-3", "birds-electricity"}
        and row["eval_split"] == "test"
        and row["eval_scope"] == "full"
    ]
    picked.sort(
        key=lambda row: (
            -1 if row["train_examples"] is None else row["train_examples"],
            row["eval_config_name"],
            VARIANT_ORDER.get(row["variant"], 99),
        )
    )

    lines = [
        "\\begin{tabular}{rllccc}",
        "\\toprule",
        "Train & Eval Set & Variant & Acc. & Faithful & Joint \\\\",
        "\\midrule",
    ]
    last_train = None
    last_dataset = None
    for row in picked:
        train_display = "full" if row["train_examples"] is None else str(row["train_examples"])
        if last_train is not None and last_train != train_display:
            lines.append("\\midrule")
            last_dataset = None
        elif last_dataset is not None and last_dataset != row["eval_config_name"]:
            lines.append("\\cmidrule(lr){2-6}")
        last_train = train_display
        last_dataset = row["eval_config_name"]
        lines.append(
            " & ".join(
                [
                    train_display,
                    DATASET_LABELS.get(row["eval_config_name"], row["eval_config_name"]),
                    VARIANT_LABELS.get(row["variant"], row["variant"]),
                    _fmt(row, "accuracy"),
                    _fmt(row, "faithfulness_rate"),
                    _fmt(row, "joint_accuracy"),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def build_unknown_table(rows: list[dict]) -> str:
    picked = [
        row
        for row in rows
        if row["train_examples"] == 4096
        and row["eval_config_name"] in {"depth-3ext-NatLang", "depth-5"}
        and row["eval_split"] == "test"
        and row["variant"] in {"proof_only", "proco"}
        and row["study"] == "maintrack"
        and row["eval_scope"] == "subset_4000"
    ]
    picked.sort(key=lambda row: (row["eval_config_name"], VARIANT_ORDER.get(row["variant"], 99)))

    lines = [
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Domain & Variant & Pred. Unknown & Faithful Unknown & Over-Commit \\\\",
        "\\midrule",
    ]
    current_dataset = None
    for row in picked:
        if current_dataset is not None and current_dataset != row["eval_config_name"]:
            lines.append("\\midrule")
        current_dataset = row["eval_config_name"]
        pred_unknown = f"{row['predicted_unknown_mean']:.1f}" if row["runs"] == 1 else f"{row['predicted_unknown_mean']:.1f} $\\pm$ {row['predicted_unknown_std']:.1f}"
        faithful_unknown = f"{row['faithful_unknown_mean']:.1f}" if row["runs"] == 1 else f"{row['faithful_unknown_mean']:.1f} $\\pm$ {row['faithful_unknown_std']:.1f}"
        overcommit = f"{row['overcommit_mean']:.1f}" if row["runs"] == 1 else f"{row['overcommit_mean']:.1f} $\\pm$ {row['overcommit_std']:.1f}"
        lines.append(
            " & ".join(
                [
                    DATASET_LABELS.get(row["eval_config_name"], row["eval_config_name"]),
                    VARIANT_LABELS.get(row["variant"], row["variant"]),
                    pred_unknown,
                    faithful_unknown,
                    overcommit,
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", default="results/summary_metrics_agg.csv")
    parser.add_argument("--unknown-csv", default="results/unknown_behavior_agg.csv")
    parser.add_argument("--output-dir", default="paper/generated")
    args = parser.parse_args()

    rows = load_rows(Path(args.summary_csv))
    unknown_rows = load_unknown_rows(Path(args.unknown_csv))
    output_dir = Path(args.output_dir)

    outputs = {
        "seed_table.tex": build_seed_table(rows),
        "scaling_table.tex": build_scaling_table(rows),
        "ablation_table.tex": build_ablation_table(rows),
        "transfer_table.tex": build_transfer_table(rows),
        "unknown_table.tex": build_unknown_table(unknown_rows),
    }
    for name, text in outputs.items():
        path = output_dir / name
        _write(path, text)
        print(path)


if __name__ == "__main__":
    main()
