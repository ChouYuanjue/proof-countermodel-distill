#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
GENERATED = ROOT / "paper" / "generated"

DATASET_LABELS = {
    "depth-3ext-NatLang": "ID",
    "depth-5": "Depth-OOD",
}
VARIANT_ORDER = {"answer_only": 0, "proof_only": 1, "proco": 2}
VARIANT_LABELS = {
    "answer_only": "answer-only",
    "proof_only": "proof-only",
    "proco": "\\model{}",
}


def _parse_int(value: str | None) -> int | None:
    if value in {None, "", "None"}:
        return None
    return int(float(value))


def _load_summary_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = dict(row)
            parsed["seed"] = int(parsed["seed"])
            parsed["train_examples"] = _parse_int(parsed.get("train_examples"))
            parsed["eval_max_examples"] = _parse_int(parsed.get("eval_max_examples"))
            parsed["examples"] = int(parsed["examples"])
            for key in [
                "accuracy",
                "macro_f1",
                "unknown_f1",
                "faithfulness_rate",
                "joint_accuracy",
            ]:
                if key in parsed:
                    parsed[key] = float(parsed[key])
            rows.append(parsed)
    return rows


def _load_unknown_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = dict(row)
            parsed["seed"] = int(parsed["seed"])
            parsed["train_examples"] = _parse_int(parsed.get("train_examples"))
            parsed["eval_max_examples"] = _parse_int(parsed.get("eval_max_examples"))
            for key in [
                "gold_unknown",
                "predicted_unknown",
                "faithful_unknown",
                "overcommit",
                "predicted_unknown_rate",
                "faithful_unknown_rate",
                "overcommit_rate",
            ]:
                if key in parsed:
                    parsed[key] = float(parsed[key])
            rows.append(parsed)
    return rows


def _index(rows: list[dict]) -> dict[tuple, dict]:
    indexed: dict[tuple, dict] = {}
    for row in rows:
        key = (
            row["study"],
            row["model_tag"],
            row["variant"],
            row["train_examples"],
            row["eval_config_name"],
            row["eval_split"],
            row["eval_scope"],
            row["seed"],
        )
        indexed[key] = row
    return indexed


def _group_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
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
        grouped.setdefault(key, []).append(row)

    aggregated: list[dict] = []
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
        }
        for metric in ["current_joint", "repaired_joint", "uplift_pp"]:
            values = [item[metric] for item in bucket]
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        aggregated.append(row)

    aggregated.sort(
        key=lambda row: (
            row["eval_config_name"],
            VARIANT_ORDER.get(row["variant"], 99),
        )
    )
    return aggregated


def _fmt_pct(row: dict, metric: str) -> str:
    mean_value = row[f"{metric}_mean"] * 100
    std_value = row[f"{metric}_std"] * 100
    return f"{mean_value:.1f} $\\pm$ {std_value:.1f}" if row["runs"] > 1 else f"{mean_value:.1f}"


def _fmt_uplift(row: dict) -> str:
    mean_value = row["uplift_pp_mean"] * 100
    std_value = row["uplift_pp_std"] * 100
    if row["runs"] > 1:
        return f"+{mean_value:.1f} $\\pm$ {std_value:.1f}"
    return f"+{mean_value:.1f}"


def main() -> None:
    summary_rows = _load_summary_csv(RESULTS / "summary_metrics.csv")
    unknown_rows = _load_unknown_csv(RESULTS / "unknown_behavior.csv")
    summary_index = _index(summary_rows)
    unknown_index = _index(unknown_rows)

    raw_rows: list[dict] = []
    for key, summary in summary_index.items():
        study, model_tag, variant, train_examples, eval_config_name, eval_split, eval_scope, seed = key
        if (
            study != "maintrack"
            or model_tag != "qwen7b"
            or variant not in {"answer_only", "proof_only", "proco"}
            or train_examples != 4096
            or eval_config_name not in {"depth-3ext-NatLang", "depth-5"}
            or eval_split != "test"
            or eval_scope != "subset_4000"
        ):
            continue
        unknown = unknown_index[key]
        examples = summary["examples"]
        current_joint = summary["joint_accuracy"]
        repaired_joint = current_joint + (unknown["predicted_unknown"] - unknown["faithful_unknown"]) / examples
        raw_rows.append(
            {
                "study": study,
                "model_tag": model_tag,
                "variant": variant,
                "train_examples": train_examples,
                "eval_config_name": eval_config_name,
                "eval_split": eval_split,
                "eval_scope": eval_scope,
                "seed": seed,
                "current_joint": current_joint,
                "repaired_joint": repaired_joint,
                "uplift_pp": repaired_joint - current_joint,
            }
        )

    if not raw_rows:
        raise SystemExit("no matching rows for symbolic repair summary")

    aggregated = _group_rows(raw_rows)
    RESULTS.mkdir(exist_ok=True)
    GENERATED.mkdir(parents=True, exist_ok=True)

    csv_path = RESULTS / "symbolic_repair_agg.csv"
    with csv_path.open("w", newline="") as handle:
        fieldnames = [
            "study",
            "model_tag",
            "variant",
            "train_examples",
            "eval_config_name",
            "eval_split",
            "eval_scope",
            "runs",
            "current_joint_mean",
            "current_joint_std",
            "repaired_joint_mean",
            "repaired_joint_std",
            "uplift_pp_mean",
            "uplift_pp_std",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregated)

    md_lines = [
        "# Post-hoc Symbolic Repair Summary",
        "",
        "This is a solver-assisted upper bound: when the model already predicts `Unknown`, we attach the canonical witness from the same closure search.",
        "",
        "| Domain | Variant | Current joint | Repaired joint | Uplift |",
        "|--------|---------|---------------|----------------|--------|",
    ]
    for row in aggregated:
        md_lines.append(
            "| "
            + " | ".join(
                [
                    DATASET_LABELS[row["eval_config_name"]],
                    VARIANT_LABELS[row["variant"]],
                    _fmt_pct(row, "current_joint"),
                    _fmt_pct(row, "repaired_joint"),
                    _fmt_uplift(row),
                ]
            )
            + " |"
        )
    (RESULTS / "symbolic_repair.md").write_text("\n".join(md_lines) + "\n")

    tex_lines = [
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Domain & Variant & Current joint & Repaired joint & Uplift \\\\",
        "\\midrule",
    ]
    current_dataset = None
    for row in aggregated:
        if current_dataset is not None and current_dataset != row["eval_config_name"]:
            tex_lines.append("\\midrule")
        current_dataset = row["eval_config_name"]
        tex_lines.append(
            " & ".join(
                [
                    DATASET_LABELS[row["eval_config_name"]],
                    VARIANT_LABELS[row["variant"]],
                    _fmt_pct(row, "current_joint"),
                    _fmt_pct(row, "repaired_joint"),
                    _fmt_uplift(row),
                ]
            )
            + " \\\\"
        )
    tex_lines.extend(["\\bottomrule", "\\end{tabular}"])
    (GENERATED / "symbolic_repair_table.tex").write_text("\n".join(tex_lines))

    print(csv_path)
    print(RESULTS / "symbolic_repair.md")
    print(GENERATED / "symbolic_repair_table.tex")


if __name__ == "__main__":
    main()
