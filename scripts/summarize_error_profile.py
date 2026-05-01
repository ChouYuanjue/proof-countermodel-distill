#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
GENERATED = ROOT / "paper" / "generated"

VARIANT_ORDER = {
    "answer_only": 0,
    "proof_only": 1,
    "proco": 2,
}
VARIANT_LABELS = {
    "answer_only": "answer-only",
    "proof_only": "proof-only",
    "proco": "\\model{}",
}
DATASET_LABELS = {
    "depth-3ext-NatLang": "ID",
    "depth-5": "Depth-OOD",
}


def _parse_int(value: object) -> int | None:
    if value in {None, "", "None"}:
        return None
    return int(float(value))


def _load_summary_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = dict(row)
            parsed["seed"] = int(parsed["seed"])
            parsed["train_examples"] = _parse_int(parsed.get("train_examples"))
            parsed["eval_max_examples"] = _parse_int(parsed.get("eval_max_examples"))
            parsed["eval_scope"] = parsed.get("eval_scope") or (
                "full" if parsed["eval_max_examples"] is None else f"subset_{parsed['eval_max_examples']}"
            )
            rows.append(parsed)
    return rows


def _metrics_from_predictions(predictions: list[dict]) -> dict:
    gold_unknown = [row for row in predictions if row["gold_label"] == "Unknown"]
    gold_supported = [row for row in predictions if row["gold_label"] != "Unknown"]
    predicted_unknown = [row for row in predictions if row["pred_label"] == "Unknown"]

    support_abstain = [row for row in gold_supported if row["pred_label"] == "Unknown"]
    support_correct = [row for row in gold_supported if row["pred_label"] == row["gold_label"]]
    overcommit = [row for row in gold_unknown if row["pred_label"] != "Unknown"]
    faithful_unknown = [row for row in gold_unknown if row["pred_label"] == "Unknown" and bool(row.get("faithful"))]
    predicted_unknown_correct = [row for row in predicted_unknown if row["gold_label"] == "Unknown"]

    matrix = defaultdict(int)
    for row in predictions:
        matrix[(row["gold_label"], row["pred_label"])] += 1

    return {
        "support_examples": len(gold_supported),
        "support_abstain": len(support_abstain),
        "support_accuracy": len(support_correct),
        "support_abstain_rate": len(support_abstain) / max(1, len(gold_supported)),
        "support_accuracy_rate": len(support_correct) / max(1, len(gold_supported)),
        "unknown_precision": len(predicted_unknown_correct) / max(1, len(predicted_unknown)),
        "unknown_recall": len(predicted_unknown_correct) / max(1, len(gold_unknown)),
        "overcommit_rate": len(overcommit) / max(1, len(gold_unknown)),
        "faithful_unknown_rate": len(faithful_unknown) / max(1, len(gold_unknown)),
        "matrix": matrix,
    }


def _format_pct(mean_value: float, std_value: float, runs: int) -> str:
    if runs > 1:
        return f"{mean_value * 100:.1f} $\\pm$ {std_value * 100:.1f}"
    return f"{mean_value * 100:.1f}"


def _format_count(mean_value: float, std_value: float, runs: int) -> str:
    if runs > 1:
        return f"{mean_value:.1f} $\\pm$ {std_value:.1f}"
    return f"{mean_value:.1f}"


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

    out: list[dict] = []
    for key, bucket in grouped.items():
        study, model_tag, variant, train_examples, eval_config_name, eval_split, eval_scope = key
        record = {
            "study": study,
            "model_tag": model_tag,
            "variant": variant,
            "train_examples": train_examples,
            "eval_config_name": eval_config_name,
            "eval_split": eval_split,
            "eval_scope": eval_scope,
            "runs": len(bucket),
            "seeds": ",".join(str(item["seed"]) for item in sorted(bucket, key=lambda item: item["seed"])),
        }
        metric_names = [
            "support_examples",
            "support_abstain",
            "support_accuracy",
            "support_abstain_rate",
            "support_accuracy_rate",
            "unknown_precision",
            "unknown_recall",
            "overcommit_rate",
            "faithful_unknown_rate",
        ]
        for metric in metric_names:
            values = [item[metric] for item in bucket]
            record[f"{metric}_mean"] = mean(values)
            record[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0

        matrix_keys = ["True", "False", "Unknown"]
        for gold in matrix_keys:
            for pred in matrix_keys:
                values = [item["matrix"][(gold, pred)] for item in bucket]
                record[f"{gold}_to_{pred}_mean"] = mean(values)
                record[f"{gold}_to_{pred}_std"] = pstdev(values) if len(values) > 1 else 0.0
        out.append(record)

    out.sort(
        key=lambda row: (
            row["eval_config_name"],
            VARIANT_ORDER.get(row["variant"], 99),
        )
    )
    return out


def _load_records() -> list[dict]:
    rows = _load_summary_rows(RESULTS / "summary_metrics.csv")
    out: list[dict] = []
    for row in rows:
        if (
            row["study"] != "maintrack"
            or row["model_tag"] != "qwen7b"
            or row["train_examples"] != 4096
            or row["eval_config_name"] not in {"depth-3ext-NatLang", "depth-5"}
            or row["eval_split"] != "test"
            or row["eval_scope"] != "subset_4000"
            or row["variant"] not in VARIANT_ORDER
        ):
            continue
        payload = json.loads(Path(row["path"]).read_text())
        metrics = _metrics_from_predictions(payload["predictions"])
        out.append({**row, **metrics})
    return out


def _write_error_profile(rows: list[dict]) -> None:
    lines = [
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Domain & Variant & Support acc. & Support abstain & Unknown precision & Unknown recall & Over-commit \\\\",
        "\\midrule",
    ]
    current = None
    for row in rows:
        domain = DATASET_LABELS[row["eval_config_name"]]
        if current is not None and domain != current:
            lines.append("\\midrule")
        current = domain
        lines.append(
            " & ".join(
                [
                    domain,
                    VARIANT_LABELS[row["variant"]],
                    _format_pct(row["support_accuracy_rate_mean"], row["support_accuracy_rate_std"], row["runs"]),
                    _format_pct(row["support_abstain_rate_mean"], row["support_abstain_rate_std"], row["runs"]),
                    _format_pct(row["unknown_precision_mean"], row["unknown_precision_std"], row["runs"]),
                    _format_pct(row["unknown_recall_mean"], row["unknown_recall_std"], row["runs"]),
                    _format_pct(row["overcommit_rate_mean"], row["overcommit_rate_std"], row["runs"]),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (GENERATED / "error_profile_table.tex").write_text("\n".join(lines))


def _write_confusion_matrix(rows: list[dict]) -> None:
    lines = [
        "\\begin{tabular}{lllccc}",
        "\\toprule",
        "Domain & Variant & Gold label & Pred True & Pred False & Pred Unknown \\\\",
        "\\midrule",
    ]
    current = None
    for row in rows:
        domain = DATASET_LABELS[row["eval_config_name"]]
        if current is not None and domain != current:
            lines.append("\\midrule")
        current = domain
        for gold in ["True", "False", "Unknown"]:
            lines.append(
                " & ".join(
                    [
                        domain,
                        VARIANT_LABELS[row["variant"]],
                        gold,
                        _format_count(row[f"{gold}_to_True_mean"], row[f"{gold}_to_True_std"], row["runs"]),
                        _format_count(row[f"{gold}_to_False_mean"], row[f"{gold}_to_False_std"], row["runs"]),
                        _format_count(row[f"{gold}_to_Unknown_mean"], row[f"{gold}_to_Unknown_std"], row["runs"]),
                    ]
                )
                + " \\\\"
            )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (GENERATED / "confusion_matrix_table.tex").write_text("\n".join(lines))


def main() -> None:
    rows = _aggregate(_load_records())
    if not rows:
        raise SystemExit("no matching records")
    (RESULTS / "error_profile.csv").parent.mkdir(exist_ok=True)
    with (RESULTS / "error_profile.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _write_error_profile(rows)
    _write_confusion_matrix(rows)


if __name__ == "__main__":
    main()
