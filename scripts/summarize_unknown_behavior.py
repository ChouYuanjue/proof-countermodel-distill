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


def _parse_int(value: str | None) -> int | None:
    if value in {None, "", "None"}:
        return None
    return int(float(value))


def _scope_sort_key(scope: str) -> tuple[int, int]:
    if scope.startswith("subset_"):
        return (0, _parse_int(scope.split("_", 1)[1]) or 0)
    return (1, 0)


def load_summary_rows(path: Path) -> list[dict]:
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


def summarize_unknown(row: dict) -> dict:
    payload = json.loads(Path(row["path"]).read_text())
    predictions = payload["predictions"]
    gold_unknown = [item for item in predictions if item["gold_label"] == "Unknown"]
    predicted_unknown = [item for item in gold_unknown if item["pred_label"] == "Unknown"]
    faithful_unknown = [item for item in gold_unknown if item["pred_label"] == "Unknown" and item["faithful"]]
    overcommit = [item for item in gold_unknown if item["pred_label"] != "Unknown"]

    return {
        "study": row["study"],
        "model_tag": row["model_tag"],
        "variant": row["variant"],
        "seed": row["seed"],
        "train_examples": row["train_examples"],
        "eval_config_name": row["eval_config_name"],
        "eval_split": row["eval_split"],
        "eval_max_examples": row["eval_max_examples"],
        "eval_scope": row["eval_scope"],
        "gold_unknown": len(gold_unknown),
        "predicted_unknown": len(predicted_unknown),
        "faithful_unknown": len(faithful_unknown),
        "overcommit": len(overcommit),
        "predicted_unknown_rate": len(predicted_unknown) / max(1, len(gold_unknown)),
        "faithful_unknown_rate": len(faithful_unknown) / max(1, len(gold_unknown)),
        "overcommit_rate": len(overcommit) / max(1, len(gold_unknown)),
    }


def aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["study"],
            row["model_tag"],
            row["variant"],
            row["train_examples"],
            row["eval_config_name"],
            row["eval_split"],
            row["eval_max_examples"],
            row["eval_scope"],
        )
        grouped[key].append(row)

    aggregated: list[dict] = []
    for key, bucket in grouped.items():
        study, model_tag, variant, train_examples, eval_config_name, eval_split, eval_max_examples, eval_scope = key
        agg = {
            "study": study,
            "model_tag": model_tag,
            "variant": variant,
            "train_examples": train_examples,
            "eval_config_name": eval_config_name,
            "eval_split": eval_split,
            "eval_max_examples": eval_max_examples,
            "eval_scope": eval_scope,
            "runs": len(bucket),
            "gold_unknown": bucket[0]["gold_unknown"],
            "seeds": ",".join(str(item["seed"]) for item in sorted(bucket, key=lambda item: item["seed"])),
        }
        for key_name in [
            "predicted_unknown",
            "faithful_unknown",
            "overcommit",
            "predicted_unknown_rate",
            "faithful_unknown_rate",
            "overcommit_rate",
        ]:
            values = [item[key_name] for item in bucket]
            agg[f"{key_name}_mean"] = mean(values)
            agg[f"{key_name}_std"] = pstdev(values) if len(values) > 1 else 0.0
        aggregated.append(agg)

    aggregated.sort(
        key=lambda row: (
            row["study"],
            row["model_tag"],
            -1 if row["train_examples"] is None else row["train_examples"],
            row["eval_config_name"],
            row["eval_split"],
            _scope_sort_key(row["eval_scope"]),
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


def build_markdown(rows: list[dict]) -> str:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
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

    lines = ["# Unknown Behavior Summary", ""]
    for key in sorted(
        grouped.keys(),
        key=lambda item: (
            item[0],
            item[1],
            -1 if item[2] is None else item[2],
            item[3],
            item[4],
            _scope_sort_key(item[5]),
        ),
    ):
        study, model_tag, train_examples, eval_config_name, eval_split, eval_scope = key
        train_display = "full" if train_examples is None else str(train_examples)
        lines.append(
            f"## {study} / {model_tag} / train={train_display} / {eval_config_name}/{eval_split} / scope={eval_scope}"
        )
        lines.append("")
        lines.append("| Variant | Runs | Gold Unknown | Pred. Unknown | Verifier-accepted Unknown | Over-Commit |")
        lines.append("|---------|------|--------------|---------------|---------------------------|-------------|")
        for row in sorted(grouped[key], key=lambda item: VARIANT_ORDER.get(item["variant"], 99)):
            lines.append(
                "| "
                + " | ".join(
                    [
                        VARIANT_LABELS.get(row["variant"], row["variant"]),
                        str(row["runs"]),
                        str(row["gold_unknown"]),
                        f"{row['predicted_unknown_mean']:.1f} ± {row['predicted_unknown_std']:.1f}" if row["runs"] > 1 else f"{row['predicted_unknown_mean']:.1f}",
                        f"{row['faithful_unknown_mean']:.1f} ± {row['faithful_unknown_std']:.1f}" if row["runs"] > 1 else f"{row['faithful_unknown_mean']:.1f}",
                        f"{row['overcommit_mean']:.1f} ± {row['overcommit_std']:.1f}" if row["runs"] > 1 else f"{row['overcommit_mean']:.1f}",
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", default="results/summary_metrics.csv")
    parser.add_argument("--output-csv", default="results/unknown_behavior.csv")
    parser.add_argument("--aggregate-csv", default="results/unknown_behavior_agg.csv")
    parser.add_argument("--output-md", default="results/unknown_behavior.md")
    args = parser.parse_args()

    raw_summary_rows = load_summary_rows(Path(args.summary_csv))
    structured_rows = [summarize_unknown(row) for row in raw_summary_rows]
    aggregated_rows = aggregate(structured_rows)

    write_csv(
        Path(args.output_csv),
        structured_rows,
        [
            "study",
            "model_tag",
            "variant",
            "seed",
            "train_examples",
            "eval_config_name",
            "eval_split",
            "eval_max_examples",
            "eval_scope",
            "gold_unknown",
            "predicted_unknown",
            "faithful_unknown",
            "overcommit",
            "predicted_unknown_rate",
            "faithful_unknown_rate",
            "overcommit_rate",
        ],
    )
    write_csv(
        Path(args.aggregate_csv),
        aggregated_rows,
        [
            "study",
            "model_tag",
            "variant",
            "train_examples",
            "eval_config_name",
            "eval_split",
            "eval_max_examples",
            "eval_scope",
            "runs",
            "seeds",
            "gold_unknown",
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
        ],
    )

    markdown = build_markdown(aggregated_rows)
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
