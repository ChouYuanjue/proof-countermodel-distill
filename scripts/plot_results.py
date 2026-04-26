#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


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
VARIANT_COLORS = {
    "answer_only": "#1f77b4",
    "proof_only": "#ff7f0e",
    "proco_chain": "#9467bd",
    "proco_witness": "#8c564b",
    "proco": "#2ca02c",
}
EVAL_GROUP_LABELS = {
    "in_domain": "In-domain",
    "depth_ood": "Depth-OOD",
    "template_transfer": "Template Transfer",
    "domain_transfer": "Domain Transfer",
}
OVERVIEW_METRICS = [
    ("accuracy_mean", "accuracy_std", "Accuracy"),
    ("unknown_f1_mean", "unknown_f1_std", "Unknown F1"),
    ("faithfulness_rate_mean", "faithfulness_rate_std", "Faithfulness"),
    ("joint_accuracy_mean", "joint_accuracy_std", "Joint"),
]


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


def _scope_sort_key(scope: str) -> tuple[int, int]:
    if scope.startswith("subset_"):
        return (0, _parse_int(scope.split("_", 1)[1]) or 0)
    return (1, math.inf)


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = dict(row)
            parsed["train_examples"] = _parse_int(parsed.get("train_examples"))
            parsed["eval_max_examples"] = _parse_int(parsed.get("eval_max_examples"))
            parsed["eval_scope"] = _parse_scope(parsed)
            parsed["runs"] = int(parsed["runs"])
            parsed["examples"] = int(float(parsed["examples"]))
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


def _save_figure(fig, output_base: Path) -> list[Path]:
    outputs: list[Path] = []
    for suffix in ["png", "pdf"]:
        output_path = output_base.with_suffix(f".{suffix}")
        fig.savefig(output_path, bbox_inches="tight", dpi=200)
        outputs.append(output_path)
    plt.close(fig)
    return outputs


def _plot_overviews(rows: list[dict], output_dir: Path) -> list[Path]:
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

    outputs: list[Path] = []
    for key, group_rows in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            math.inf if item[0][2] is None else item[0][2],
            item[0][3],
            item[0][4],
            _scope_sort_key(item[0][5]),
        ),
    ):
        study, model_tag, train_examples, eval_config_name, eval_split, eval_scope = key
        group_rows = sorted(group_rows, key=lambda row: VARIANT_ORDER.get(row["variant"], 99))
        fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
        width = 0.15
        x_positions = list(range(len(OVERVIEW_METRICS)))

        for index, row in enumerate(group_rows):
            offset = (index - (len(group_rows) - 1) / 2.0) * width
            values = [row[mean_key] * 100.0 for mean_key, _, _ in OVERVIEW_METRICS]
            errors = [row[std_key] * 100.0 for _, std_key, _ in OVERVIEW_METRICS]
            xs = [position + offset for position in x_positions]
            ax.bar(
                xs,
                values,
                width=width,
                yerr=errors if row["runs"] > 1 else None,
                capsize=3 if row["runs"] > 1 else 0,
                label=VARIANT_LABELS.get(row["variant"], row["variant"]),
                color=VARIANT_COLORS.get(row["variant"], "#777777"),
            )

        ax.set_xticks(x_positions, [label for _, _, label in OVERVIEW_METRICS], rotation=18, ha="right")
        ax.set_ylim(0, 100)
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.25)
        eval_group = EVAL_GROUP_LABELS.get(group_rows[0]["eval_group"], group_rows[0]["eval_group"])
        train_display = "full" if train_examples is None else str(train_examples)
        ax.set_title(f"{study} / {model_tag} / train={train_display} / {eval_group} / scope={eval_scope}")
        ax.legend(frameon=False, ncol=min(3, len(group_rows)))

        stem = f"{study}_{model_tag}_train{train_display}_{eval_config_name}_{eval_split}_{eval_scope}_overview"
        outputs.extend(_save_figure(fig, output_dir / stem))
    return outputs


def _plot_scaling(rows: list[dict], output_dir: Path) -> list[Path]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if row["variant"] not in {"answer_only", "proof_only", "proco"}:
            continue
        if row["eval_scope"] != "subset_4000":
            continue
        grouped[(row["study"], row["model_tag"], row["eval_config_name"], row["eval_split"])].append(row)

    outputs: list[Path] = []
    for (study, model_tag, eval_config_name, eval_split), group_rows in sorted(grouped.items()):
        sizes = sorted(
            {row["train_examples"] for row in group_rows if row["train_examples"] is not None}
        )
        if len(sizes) < 2:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
        metrics = [
            ("accuracy_mean", "accuracy_std", "Accuracy"),
            ("joint_accuracy_mean", "joint_accuracy_std", "Joint Accuracy"),
        ]
        for axis, (mean_key, std_key, ylabel) in zip(axes, metrics):
            for variant in ["answer_only", "proof_only", "proco"]:
                variant_rows = sorted(
                    [row for row in group_rows if row["variant"] == variant and row["train_examples"] is not None],
                    key=lambda row: row["train_examples"],
                )
                if len(variant_rows) < 2:
                    continue
                xs = [row["train_examples"] for row in variant_rows]
                ys = [row[mean_key] * 100.0 for row in variant_rows]
                yerr = [row[std_key] * 100.0 for row in variant_rows]
                axis.errorbar(
                    xs,
                    ys,
                    yerr=yerr,
                    marker="o",
                    linewidth=2,
                    capsize=3,
                    color=VARIANT_COLORS.get(variant, "#777777"),
                    label=VARIANT_LABELS.get(variant, variant),
                )
            axis.set_xscale("log", base=2)
            axis.set_xlabel("Train Questions")
            axis.set_ylabel(ylabel)
            axis.set_ylim(0, 100)
            axis.grid(alpha=0.25)

        eval_label = EVAL_GROUP_LABELS.get(group_rows[0]["eval_group"], group_rows[0]["eval_group"])
        fig.suptitle(f"{study} / {model_tag} / {eval_config_name}/{eval_split} / {eval_label}")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, frameon=False, loc="upper center", ncol=len(labels))

        stem = f"{study}_{model_tag}_{eval_config_name}_{eval_split}_scaling"
        outputs.extend(_save_figure(fig, output_dir / stem))
    return outputs


def _plot_ablations(rows: list[dict], output_dir: Path) -> list[Path]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if row["variant"] not in {"proof_only", "proco_chain", "proco_witness", "proco"}:
            continue
        if row["eval_scope"] != "subset_4000":
            continue
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

    outputs: list[Path] = []
    for (study, model_tag, train_examples, eval_config_name, eval_split, eval_scope), group_rows in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            math.inf if item[0][2] is None else item[0][2],
            item[0][3],
            item[0][4],
            _scope_sort_key(item[0][5]),
        ),
    ):
        variants = {row["variant"] for row in group_rows}
        if not {"proco_chain", "proco_witness"}.intersection(variants):
            continue

        group_rows = sorted(group_rows, key=lambda row: VARIANT_ORDER.get(row["variant"], 99))
        fig, ax = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)
        x_positions = list(range(3))
        metric_triplet = [
            ("accuracy_mean", "accuracy_std", "Accuracy"),
            ("faithfulness_rate_mean", "faithfulness_rate_std", "Faithfulness"),
            ("joint_accuracy_mean", "joint_accuracy_std", "Joint"),
        ]
        width = 0.16
        for index, row in enumerate(group_rows):
            offset = (index - (len(group_rows) - 1) / 2.0) * width
            xs = [position + offset for position in x_positions]
            ys = [row[mean_key] * 100.0 for mean_key, _, _ in metric_triplet]
            yerr = [row[std_key] * 100.0 for _, std_key, _ in metric_triplet]
            ax.bar(
                xs,
                ys,
                width=width,
                yerr=yerr if row["runs"] > 1 else None,
                capsize=3 if row["runs"] > 1 else 0,
                color=VARIANT_COLORS.get(row["variant"], "#777777"),
                label=VARIANT_LABELS.get(row["variant"], row["variant"]),
            )

        ax.set_xticks(x_positions, [label for _, _, label in metric_triplet])
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.25)
        train_display = "full" if train_examples is None else str(train_examples)
        ax.set_title(
            f"{study} / {model_tag} / train={train_display} / {eval_config_name}/{eval_split} / scope={eval_scope}"
        )
        ax.legend(frameon=False, ncol=min(4, len(group_rows)))

        stem = f"{study}_{model_tag}_train{train_display}_{eval_config_name}_{eval_split}_{eval_scope}_ablations"
        outputs.extend(_save_figure(fig, output_dir / stem))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", default="results/summary_metrics_agg.csv")
    parser.add_argument("--output-dir", default="paper/figures")
    args = parser.parse_args()

    rows = _load_rows(Path(args.summary_csv))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    outputs.extend(_plot_overviews(rows, output_dir))
    outputs.extend(_plot_scaling(rows, output_dir))
    outputs.extend(_plot_ablations(rows, output_dir))

    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
