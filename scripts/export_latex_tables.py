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
    "proco_no_refute": 4,
    "proco": 5,
}
VARIANT_LABELS = {
    "answer_only": "answer-only",
    "proof_only": "proof-only",
    "proco_chain": "ProCo-chain",
    "proco_witness": "ProCo-witness",
    "proco_no_refute": "ProCo-no-refute",
    "proco": "ProCo",
}
DATASET_LABELS = {
    "depth-3ext-NatLang": "ID",
    "depth-5": "Depth-OOD",
    "NatLang": "NatLang Transfer",
    "depth-3": "Depth-3 Transfer",
    "birds-electricity": "Birds-Electricity",
    "support-deletion": "Support-Deletion",
}


def _fmt_pct_value(mean_value: float, std_value: float, runs: int) -> str:
    if runs > 1:
        return f"{mean_value * 100:.1f} $\\pm$ {std_value * 100:.1f}"
    return f"{mean_value * 100:.1f}"


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


def load_per_class_rows(path: Path) -> list[dict]:
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
                "precision_mean",
                "precision_std",
                "recall_mean",
                "recall_std",
                "f1_mean",
                "f1_std",
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
                "predicted_unknown_rate_mean",
                "predicted_unknown_rate_std",
                "faithful_unknown_rate_mean",
                "faithful_unknown_rate_std",
                "overcommit_rate_mean",
                "overcommit_rate_std",
            ]:
                if key in parsed:
                    parsed[key] = float(parsed[key])
            rows.append(parsed)
    return rows


def _fmt(row: dict, key: str) -> str:
    mean_key = f"{key}_mean"
    std_key = f"{key}_std"
    if row["runs"] > 1:
        return f"{row[mean_key] * 100:.1f} $\\pm$ {row[std_key] * 100:.1f}"
    return f"{row[mean_key] * 100:.1f}"


def _summary_index(rows: list[dict]) -> dict[tuple, dict]:
    indexed: dict[tuple, dict] = {}
    for row in rows:
        key = (
            row["study"],
            row["variant"],
            row["train_examples"],
            row["eval_config_name"],
            row["eval_split"],
            row["eval_scope"],
        )
        indexed[key] = row
    return indexed


def _per_class_index(rows: list[dict]) -> dict[tuple, dict]:
    indexed: dict[tuple, dict] = {}
    for row in rows:
        key = (
            row["study"],
            row["variant"],
            row["train_examples"],
            row["eval_config_name"],
            row["eval_split"],
            row["eval_scope"],
            row["label"],
        )
        indexed[key] = row
    return indexed


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


def _is_qwen_main(row: dict) -> bool:
    return row.get("model_tag") == "qwen7b"


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
        and _is_qwen_main(row)
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
        and _is_qwen_main(row)
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
        and _is_qwen_main(row)
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
        and _is_qwen_main(row)
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


def build_support_deletion_table(rows: list[dict]) -> str:
    picked = [
        row
        for row in rows
        if row["study"] == "mutation"
        and row["model_tag"] == "qwen7b"
        and row["train_examples"] == 4096
        and row["eval_config_name"] == "support-deletion"
        and row["eval_split"] == "test"
        and row["eval_scope"] == "subset_4000"
        and row["variant"] in {"answer_only", "proof_only", "proco"}
    ]
    picked.sort(key=lambda row: VARIANT_ORDER.get(row["variant"], 99))

    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Variant & Runs & Pred. Unknown & Faithful Unknown & Joint \\\\",
        "\\midrule",
    ]
    for row in picked:
        pred_unknown = f"{row['predicted_unknown_mean']:.1f}" if row["runs"] == 1 else f"{row['predicted_unknown_mean']:.1f} $\\pm$ {row['predicted_unknown_std']:.1f}"
        faithful_unknown = f"{row['faithful_unknown_mean']:.1f}" if row["runs"] == 1 else f"{row['faithful_unknown_mean']:.1f} $\\pm$ {row['faithful_unknown_std']:.1f}"
        joint = f"{row['faithful_unknown_rate_mean'] * 100:.1f}" if row["runs"] == 1 else f"{row['faithful_unknown_rate_mean'] * 100:.1f} $\\pm$ {row['faithful_unknown_rate_std'] * 100:.1f}"
        lines.append(
            " & ".join(
                [
                    VARIANT_LABELS.get(row["variant"], row["variant"]),
                    str(row["runs"]),
                    pred_unknown,
                    faithful_unknown,
                    joint,
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def build_per_class_table(rows: list[dict]) -> str:
    picked = [
        row
        for row in rows
        if row["study"] == "maintrack"
        and _is_qwen_main(row)
        and row["train_examples"] == 4096
        and row["eval_config_name"] in {"depth-3ext-NatLang", "depth-5"}
        and row["eval_split"] == "test"
        and row["eval_scope"] == "subset_4000"
        and row["variant"] in {"answer_only", "proof_only", "proco"}
        and row["label"] in {"True", "False", "Unknown"}
    ]
    picked.sort(
        key=lambda row: (
            row["eval_config_name"],
            VARIANT_ORDER.get(row["variant"], 99),
            {"True": 0, "False": 1, "Unknown": 2}[row["label"]],
        )
    )
    indexed = _per_class_index(picked)

    lines = [
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Domain & Variant & True F1 & False F1 & Unknown F1 \\\\",
        "\\midrule",
    ]
    for dataset in ["depth-3ext-NatLang", "depth-5"]:
        if dataset != "depth-3ext-NatLang":
            lines.append("\\midrule")
        for variant in ["answer_only", "proof_only", "proco"]:
            row_true = indexed[( "maintrack", variant, 4096, dataset, "test", "subset_4000", "True")]
            row_false = indexed[( "maintrack", variant, 4096, dataset, "test", "subset_4000", "False")]
            row_unknown = indexed[( "maintrack", variant, 4096, dataset, "test", "subset_4000", "Unknown")]
            lines.append(
                " & ".join(
                    [
                        DATASET_LABELS[dataset],
                        VARIANT_LABELS[variant],
                        _fmt_pct_value(row_true["f1_mean"], row_true["f1_std"], row_true["runs"]),
                        _fmt_pct_value(row_false["f1_mean"], row_false["f1_std"], row_false["runs"]),
                        _fmt_pct_value(row_unknown["f1_mean"], row_unknown["f1_std"], row_unknown["runs"]),
                    ]
                )
                + " \\\\"
            )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def build_claims_table(summary_rows: list[dict], unknown_rows: list[dict]) -> str:
    summary_rows = [row for row in summary_rows if _is_qwen_main(row)]
    unknown_rows = [row for row in unknown_rows if _is_qwen_main(row)]
    summary = _summary_index(summary_rows)
    unknown = _summary_index(unknown_rows)

    id_proof = summary[("maintrack", "proof_only", 4096, "depth-3ext-NatLang", "test", "subset_4000")]
    id_proco = summary[("maintrack", "proco", 4096, "depth-3ext-NatLang", "test", "subset_4000")]
    ood_proof = summary[("maintrack", "proof_only", 4096, "depth-5", "test", "subset_4000")]
    ood_proco = summary[("maintrack", "proco", 4096, "depth-5", "test", "subset_4000")]
    id_unknown_proof = unknown[("maintrack", "proof_only", 4096, "depth-3ext-NatLang", "test", "subset_4000")]
    id_unknown_proco = unknown[("maintrack", "proco", 4096, "depth-3ext-NatLang", "test", "subset_4000")]
    ood_unknown_proof = unknown[("maintrack", "proof_only", 4096, "depth-5", "test", "subset_4000")]
    ood_unknown_proco = unknown[("maintrack", "proco", 4096, "depth-5", "test", "subset_4000")]
    full_id_answer = summary[("maintrack", "answer_only", 112062, "depth-3ext-NatLang", "test", "subset_4000")]
    full_id_proco = summary[("maintrack", "proco", 112062, "depth-3ext-NatLang", "test", "subset_4000")]
    full_ood_answer = summary[("maintrack", "answer_only", 112062, "depth-5", "test", "subset_4000")]
    full_ood_proco = summary[("maintrack", "proco", 112062, "depth-5", "test", "subset_4000")]

    rows = [
        (
            "At 7B and 4k train, ProCo greatly improves in-domain joint accuracy over proof-only.",
            f"{_fmt(id_proco, 'joint_accuracy')} vs {_fmt(id_proof, 'joint_accuracy')} (Table~\\ref{{tab:seeded-results}})",
            "Qwen2.5-7B, depth-3ext-NatLang/test, subset 4000, 3 seeds.",
        ),
        (
            "At 7B and 4k train, the same gain remains on depth-OOD.",
            f"{_fmt(ood_proco, 'joint_accuracy')} vs {_fmt(ood_proof, 'joint_accuracy')} (Table~\\ref{{tab:seeded-results}})",
            "Qwen2.5-7B, depth-5/test, subset 4000, 3 seeds.",
        ),
        (
            "The unknown-case gain comes from explanation quality rather than abstention rate.",
            (
                f"ID predicted unknown: {id_unknown_proco['predicted_unknown_mean']:.1f} vs {id_unknown_proof['predicted_unknown_mean']:.1f}; "
                f"ID faithful unknown: {id_unknown_proco['faithful_unknown_mean']:.1f} vs {id_unknown_proof['faithful_unknown_mean']:.1f}. "
                f"Depth-OOD faithful unknown: {ood_unknown_proco['faithful_unknown_mean']:.1f} vs {ood_unknown_proof['faithful_unknown_mean']:.1f} "
                f"(Table~\\ref{{tab:abstention-evidence}})"
            ),
            "Gold-unknown slices on the fixed 7B test subset.",
        ),
        (
            "With full training, the in-domain raw-accuracy gap is nearly gone.",
            f"ProCo acc. {_fmt(full_id_proco, 'accuracy')} vs answer-only {_fmt(full_id_answer, 'accuracy')}; ProCo joint {_fmt(full_id_proco, 'joint_accuracy')} (Table~\\ref{{tab:scaling-results}})",
            "Qwen2.5-7B, depth-3ext-NatLang/test, subset 4000, train=112,062.",
        ),
        (
            "With full training, ProCo slightly surpasses answer-only on depth-5 subset accuracy.",
            f"ProCo acc. {_fmt(full_ood_proco, 'accuracy')} vs answer-only {_fmt(full_ood_answer, 'accuracy')} (Table~\\ref{{tab:scaling-results}})",
            "Qwen2.5-7B, depth-5/test, subset 4000, train=112,062.",
        ),
    ]

    lines = [
        "\\begin{tabular}{p{0.36\\linewidth}p{0.34\\linewidth}p{0.22\\linewidth}}",
        "\\toprule",
        "Claim & Supporting Evidence & Scope \\\\",
        "\\midrule",
    ]
    for claim, evidence, scope in rows:
        lines.append(f"{claim} & {evidence} & {scope} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def build_backbone_table(rows: list[dict]) -> str:
    picked = [
        row
        for row in rows
        if row["study"] == "maintrack"
        and row["model_tag"] in {"qwen7b", "mistral7b"}
        and row["train_examples"] == 4096
        and row["eval_config_name"] in {"depth-3ext-NatLang", "depth-5"}
        and row["eval_split"] == "test"
        and row["eval_scope"] == "subset_4000"
        and row["variant"] in {"answer_only", "proof_only", "proco"}
    ]
    picked.sort(
        key=lambda row: (
            {"qwen7b": 0, "mistral7b": 1}.get(row["model_tag"], 99),
            row["eval_config_name"],
            VARIANT_ORDER.get(row["variant"], 99),
        )
    )

    model_labels = {"qwen7b": "Qwen2.5-7B", "mistral7b": "Mistral-7B"}
    lines = [
        "\\begin{tabular}{lllcccc}",
        "\\toprule",
        "Model & Domain & Variant & Runs & Acc. & Faithful & Joint \\\\",
        "\\midrule",
    ]
    current_model = None
    current_dataset = None
    for row in picked:
        if current_model is not None and current_model != row["model_tag"]:
            lines.append("\\midrule")
            current_dataset = None
        elif current_dataset is not None and current_dataset != row["eval_config_name"]:
            lines.append("\\cmidrule(lr){2-7}")
        current_model = row["model_tag"]
        current_dataset = row["eval_config_name"]
        lines.append(
            " & ".join(
                [
                    model_labels.get(row["model_tag"], row["model_tag"]),
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


def build_refute_ablation_table(summary_rows: list[dict], per_class_rows: list[dict]) -> str:
    summary = _summary_index(summary_rows)
    per_class = _per_class_index(per_class_rows)

    lines = [
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Domain & Variant & Acc. & False F1 & Unknown F1 & Faithful & Joint \\\\",
        "\\midrule",
    ]
    for dataset in ["depth-3ext-NatLang", "depth-5"]:
        if dataset != "depth-3ext-NatLang":
            lines.append("\\midrule")
        for variant in ["proof_only", "proco_no_refute", "proco"]:
            summary_row = summary[("ablation", variant, 4096, dataset, "test", "subset_4000")]
            false_row = per_class[("ablation", variant, 4096, dataset, "test", "subset_4000", "False")]
            unknown_row = per_class[("ablation", variant, 4096, dataset, "test", "subset_4000", "Unknown")]
            lines.append(
                " & ".join(
                    [
                        DATASET_LABELS[dataset],
                        VARIANT_LABELS[variant],
                        _fmt(summary_row, "accuracy"),
                        _fmt_pct_value(false_row["f1_mean"], false_row["f1_std"], false_row["runs"]),
                        _fmt_pct_value(unknown_row["f1_mean"], unknown_row["f1_std"], unknown_row["runs"]),
                        _fmt(summary_row, "faithfulness_rate"),
                        _fmt(summary_row, "joint_accuracy"),
                    ]
                )
                + " \\\\"
            )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def has_refute_ablation_data(summary_rows: list[dict], per_class_rows: list[dict]) -> bool:
    summary = _summary_index(summary_rows)
    per_class = _per_class_index(per_class_rows)

    required_summary_keys = [
        ("ablation", variant, 4096, dataset, "test", "subset_4000")
        for dataset in ["depth-3ext-NatLang", "depth-5"]
        for variant in ["proof_only", "proco_no_refute", "proco"]
    ]
    required_per_class_keys = [
        ("ablation", variant, 4096, dataset, "test", "subset_4000", label)
        for dataset in ["depth-3ext-NatLang", "depth-5"]
        for variant in ["proof_only", "proco_no_refute", "proco"]
        for label in ["False", "Unknown"]
    ]
    return all(key in summary for key in required_summary_keys) and all(
        key in per_class for key in required_per_class_keys
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", default="results/summary_metrics_agg.csv")
    parser.add_argument("--unknown-csv", default="results/unknown_behavior_agg.csv")
    parser.add_argument("--per-class-csv", default="results/per_class_metrics_agg.csv")
    parser.add_argument("--output-dir", default="paper/generated")
    args = parser.parse_args()

    rows = load_rows(Path(args.summary_csv))
    unknown_rows = load_unknown_rows(Path(args.unknown_csv))
    per_class_rows = load_per_class_rows(Path(args.per_class_csv))
    output_dir = Path(args.output_dir)

    outputs = {
        "seed_table.tex": build_seed_table(rows),
        "scaling_table.tex": build_scaling_table(rows),
        "ablation_table.tex": build_ablation_table(rows),
        "transfer_table.tex": build_transfer_table(rows),
        "unknown_table.tex": build_unknown_table(unknown_rows),
        "support_deletion_table.tex": build_support_deletion_table(unknown_rows),
        "per_class_table.tex": build_per_class_table(per_class_rows),
        "claims_table.tex": build_claims_table(rows, unknown_rows),
        "backbone_table.tex": build_backbone_table(rows),
    }
    if has_refute_ablation_data(rows, per_class_rows):
        outputs["refute_ablation_table.tex"] = build_refute_ablation_table(rows, per_class_rows)
    else:
        refute_path = output_dir / "refute_ablation_table.tex"
        if refute_path.exists():
            refute_path.unlink()
        print("skipping refute_ablation_table.tex: missing ProCo-no-refute aggregated rows")

    for name, text in outputs.items():
        path = output_dir / name
        _write(path, text)
        print(path)


if __name__ == "__main__":
    main()
