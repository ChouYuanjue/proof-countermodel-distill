#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from pocd.dataset import build_records


def _infer_config(path: Path, payload: dict) -> tuple[str, str, str, int]:
    config = payload.get("config", {})
    summary = payload.get("summary", {})
    variant = summary.get("variant") or config.get("variant")
    split = summary.get("split") or config.get("split") or "dev"
    seed = config.get("seed", 0)
    config_name = summary.get("config_name") or config.get("config_name")
    if config_name:
        return config_name, split, variant, seed

    if path.stem.endswith("_depth5_dev"):
        return "depth-5", split, variant, seed
    return "depth-3ext-NatLang", split, variant, seed


def _pct(value: float) -> str:
    return f"{value * 100:.1f}"


def _render_case(title: str, case: dict, record: dict) -> list[str]:
    lines = [f"### {title}", ""]
    lines.append(f"- Example: `{case['example_id']}` / `{case['question_id']}`")
    lines.append(f"- Gold: `{case['gold_label']}`")
    lines.append(f"- Predicted: `{case['pred_label']}`")
    lines.append(f"- Faithful: `{case['faithful']}`")
    if case.get("parsed", {}).get("mode"):
        lines.append(f"- Mode: `{case['parsed']['mode']}`")
    lines.append(f"- Question: {record['question_text']}")
    lines.append(f"- Gold witness: {record['gold_witness']}")
    lines.append("- Model output:")
    lines.append("")
    lines.append("```text")
    lines.append(case["raw_output"].strip())
    lines.append("```")
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--max-cases", type=int, default=3)
    args = parser.parse_args()

    input_path = Path(args.input_path)
    payload = json.loads(input_path.read_text())
    predictions = payload["predictions"]
    summary = payload["summary"]

    config_name, split, variant, seed = _infer_config(input_path, payload)
    records = build_records(
        config_name=config_name,
        split=split,
        variant=variant,
        max_examples=len(predictions),
        seed=seed,
    )
    record_map = {(record["example_id"], record["question_id"]): record for record in records}

    by_gold: dict[str, list[dict]] = defaultdict(list)
    unknown_pred_labels: Counter[str] = Counter()
    unknown_modes: Counter[str] = Counter()

    for prediction in predictions:
        by_gold[prediction["gold_label"]].append(prediction)
        if prediction["gold_label"] == "Unknown":
            unknown_pred_labels[prediction["pred_label"]] += 1
            unknown_modes[prediction.get("parsed", {}).get("mode") or "NONE"] += 1

    lines = ["# Prediction Analysis", ""]
    lines.append(f"- Variant: `{summary['variant']}`")
    lines.append(f"- Dataset: `{config_name}` / `{split}`")
    lines.append(f"- Examples: `{summary['examples']}`")
    lines.append(
        f"- Accuracy / Faithfulness / Joint: `{_pct(summary['accuracy'])}` / `{_pct(summary['faithfulness_rate'])}` / `{_pct(summary['joint_accuracy'])}`"
    )
    lines.append("")

    lines.append("## By Gold Label")
    lines.append("")
    lines.append("| Gold label | Count | Accuracy | Faithfulness | Joint |")
    lines.append("|------------|-------|----------|--------------|-------|")
    for gold_label in ["True", "False", "Unknown"]:
        bucket = by_gold.get(gold_label, [])
        count = len(bucket)
        accuracy = sum(item["pred_label"] == gold_label for item in bucket) / max(1, count)
        faithfulness = sum(item["faithful"] for item in bucket) / max(1, count)
        joint = sum((item["pred_label"] == gold_label) and item["faithful"] for item in bucket) / max(1, count)
        lines.append(f"| {gold_label} | {count} | {_pct(accuracy)} | {_pct(faithfulness)} | {_pct(joint)} |")
    lines.append("")

    lines.append("## Unknown Behavior")
    lines.append("")
    lines.append("### Predicted Labels On Gold Unknown")
    lines.append("")
    for label, count in sorted(unknown_pred_labels.items()):
        lines.append(f"- `{label}`: {count}")
    lines.append("")
    lines.append("### Predicted Modes On Gold Unknown")
    lines.append("")
    for mode, count in sorted(unknown_modes.items()):
        lines.append(f"- `{mode}`: {count}")
    lines.append("")

    faithful_unknowns = [item for item in predictions if item["gold_label"] == "Unknown" and item["faithful"]]
    overcommits = [item for item in predictions if item["gold_label"] == "Unknown" and item["pred_label"] != "Unknown"]

    lines.append(f"- Faithful unknown explanations: `{len(faithful_unknowns)}`")
    lines.append(f"- Unknown over-commit errors: `{len(overcommits)}`")
    lines.append("")

    for index, case in enumerate(faithful_unknowns[: args.max_cases], start=1):
        record = record_map[(case["example_id"], case["question_id"])]
        lines.extend(_render_case(f"Faithful Unknown {index}", case, record))

    for index, case in enumerate(overcommits[: args.max_cases], start=1):
        record = record_map[(case["example_id"], case["question_id"])]
        lines.extend(_render_case(f"Unknown Over-Commit {index}", case, record))

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(output_path)


if __name__ == "__main__":
    main()
