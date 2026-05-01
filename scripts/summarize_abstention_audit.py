#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER_GENERATED = ROOT / "paper" / "generated"

VARIANT_LABELS = {
    "proof_only": "proof-only",
    "proco": "\\model{}",
}
DATASET_LABELS = {
    "depth-3ext-NatLang": "ID",
    "depth-5": "Depth-OOD",
    "support-deletion": "Support deletion",
}


def _norm(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _subtype(tag: object) -> str:
    tag_text = _norm(tag)
    if tag_text == "fail[no_rule]":
        return "no_rule"
    if re.fullmatch(r"fail\[rule\d+\]", tag_text):
        return "rule"
    if tag_text.startswith("fail["):
        return "other_fail"
    return "none"


def _missing_count(witness: object) -> int:
    text = str(witness or "")
    if "missing support:" not in text:
        return 0
    tail = text.split("missing support:", 1)[1].strip()
    if not tail:
        return 0
    return len([part for part in tail.split(";") if part.strip()])


def _metadata(path: Path, payload: dict) -> dict:
    summary = payload["summary"]
    config = payload.get("config", {})
    metadata = payload.get("metadata", {})
    train_metadata = metadata.get("train_metadata") or {}
    train_config = train_metadata.get("config") or {}
    train_examples = metadata.get("train_records") or train_metadata.get("train_records")
    if train_examples is None:
        train_examples = train_config.get("train_max_examples")
    return {
        "study": metadata.get("study_tag") or config.get("study_tag") or path.stem.split("_", 1)[0],
        "model_tag": metadata.get("model_tag") or config.get("model_tag") or "base",
        "variant": summary.get("variant") or config.get("variant"),
        "train_examples": int(train_examples) if train_examples not in {None, "", "None"} else None,
        "eval_config_name": summary.get("config_name") or config.get("config_name"),
        "eval_split": summary.get("split") or config.get("split") or "test",
        "eval_scope": metadata.get("eval_scope") or (
            "full" if config.get("max_examples") is None else f"subset_{int(config.get('max_examples'))}"
        ),
        "seed": int(config.get("seed", train_config.get("seed", 0))),
    }


def _summarize(path: Path) -> dict | None:
    payload = json.loads(path.read_text())
    if "summary" not in payload or "predictions" not in payload:
        return None
    meta = _metadata(path, payload)
    if meta["variant"] not in {"proof_only", "proco"}:
        return None
    if not (
        meta["study"] == "maintrack"
        and meta["model_tag"] == "qwen7b"
        and meta["train_examples"] == 4096
        and meta["eval_config_name"] in {"depth-3ext-NatLang", "depth-5"}
        and meta["eval_scope"] == "subset_4000"
    ):
        return None

    gold_unknown = [row for row in payload["predictions"] if row["gold_label"] == "Unknown"]
    pred_unknown = [row for row in gold_unknown if row["pred_label"] == "Unknown"]
    faithful = [row for row in pred_unknown if bool(row.get("faithful"))]
    strict = [
        row
        for row in pred_unknown
        if _norm((row.get("parsed") or {}).get("mode")) == "abstain"
        and _norm((row.get("parsed") or {}).get("chain_text")) == _norm(row.get("gold_failure_tag"))
        and _norm((row.get("parsed") or {}).get("witness")) == _norm(row.get("gold_failure_witness"))
    ]
    gold_rule = [row for row in gold_unknown if _subtype(row.get("gold_failure_tag")) == "rule"]
    gold_no_rule = [row for row in gold_unknown if _subtype(row.get("gold_failure_tag")) == "no_rule"]
    strict_rule = [row for row in strict if _subtype(row.get("gold_failure_tag")) == "rule"]
    strict_no_rule = [row for row in strict if _subtype(row.get("gold_failure_tag")) == "no_rule"]

    gold_lengths = [_missing_count(row.get("gold_failure_witness")) for row in gold_unknown]
    strict_lengths = [_missing_count((row.get("parsed") or {}).get("witness")) for row in strict]

    return {
        **meta,
        "gold_unknown": len(gold_unknown),
        "gold_rule": len(gold_rule),
        "gold_no_rule": len(gold_no_rule),
        "pred_unknown": len(pred_unknown),
        "faithful": len(faithful),
        "strict": len(strict),
        "strict_rule": len(strict_rule),
        "strict_no_rule": len(strict_no_rule),
        "faithful_rate": len(faithful) / max(1, len(gold_unknown)),
        "strict_rate": len(strict) / max(1, len(gold_unknown)),
        "strict_rule_rate": len(strict_rule) / max(1, len(gold_rule)),
        "strict_no_rule_rate": len(strict_no_rule) / max(1, len(gold_no_rule)),
        "gold_len": mean(gold_lengths) if gold_lengths else 0.0,
        "strict_len": mean(strict_lengths) if strict_lengths else 0.0,
    }


def _aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (row["variant"], row["eval_config_name"])
        groups[key].append(row)
    metrics = [
        "gold_unknown",
        "gold_rule",
        "gold_no_rule",
        "pred_unknown",
        "faithful",
        "strict",
        "strict_rule",
        "strict_no_rule",
        "faithful_rate",
        "strict_rate",
        "strict_rule_rate",
        "strict_no_rule_rate",
        "gold_len",
        "strict_len",
    ]
    out = []
    for (variant, eval_config_name), bucket in groups.items():
        row = {
            "variant": variant,
            "eval_config_name": eval_config_name,
            "runs": len(bucket),
            "seeds": ",".join(str(item["seed"]) for item in sorted(bucket, key=lambda x: x["seed"])),
        }
        for metric in metrics:
            values = [item[metric] for item in bucket]
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        out.append(row)
    out.sort(key=lambda r: (r["eval_config_name"], 0 if r["variant"] == "proof_only" else 1))
    return out


def _fmt_count(row: dict, metric: str) -> str:
    avg = row[f"{metric}_mean"]
    sd = row[f"{metric}_std"]
    return f"{avg:.1f} $\\pm$ {sd:.1f}" if row["runs"] > 1 else f"{avg:.1f}"


def _fmt_pct(row: dict, metric: str) -> str:
    avg = row[f"{metric}_mean"] * 100
    sd = row[f"{metric}_std"] * 100
    return f"{avg:.1f} $\\pm$ {sd:.1f}" if row["runs"] > 1 else f"{avg:.1f}"


def _write_outputs(rows: list[dict]) -> None:
    RESULTS.mkdir(exist_ok=True)
    PAPER_GENERATED.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS / "abstention_audit_agg.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    tex_lines = [
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Domain & Variant & Verifier-accepted & Strict & Rule strict & No-rule strict \\\\",
        "\\midrule",
    ]
    current = None
    for row in rows:
        domain = DATASET_LABELS.get(row["eval_config_name"], row["eval_config_name"])
        if current is not None and domain != current:
            tex_lines.append("\\midrule")
        current = domain
        tex_lines.append(
            " & ".join(
                [
                    domain,
                    VARIANT_LABELS.get(row["variant"], row["variant"]),
                    _fmt_pct(row, "faithful_rate"),
                    _fmt_pct(row, "strict_rate"),
                    _fmt_pct(row, "strict_rule_rate"),
                    _fmt_pct(row, "strict_no_rule_rate"),
                ]
            )
            + " \\\\"
        )
    tex_lines.extend(["\\bottomrule", "\\end{tabular}"])
    (PAPER_GENERATED / "abstention_audit_table.tex").write_text("\n".join(tex_lines))


def main() -> None:
    rows = [row for path in RESULTS.glob("*.json") if (row := _summarize(path)) is not None]
    if not rows:
        raise SystemExit("no matching result files")
    _write_outputs(_aggregate(rows))


if __name__ == "__main__":
    main()
