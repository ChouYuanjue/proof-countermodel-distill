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
GENERATED = ROOT / "paper" / "generated"

VARIANT_LABELS = {
    "answer_only": "answer-only",
    "proof_only": "proof-only",
    "proco": "\\model{}",
}
DOMAIN_LABELS = {
    "depth-3ext-NatLang": "ID",
    "depth-5": "Depth-OOD",
    "support-deletion": "Support deletion",
}


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


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
        "eval_scope": metadata.get("eval_scope") or (
            "full" if config.get("max_examples") is None else f"subset_{int(config.get('max_examples'))}"
        ),
        "seed": int(config.get("seed", train_config.get("seed", 0))),
    }


def _progress(path: Path) -> dict:
    progress_path = path.with_name(path.stem + ".progress.json")
    if not progress_path.exists():
        return {}
    return json.loads(progress_path.read_text())


def _summarize_file(path: Path) -> dict | None:
    payload = json.loads(path.read_text())
    if "summary" not in payload or "predictions" not in payload:
        return None
    meta = _metadata(path, payload)
    if meta["variant"] not in {"answer_only", "proof_only", "proco"}:
        return None

    is_main = (
        meta["study"] == "maintrack"
        and meta["model_tag"] == "qwen7b"
        and meta["train_examples"] == 4096
        and meta["eval_config_name"] in {"depth-3ext-NatLang", "depth-5"}
        and meta["eval_scope"] == "subset_4000"
    )
    is_mutation = (
        meta["study"] == "mutation"
        and meta["model_tag"] == "qwen7b"
        and meta["train_examples"] == 4096
        and meta["eval_config_name"] == "support-deletion"
        and meta["eval_scope"] == "subset_4000"
    )
    if not (is_main or is_mutation):
        return None

    texts = [row.get("raw_output") or "" for row in payload["predictions"]]
    progress = _progress(path)
    return {
        **meta,
        "examples": len(texts),
        "avg_chars": mean(len(text) for text in texts),
        "avg_words": mean(_word_count(text) for text in texts),
        "examples_per_second": float(progress["examples_per_second"]) if progress else None,
    }


def _aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["study"], row["eval_config_name"], row["variant"])].append(row)
    out = []
    for (study, eval_config_name, variant), bucket in groups.items():
        row = {
            "study": study,
            "eval_config_name": eval_config_name,
            "variant": variant,
            "runs": len(bucket),
        }
        for metric in ["avg_chars", "avg_words"]:
            values = [item[metric] for item in bucket]
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        eps_values = [item["examples_per_second"] for item in bucket if item["examples_per_second"] is not None]
        row["examples_per_second_mean"] = mean(eps_values) if eps_values else None
        out.append(row)
    out.sort(key=lambda r: (r["study"], r["eval_config_name"], {"answer_only": 0, "proof_only": 1, "proco": 2}[r["variant"]]))
    return out


def _fmt(row: dict, metric: str) -> str:
    avg = row[f"{metric}_mean"]
    sd = row[f"{metric}_std"]
    return f"{avg:.1f} $\\pm$ {sd:.1f}" if row["runs"] > 1 else f"{avg:.1f}"


def _fmt_eps(row: dict) -> str:
    value = row["examples_per_second_mean"]
    return "--" if value is None else f"{value:.2f}"


def _write(rows: list[dict]) -> None:
    RESULTS.mkdir(exist_ok=True)
    GENERATED.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "decoding_cost_agg.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    main_rows = [
        row
        for row in rows
        if row["study"] == "maintrack" and row["eval_config_name"] == "depth-3ext-NatLang"
    ]
    mutation_rows = [row for row in rows if row["study"] == "mutation"]
    lines = [
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Setting & Variant & Output words & Output chars & Ex./sec. \\\\",
        "\\midrule",
    ]
    for row in main_rows + mutation_rows:
        lines.append(
            " & ".join(
                [
                    DOMAIN_LABELS.get(row["eval_config_name"], row["eval_config_name"]),
                    VARIANT_LABELS[row["variant"]],
                    _fmt(row, "avg_words"),
                    _fmt(row, "avg_chars"),
                    _fmt_eps(row),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (GENERATED / "decoding_cost_table.tex").write_text("\n".join(lines))


def main() -> None:
    rows = [row for path in RESULTS.glob("*.json") if (row := _summarize_file(path)) is not None]
    if not rows:
        raise SystemExit("no matching result files")
    _write(_aggregate(rows))


if __name__ == "__main__":
    main()
