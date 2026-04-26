#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _parse_int(value: str | None) -> int | None:
    if value in {None, "", "None"}:
        return None
    return int(float(value))


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = dict(row)
            parsed["train_examples"] = _parse_int(parsed.get("train_examples"))
            parsed["eval_max_examples"] = _parse_int(parsed.get("eval_max_examples"))
            parsed["runs"] = int(parsed["runs"])
            for key, value in list(parsed.items()):
                if key.endswith("_mean") or key.endswith("_std"):
                    parsed[key] = float(value)
            rows.append(parsed)
    return rows


def index_rows(rows: list[dict]) -> dict[tuple, dict]:
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


def _check_equal(name: str, actual: float, expected: float, tol: float = 1e-9) -> dict:
    ok = abs(actual - expected) <= tol
    return {
        "claim": name,
        "actual": actual,
        "expected": expected,
        "status": "PASS" if ok else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", default="results/summary_metrics_agg.csv")
    parser.add_argument("--unknown-csv", default="results/unknown_behavior_agg.csv")
    parser.add_argument("--output-md", default="docs/PAPER_CLAIM_CHECK_2026-04-26.md")
    parser.add_argument("--output-json", default="docs/PAPER_CLAIM_CHECK_2026-04-26.json")
    args = parser.parse_args()

    summary = index_rows(load_rows(Path(args.summary_csv)))
    unknown = index_rows(load_rows(Path(args.unknown_csv)))

    checks = [
        _check_equal(
            "7B 4k ID proof-only joint",
            summary[("maintrack", "proof_only", 4096, "depth-3ext-NatLang", "test", "subset_4000")]["joint_accuracy_mean"],
            0.399,
            tol=5e-4,
        ),
        _check_equal(
            "7B 4k ID ProCo joint",
            summary[("maintrack", "proco", 4096, "depth-3ext-NatLang", "test", "subset_4000")]["joint_accuracy_mean"],
            0.750,
            tol=5e-4,
        ),
        _check_equal(
            "7B 4k Depth-OOD proof-only joint",
            summary[("maintrack", "proof_only", 4096, "depth-5", "test", "subset_4000")]["joint_accuracy_mean"],
            0.265,
            tol=5e-4,
        ),
        _check_equal(
            "7B 4k Depth-OOD ProCo joint",
            summary[("maintrack", "proco", 4096, "depth-5", "test", "subset_4000")]["joint_accuracy_mean"],
            0.550,
            tol=5e-4,
        ),
        _check_equal(
            "7B 4k ID faithful unknown for proof-only",
            unknown[("maintrack", "proof_only", 4096, "depth-3ext-NatLang", "test", "subset_4000")]["faithful_unknown_mean"],
            0.0,
        ),
        _check_equal(
            "7B 4k ID faithful unknown for ProCo",
            unknown[("maintrack", "proco", 4096, "depth-3ext-NatLang", "test", "subset_4000")]["faithful_unknown_mean"],
            1444.3,
            tol=0.11,
        ),
        _check_equal(
            "Full-train ID ProCo accuracy",
            summary[("maintrack", "proco", 112062, "depth-3ext-NatLang", "test", "subset_4000")]["accuracy_mean"],
            0.982,
            tol=5e-4,
        ),
        _check_equal(
            "Full-train ID ProCo joint",
            summary[("maintrack", "proco", 112062, "depth-3ext-NatLang", "test", "subset_4000")]["joint_accuracy_mean"],
            0.968,
            tol=5e-4,
        ),
        _check_equal(
            "Full-train Depth-OOD ProCo accuracy",
            summary[("maintrack", "proco", 112062, "depth-5", "test", "subset_4000")]["accuracy_mean"],
            0.938,
            tol=5e-4,
        ),
        _check_equal(
            "Full-train Depth-OOD ProCo joint",
            summary[("maintrack", "proco", 112062, "depth-5", "test", "subset_4000")]["joint_accuracy_mean"],
            0.862,
            tol=5e-4,
        ),
    ]

    overall = "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL"

    md_lines = [
        "# Paper Claim Check",
        "",
        "Scope: headline numbers used in the abstract, introduction, and main discussion.",
        "",
        f"Overall: **{overall}**",
        "",
        "| Claim | Actual | Expected | Status |",
        "|-------|--------|----------|--------|",
    ]
    for item in checks:
        md_lines.append(
            f"| {item['claim']} | {item['actual']} | {item['expected']} | {item['status']} |"
        )

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(md_lines) + "\n")

    output_json = Path(args.output_json)
    output_json.write_text(json.dumps({"overall": overall, "checks": checks}, indent=2) + "\n")

    print(output_md)
    print(output_json)
    print(overall)


if __name__ == "__main__":
    main()
