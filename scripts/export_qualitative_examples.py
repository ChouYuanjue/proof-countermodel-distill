#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
GENERATED = ROOT / "paper" / "generated"
sys.path.append(str(ROOT / "src"))

from pocd.dataset import build_records


def _norm(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _strict_match(pred: dict) -> bool:
    parsed = pred.get("parsed") or {}
    return (
        _norm(parsed.get("mode")) == "abstain"
        and _norm(parsed.get("chain_text")) == _norm(pred.get("gold_failure_tag"))
        and _norm(parsed.get("witness")) == _norm(pred.get("gold_failure_witness"))
    )


def _missing_leaves(text: str | None) -> list[str]:
    if not text or "missing support:" not in text.lower():
        return []
    tail = re.split(r"missing support:", text, flags=re.IGNORECASE, maxsplit=1)[1]
    return [part.strip() for part in tail.split(";") if part.strip()]


def _keywords(*texts: str | None) -> set[str]:
    stopwords = {
        "the",
        "and",
        "are",
        "for",
        "from",
        "that",
        "this",
        "with",
        "then",
        "they",
        "them",
        "does",
        "doesnt",
        "support",
        "missing",
        "blocked",
        "rule",
        "fact",
        "query",
        "label",
        "proof",
        "valid",
        "unknown",
        "witness",
        "mode",
        "chain",
    }
    tokens: set[str] = set()
    for text in texts:
        for token in re.findall(r"[A-Za-z]+", text or ""):
            normalized = token.lower()
            if len(normalized) >= 4 and normalized not in stopwords:
                tokens.add(normalized)
    return tokens


def _theory_excerpt(record: dict, focus_texts: list[str], max_lines: int = 4) -> str:
    keywords = _keywords(*focus_texts)
    lines = [
        line
        for line in (record["fact_lines"] + record["rule_lines"])
        if any(keyword in line.lower() for keyword in keywords)
    ]
    if not lines:
        lines = record["fact_lines"][:2] + record["rule_lines"][:2]
    return "\n".join(lines[:max_lines])


def _load_predictions(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    return payload["predictions"]


def _load_record_map(config_name: str, variant: str) -> dict[tuple[str, str], dict]:
    records = build_records(
        config_name=config_name,
        split="test",
        variant=variant,
        max_examples=4000,
        seed=0,
    )
    return {(record["example_id"], record["question_id"]): record for record in records}


def _pick_example(
    predictions: list[dict],
    record_map: dict[tuple[str, str], dict],
    predicate,
) -> dict:
    for pred in predictions:
        if predicate(pred):
            record = record_map[(pred["example_id"], pred["question_id"])]
            return {"prediction": pred, "record": record}
    raise SystemExit("could not find a qualitative example matching the requested condition")


def _render_example(title: str, item: dict) -> str:
    pred = item["prediction"]
    record = item["record"]
    parsed = pred.get("parsed") or {}
    strict = _strict_match(pred)
    lines = [
        f"\\item \\textbf{{{title}}}",
        "\\begin{itemize}",
        "\\item Theory excerpt:",
        "\\begin{verbatim}",
        _theory_excerpt(record, [pred["question_text"], pred.get("gold_failure_witness") or pred.get("gold_witness") or ""]),
        "\\end{verbatim}",
        f"\\item Query: {pred['question_text']}",
        f"\\item Gold label: \\texttt{{{pred['gold_label']}}}",
        f"\\item Gold witness: \\texttt{{{(pred.get('gold_failure_witness') or pred.get('gold_witness') or '').replace('_', '\\\\_')}}}",
        "\\item Model output:",
        "\\begin{verbatim}",
        pred["raw_output"].strip(),
        "\\end{verbatim}",
        f"\\item Verifier decision: verifier-accepted={bool(pred.get('faithful'))}; strict={strict}.",
        "\\item What it shows: "
    ]
    if title == "Clean FAIL[no\\_rule] success":
        lines[-1] += "A canonical no-rule abstention where tolerant and strict checks agree."
    elif title == "Rule witness accepted by tolerant verifier":
        lines[-1] += "The tolerant checker can accept a non-canonical rule witness when the canonical missing leaves are present, which is why strict audit is lower."
    elif title == "Proof-only generic abstention":
        lines[-1] += "Proof-only predicts Unknown here, but it emits no missing-support witness and remains unverifiable."
    else:
        lines[-1] += "Exact rule-tag matching is not enough to guarantee a strict canonical match when the witness text is non-canonical."
    lines.extend([
        "\\end{itemize}",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    proco_path = RESULTS / "maintrack_proco_qwen7b_train4096_s0_depth-3ext-NatLang_test.json"
    proof_path = RESULTS / "maintrack_proof_only_qwen7b_train4096_s0_depth-3ext-NatLang_test.json"
    proco_predictions = _load_predictions(proco_path)
    proof_predictions = _load_predictions(proof_path)
    proco_records = _load_record_map("depth-3ext-NatLang", "proco")
    proof_records = _load_record_map("depth-3ext-NatLang", "proof_only")

    no_rule = _pick_example(
        proco_predictions,
        proco_records,
        lambda pred: (
            pred["gold_label"] == "Unknown"
            and pred["pred_label"] == "Unknown"
            and bool(pred.get("faithful"))
            and (pred.get("gold_failure_tag") or "").lower() == "fail[no_rule]"
            and _strict_match(pred)
        ),
    )

    tolerant_rule = _pick_example(
        proco_predictions,
        proco_records,
        lambda pred: (
            pred["gold_label"] == "Unknown"
            and pred["pred_label"] == "Unknown"
            and bool(pred.get("faithful"))
            and (pred.get("gold_failure_tag") or "").lower().startswith("fail[rule")
            and _norm((pred.get("parsed") or {}).get("chain_text")) != _norm(pred.get("gold_failure_tag"))
            and all(
                _norm(missing) in _norm((pred.get("parsed") or {}).get("witness"))
                for missing in _missing_leaves(pred.get("gold_failure_witness"))
            )
        ),
    )

    proof_generic = _pick_example(
        proof_predictions,
        proof_records,
        lambda pred: (
            pred["gold_label"] == "Unknown"
            and pred["pred_label"] == "Unknown"
            and not bool(pred.get("faithful"))
            and _norm((pred.get("parsed") or {}).get("mode")) == "abstain"
            and _norm((pred.get("parsed") or {}).get("chain_text")) in {"", "none"}
        ),
    )

    strict_mismatch = _pick_example(
        proco_predictions,
        proco_records,
        lambda pred: (
            pred["gold_label"] == "Unknown"
            and pred["pred_label"] == "Unknown"
            and bool(pred.get("faithful"))
            and (pred.get("gold_failure_tag") or "").lower().startswith("fail[rule")
            and _norm((pred.get("parsed") or {}).get("chain_text")) == _norm(pred.get("gold_failure_tag"))
            and _norm((pred.get("parsed") or {}).get("witness")) != _norm(pred.get("gold_failure_witness"))
        ),
    )

    examples = [
        _render_example("Clean FAIL[no\\_rule] success", no_rule),
        _render_example("Rule witness accepted by tolerant verifier", tolerant_rule),
        _render_example("Proof-only generic abstention", proof_generic),
        _render_example("Strict canonical mismatch on FAIL[ruleX]", strict_mismatch),
    ]

    GENERATED.mkdir(parents=True, exist_ok=True)
    output_tex = GENERATED / "qualitative_examples.tex"
    output_tex.write_text(
        "\\begin{enumerate}\n"
        + "\n".join(examples)
        + "\\end{enumerate}\n"
    )

    output_json = RESULTS / "qualitative_examples.json"
    output_json.write_text(
        json.dumps(
            {
                "proco_path": str(proco_path),
                "proof_path": str(proof_path),
                "examples": [
                    {
                        "title": "Clean FAIL[no_rule] success",
                        "example_id": no_rule["prediction"]["example_id"],
                        "question_id": no_rule["prediction"]["question_id"],
                    },
                    {
                        "title": "Rule witness accepted by tolerant verifier",
                        "example_id": tolerant_rule["prediction"]["example_id"],
                        "question_id": tolerant_rule["prediction"]["question_id"],
                    },
                    {
                        "title": "Proof-only generic abstention",
                        "example_id": proof_generic["prediction"]["example_id"],
                        "question_id": proof_generic["prediction"]["question_id"],
                    },
                    {
                        "title": "Strict canonical mismatch on FAIL[ruleX]",
                        "example_id": strict_mismatch["prediction"]["example_id"],
                        "question_id": strict_mismatch["prediction"]["question_id"],
                    },
                ],
            },
            indent=2,
        )
        + "\n"
    )

    print(output_tex)
    print(output_json)


if __name__ == "__main__":
    main()
