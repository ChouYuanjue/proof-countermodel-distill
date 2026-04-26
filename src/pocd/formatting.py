from __future__ import annotations

import re


LABEL_RE = re.compile(r"^\s*LABEL:\s*(True|False|Unknown)\s*$", re.IGNORECASE | re.MULTILINE)
MODE_RE = re.compile(r"^\s*MODE:\s*(PROVE|REFUTE|ABSTAIN)\s*$", re.IGNORECASE | re.MULTILINE)
CHAIN_RE = re.compile(r"^\s*CHAIN:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
WITNESS_RE = re.compile(r"^\s*WITNESS:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


VARIANT_INSTRUCTIONS = {
    "answer_only": (
        "Return only one line in the format `LABEL: <True|False|Unknown>`."
    ),
    "proof_only": (
        "Return four lines exactly:\n"
        "MODE: <PROVE|REFUTE|ABSTAIN>\n"
        "LABEL: <True|False|Unknown>\n"
        "CHAIN: <triple/rule chain or NONE>\n"
        "WITNESS: <proved statement or concise note>"
    ),
    "proco": (
        "Return four lines exactly:\n"
        "MODE: <PROVE|REFUTE|ABSTAIN>\n"
        "LABEL: <True|False|Unknown>\n"
        "CHAIN: <triple/rule chain or FAIL[...]>\n"
        "WITNESS: <proved opposite statement or missing support explanation>"
    ),
    "proco_chain": (
        "Return four lines exactly:\n"
        "MODE: <PROVE|REFUTE|ABSTAIN>\n"
        "LABEL: <True|False|Unknown>\n"
        "CHAIN: <triple/rule chain or FAIL[...]>\n"
        "WITNESS: <proved statement, proved opposite statement, or concise note>"
    ),
    "proco_witness": (
        "Return four lines exactly:\n"
        "MODE: <PROVE|REFUTE|ABSTAIN>\n"
        "LABEL: <True|False|Unknown>\n"
        "CHAIN: <triple/rule chain or NONE>\n"
        "WITNESS: <proved opposite statement or missing support explanation>"
    ),
    "proco_no_refute": (
        "Return four lines exactly:\n"
        "MODE: <PROVE|REFUTE|ABSTAIN>\n"
        "LABEL: <True|False|Unknown>\n"
        "CHAIN: <triple/rule chain, FAIL[...], or NONE>\n"
        "WITNESS: <proved statement, concise false-case note, or missing support explanation>"
    ),
}


def format_context(record: dict) -> str:
    facts = "\n".join(record["fact_lines"])
    rules = "\n".join(record["rule_lines"])
    return f"FACTS:\n{facts}\n\nRULES:\n{rules}"


def format_prompt(record: dict, variant: str) -> str:
    return (
        "You are a careful logical reasoner.\n"
        "Use only the numbered facts and rules below.\n"
        f"{VARIANT_INSTRUCTIONS[variant]}\n\n"
        f"{format_context(record)}\n\n"
        f"QUESTION: {record['question_text']}\n"
        "ANSWER:\n"
    )


def format_target(record: dict, variant: str) -> str:
    if variant == "answer_only":
        return f"LABEL: {record['gold_label']}"
    return (
        f"MODE: {record['gold_mode']}\n"
        f"LABEL: {record['gold_label']}\n"
        f"CHAIN: {record['gold_chain_text']}\n"
        f"WITNESS: {record['gold_witness']}"
    )


def parse_model_output(text: str) -> dict[str, str | list[str] | None]:
    label_match = LABEL_RE.search(text)
    mode_match = MODE_RE.search(text)
    chain_match = CHAIN_RE.search(text)
    witness_match = WITNESS_RE.search(text)

    chain_text = chain_match.group(1).strip() if chain_match else ""
    chain_tokens = [
        token.strip()
        for token in re.split(r"\s*->\s*", chain_text)
        if token.strip().startswith(("triple", "rule"))
    ]

    return {
        "label": label_match.group(1).title() if label_match else None,
        "mode": mode_match.group(1).upper() if mode_match else None,
        "chain_text": chain_text,
        "chain_tokens": chain_tokens,
        "witness": witness_match.group(1).strip() if witness_match else "",
    }
