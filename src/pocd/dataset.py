from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

from datasets import load_dataset
import torch

from .formatting import format_prompt, format_target
from .symbolic import (
    Theory,
    build_theory,
    collect_missing_literals,
    explain_failure,
    extract_chain_tokens,
    failure_chain_tag,
    failure_witness_text,
    forward_chain,
    parse_literal_repr,
)


SUPPORTED_VARIANTS = {
    "answer_only",
    "proof_only",
    "proco",
    "proco_chain",
    "proco_witness",
    "proco_no_refute",
}


def _missing_texts_from_failure(failure) -> list[str]:
    return [literal.to_text() for literal in collect_missing_literals(failure)]


@dataclass
class TokenizedDataset(torch.utils.data.Dataset):
    items: list[dict]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


def _sorted_payload_lines(payload: dict[str, dict | None]) -> list[str]:
    lines: list[str] = []
    for key in sorted(payload.keys(), key=lambda x: int(x.replace("triple", "").replace("rule", ""))):
        value = payload[key]
        if value is None:
            continue
        lines.append(f"{key}: {value['text']}")
    return lines


def _build_record(
    raw_example: dict,
    theory: Theory,
    derivations: dict,
    question_id: str,
    question_payload: dict,
    variant: str,
    config_name: str,
    split: str,
) -> dict:
    query_literal = parse_literal_repr(question_payload["representation"])
    answer = question_payload["answer"]
    gold_mode = {"True": "PROVE", "False": "REFUTE", "Unknown": "ABSTAIN"}[answer]

    gold_failure_tag = None
    gold_failure_witness = None
    if answer == "Unknown":
        failure = explain_failure(target=query_literal, theory=theory, derivations=derivations)
        gold_failure_tag = failure_chain_tag(failure)
        gold_failure_witness = failure_witness_text(failure)
        missing_texts = _missing_texts_from_failure(failure)

        if variant in {"proof_only", "proco_witness"}:
            gold_chain_text = "NONE"
        else:
            gold_chain_text = gold_failure_tag

        if variant in {"proof_only", "proco_chain"}:
            gold_witness = "No valid proof found."
        else:
            gold_witness = gold_failure_witness
    else:
        chain_tokens = extract_chain_tokens(question_payload)
        if answer == "False" and variant == "proco_no_refute":
            gold_chain_text = "NONE"
            gold_witness = "The query is false."
        else:
            gold_chain_text = " -> ".join(chain_tokens) if chain_tokens else "NONE"
            gold_witness = (
                query_literal.to_text()
                if answer == "True"
                else query_literal.negate().to_text()
            )
        missing_texts = []

    record = {
        "example_id": raw_example["id"],
        "question_id": question_id,
        "config_name": config_name,
        "split": split,
        "theory_text": raw_example["theory"],
        "question_text": question_payload["question"],
        "fact_lines": _sorted_payload_lines(raw_example["triples"]),
        "rule_lines": _sorted_payload_lines(raw_example["rules"]),
        "gold_label": answer,
        "gold_mode": gold_mode,
        "gold_chain_text": gold_chain_text,
        "gold_witness": gold_witness,
        "gold_failure_tag": gold_failure_tag,
        "gold_failure_witness": gold_failure_witness,
        "gold_missing_texts": missing_texts,
        "query_representation": question_payload["representation"],
        "question_payload": question_payload,
        "theory": theory,
    }
    record["prompt"] = format_prompt(record, variant)
    record["target"] = format_target(record, variant)
    return record


def build_records(
    config_name: str,
    split: str,
    variant: str,
    max_examples: int | None = None,
    seed: int = 0,
) -> list[dict]:
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"Unsupported variant: {variant}")

    raw_dataset = load_dataset("hitachi-nlp/proofwriter_processed_OWA", config_name, split=split)
    records: list[dict] = []
    for raw_example in raw_dataset:
        theory = build_theory(raw_example)
        derivations = forward_chain(theory)
        for question_id, question_payload in sorted(raw_example["questions"].items()):
            if question_payload is None:
                continue
            records.append(
                _build_record(
                    raw_example=raw_example,
                    theory=theory,
                    derivations=derivations,
                    question_id=question_id,
                    question_payload=question_payload,
                    variant=variant,
                    config_name=config_name,
                    split=split,
                )
            )

    if max_examples is not None and max_examples < len(records):
        rng = random.Random(seed)
        records = rng.sample(records, max_examples)

    return records


def tokenize_records(
    records: Iterable[dict],
    tokenizer,
    max_length: int,
) -> TokenizedDataset:
    items: list[dict] = []

    for record in records:
        prompt_ids = tokenizer(record["prompt"], add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(
            record["target"] + (tokenizer.eos_token or ""),
            add_special_tokens=False,
        )["input_ids"]

        available_prompt_len = max_length - len(target_ids)
        if available_prompt_len <= 0:
            continue
        prompt_ids = prompt_ids[-available_prompt_len:]
        input_ids = prompt_ids + target_ids
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(prompt_ids) + target_ids
        items.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )

    return TokenizedDataset(items)


def build_generation_prompts(records: Iterable[dict]) -> list[str]:
    return [record["prompt"] for record in records]
