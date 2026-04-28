from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import time

from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .dataset import build_records, build_support_deletion_records
from .formatting import parse_model_output
from .symbolic import parse_literal_repr, verify_chain


@dataclass
class EvalConfig:
    model_name: str
    variant: str
    split: str = "dev"
    config_name: str = "depth-3ext-NatLang"
    max_examples: int | None = None
    batch_size: int = 4
    max_prompt_length: int = 1024
    max_new_tokens: int = 96
    seed: int = 0
    data_seed: int | None = None
    adapter_path: str | None = None
    output_path: str = "results/eval.json"
    load_in_4bit: bool = True
    trust_remote_code: bool = True
    study_tag: str = "default"
    model_tag: str = "base"
    train_metadata_path: str | None = None
    notes: str | None = None
    progress_interval_batches: int = 50
    support_deletion: bool = False
    mutation_max_source_examples: int | None = None
    mutation_delete_kinds: str = "fact,rule"


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _load_train_metadata(config: EvalConfig) -> dict | None:
    if config.train_metadata_path:
        metadata_path = Path(config.train_metadata_path)
    elif config.adapter_path:
        metadata_path = Path(config.adapter_path) / "train_metadata.json"
    else:
        return None

    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text())


def _infer_eval_group(train_config_name: str | None, eval_config_name: str) -> str:
    if eval_config_name == "support-deletion":
        return "support_deletion"
    if train_config_name and train_config_name == eval_config_name:
        return "in_domain"
    if eval_config_name == "depth-5":
        return "depth_ood"
    if eval_config_name == "birds-electricity":
        return "domain_transfer"
    return "template_transfer"


def _eval_scope(max_examples: int | None) -> str:
    return "full" if max_examples is None else f"subset_{int(max_examples)}"


def _load_model_and_tokenizer(config: EvalConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        config.adapter_path or config.model_name,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    quantization_config = None
    if config.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if config.adapter_path:
        model = PeftModel.from_pretrained(model, config.adapter_path)
    model.eval()
    return model, tokenizer


def _batch(iterable: list[dict], batch_size: int) -> list[list[dict]]:
    return [iterable[i : i + batch_size] for i in range(0, len(iterable), batch_size)]


def _classification_metrics(golds: list[str], preds: list[str]) -> dict:
    labels = ["True", "False", "Unknown"]
    accuracy = sum(g == p for g, p in zip(golds, preds)) / max(1, len(golds))
    per_class = {}
    f1s = []
    for label in labels:
        tp = sum((g == label) and (p == label) for g, p in zip(golds, preds))
        fp = sum((g != label) and (p == label) for g, p in zip(golds, preds))
        fn = sum((g == label) and (p != label) for g, p in zip(golds, preds))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        f1s.append(f1)
    return {
        "accuracy": accuracy,
        "macro_f1": sum(f1s) / len(f1s),
        "per_class": per_class,
    }


def _unknown_faithful(record: dict, parsed: dict) -> bool:
    if parsed["mode"] != "ABSTAIN":
        return False
    chain_text = _normalize_text(parsed["chain_text"] or "")
    gold_failure_tag = _normalize_text(record.get("gold_failure_tag") or "")
    if gold_failure_tag and chain_text == gold_failure_tag:
        return True

    witness = _normalize_text(parsed["witness"] or "")
    gold_failure_witness = _normalize_text(record.get("gold_failure_witness") or "")
    if gold_failure_witness and witness == gold_failure_witness:
        return True

    missing_texts = [_normalize_text(text) for text in record.get("gold_missing_texts", []) if text]
    if missing_texts and all(missing in witness for missing in missing_texts):
        return True

    return False


def evaluate_model(config: EvalConfig) -> dict:
    data_seed = config.seed if config.data_seed is None else config.data_seed
    eval_config_name = "support-deletion" if config.support_deletion else config.config_name
    if config.support_deletion:
        deletion_kinds = {
            item.strip()
            for item in config.mutation_delete_kinds.split(",")
            if item.strip()
        }
        records = build_support_deletion_records(
            config_name=config.config_name,
            split=config.split,
            variant=config.variant,
            max_source_examples=config.mutation_max_source_examples,
            max_mutants=config.max_examples,
            seed=data_seed,
            deletion_kinds=deletion_kinds,
        )
    else:
        records = build_records(
            config_name=config.config_name,
            split=config.split,
            variant=config.variant,
            max_examples=config.max_examples,
            seed=data_seed,
        )
    model, tokenizer = _load_model_and_tokenizer(config)
    train_metadata = _load_train_metadata(config)
    train_config = (train_metadata or {}).get("config", {})
    eval_group = _infer_eval_group(
        train_config_name=train_config.get("train_config_name"),
        eval_config_name=eval_config_name,
    )

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = output_path.with_name(output_path.stem + ".progress.json")

    predictions: list[dict] = []
    golds: list[str] = []
    preds: list[str] = []
    valid_format = 0
    faithful = 0
    joint = 0
    batches = _batch(records, config.batch_size)
    total_batches = len(batches)
    eval_scope = _eval_scope(config.max_examples)
    start_time = time.time()

    def write_progress(processed_examples: int, batch_index: int, completed: bool = False) -> None:
        elapsed_seconds = time.time() - start_time
        progress_payload = {
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "variant": config.variant,
            "config_name": eval_config_name,
            "split": config.split,
            "eval_scope": eval_scope,
            "requested_max_examples": config.max_examples,
            "output_path": str(output_path),
            "processed_examples": processed_examples,
            "total_examples": len(records),
            "processed_batches": batch_index,
            "total_batches": total_batches,
            "pct_complete": processed_examples / max(1, len(records)),
            "elapsed_seconds": elapsed_seconds,
            "examples_per_second": processed_examples / max(elapsed_seconds, 1e-6),
            "completed": completed,
        }
        progress_path.write_text(json.dumps(progress_payload, indent=2) + "\n")

    print(
        f"[start] variant={config.variant} eval={config.config_name}/{config.split} "
        f"scope={eval_scope} examples={len(records)} batches={total_batches}",
        flush=True,
    )
    write_progress(processed_examples=0, batch_index=0, completed=False)

    for batch_index, batch in enumerate(batches, start=1):
        prompts = [record["prompt"] for record in batch]
        inputs = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=config.max_prompt_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        texts = tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)
        for record, text in zip(batch, texts):
            parsed = parse_model_output(text)
            pred_label = parsed["label"] or "Unknown"
            gold_label = record["gold_label"]
            golds.append(gold_label)
            preds.append(pred_label)

            is_valid = pred_label in {"True", "False", "Unknown"}
            if config.variant != "answer_only":
                is_valid = is_valid and parsed["mode"] in {"PROVE", "REFUTE", "ABSTAIN"}
            valid_format += int(is_valid)

            is_faithful = False
            if gold_label in {"True", "False"} and parsed["chain_tokens"]:
                target = parse_literal_repr(record["query_representation"])
                if gold_label == "False":
                    target = target.negate()
                is_faithful = verify_chain(record["theory"], parsed["chain_tokens"], target)
            elif gold_label == "Unknown" and config.variant != "answer_only":
                is_faithful = _unknown_faithful(record, parsed)

            faithful += int(is_faithful)
            joint += int((pred_label == gold_label) and is_faithful)

            predictions.append(
                {
                    "example_id": record["example_id"],
                    "question_id": record["question_id"],
                    "gold_label": gold_label,
                    "pred_label": pred_label,
                    "gold_mode": record["gold_mode"],
                    "question_text": record["question_text"],
                    "query_representation": record["query_representation"],
                    "gold_chain_text": record["gold_chain_text"],
                    "gold_witness": record["gold_witness"],
                    "gold_failure_tag": record.get("gold_failure_tag"),
                    "gold_failure_witness": record.get("gold_failure_witness"),
                    "mutation_metadata": record.get("mutation_metadata"),
                    "raw_output": text,
                    "parsed": parsed,
                    "faithful": is_faithful,
                }
            )

        processed_examples = len(predictions)
        if (
            batch_index == 1
            or batch_index == total_batches
            or (
                config.progress_interval_batches > 0
                and batch_index % config.progress_interval_batches == 0
            )
        ):
            elapsed_seconds = time.time() - start_time
            print(
                f"[progress] variant={config.variant} eval={config.config_name}/{config.split} "
                f"scope={eval_scope} batch={batch_index}/{total_batches} "
                f"examples={processed_examples}/{len(records)} "
                f"elapsed_min={elapsed_seconds / 60.0:.1f} "
                f"ex_per_sec={processed_examples / max(elapsed_seconds, 1e-6):.2f}",
                flush=True,
            )
            write_progress(processed_examples=processed_examples, batch_index=batch_index, completed=False)

    summary = _classification_metrics(golds, preds)
    summary.update(
        {
            "variant": config.variant,
            "config_name": eval_config_name,
            "split": config.split,
            "examples": len(records),
            "valid_format_rate": valid_format / max(1, len(records)),
            "faithfulness_rate": faithful / max(1, len(records)),
            "joint_accuracy": joint / max(1, len(records)),
            "eval_group": eval_group,
            "source_config_name": config.config_name if config.support_deletion else None,
        }
    )

    payload = {
        "summary": summary,
        "config": asdict(config),
        "metadata": {
            "study_tag": config.study_tag,
            "model_tag": config.model_tag,
            "train_metadata_path": config.train_metadata_path,
            "train_metadata": train_metadata,
            "train_config_name": train_config.get("train_config_name"),
            "train_split": train_config.get("train_split"),
            "train_max_examples": train_config.get("train_max_examples"),
            "train_records": (train_metadata or {}).get("train_records"),
            "eval_group": eval_group,
            "eval_scope": eval_scope,
            "eval_max_examples": config.max_examples,
            "support_deletion": config.support_deletion,
            "mutation_max_source_examples": config.mutation_max_source_examples,
            "mutation_delete_kinds": config.mutation_delete_kinds,
            "notes": config.notes,
        },
        "predictions": predictions,
    }
    output_path.write_text(json.dumps(payload, indent=2))
    write_progress(processed_examples=len(records), batch_index=total_batches, completed=True)
    print(
        f"[done] variant={config.variant} eval={config.config_name}/{config.split} "
        f"scope={eval_scope} accuracy={summary['accuracy']:.4f} "
        f"faithfulness={summary['faithfulness_rate']:.4f} joint={summary['joint_accuracy']:.4f}",
        flush=True,
    )
    return summary
