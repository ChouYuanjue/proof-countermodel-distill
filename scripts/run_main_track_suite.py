#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import fcntl
import io
import os
from pathlib import Path
import queue
import re
import subprocess
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
GPU_RESOURCE_ERROR_PATTERNS = [
    r"cuda out of memory",
    r"torch\.cuda\.outofmemoryerror",
    r"cublas_status_alloc_failed",
    r"cudnn_status_alloc_failed",
    r"no cuda gpus are available",
    r"all cuda-capable devices are busy or unavailable",
    r"cuda-capable device.*busy or unavailable",
    r"device.*busy or unavailable",
    r"some modules are dispatched on the cpu or the disk",
    r"make sure you have enough gpu ram",
    r"runtimeerror: cuda error",
    r"cuda error: initialization error",
]
GPU_PRECHECK_USED_MEMORY_MIB = 4096


@dataclass(frozen=True)
class EvalSpec:
    config_name: str
    split: str
    max_examples: int | None
    batch_size: int = 16
    max_prompt_length: int = 512
    max_new_tokens: int = 64

    @property
    def scope_tag(self) -> str:
        return "full" if self.max_examples is None else f"subset{self.max_examples}"


@dataclass(frozen=True)
class RunSpec:
    study_tag: str
    model_name: str
    model_tag: str
    variant: str
    seed: int
    train_max_examples: int | None
    evals: tuple[EvalSpec, ...]
    train_config_name: str = "depth-3ext-NatLang"
    train_split: str = "train"
    max_length: int = 512
    epochs: float = 1.0
    lr: float = 2e-4
    batch_size: int = 2
    eval_batch_size: int = 2
    grad_accum: int = 8
    notes: str | None = None

    @property
    def train_label(self) -> str:
        return "full" if self.train_max_examples is None else str(self.train_max_examples)

    @property
    def artifact_dir(self) -> Path:
        return ROOT / "artifacts" / (
            f"{self.study_tag}_{self.variant}_{self.model_tag}_train{self.train_label}_s{self.seed}"
        )


class GPUResourceError(RuntimeError):
    pass


def _result_path(spec: RunSpec, eval_spec: EvalSpec) -> Path:
    return ROOT / "results" / (
        f"{spec.study_tag}_{spec.variant}_{spec.model_tag}_train{spec.train_label}_s{spec.seed}_"
        f"{eval_spec.config_name}_{eval_spec.split}_{eval_spec.scope_tag}.json"
    )


def _log_path(spec: RunSpec, suffix: str) -> Path:
    log_dir = ROOT / "logs" / "maintrack"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{spec.study_tag}_{spec.variant}_{spec.model_tag}_train{spec.train_label}_s{spec.seed}_{suffix}.log"


def _train_lock_path(spec: RunSpec) -> Path:
    lock_dir = ROOT / "state" / "train_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    return lock_dir / f"{spec.study_tag}_{spec.variant}_{spec.model_tag}_train{spec.train_label}_s{spec.seed}.lock"


def _train_command(spec: RunSpec) -> list[str]:
    command = [
        "python",
        "scripts/train_variant.py",
        "--model-name",
        spec.model_name,
        "--variant",
        spec.variant,
        "--output-dir",
        str(spec.artifact_dir),
        "--train-config-name",
        spec.train_config_name,
        "--eval-config-name",
        spec.train_config_name,
        "--train-split",
        spec.train_split,
        "--eval-split",
        "dev",
        "--eval-max-examples",
        "0",
        "--max-length",
        str(spec.max_length),
        "--epochs",
        str(spec.epochs),
        "--lr",
        str(spec.lr),
        "--batch-size",
        str(spec.batch_size),
        "--eval-batch-size",
        str(spec.eval_batch_size),
        "--grad-accum",
        str(spec.grad_accum),
        "--seed",
        str(spec.seed),
        "--study-tag",
        spec.study_tag,
        "--model-tag",
        spec.model_tag,
    ]
    if spec.train_max_examples is not None:
        command.extend(["--train-max-examples", str(spec.train_max_examples)])
    if spec.notes:
        command.extend(["--notes", spec.notes])
    return command


def _eval_command(spec: RunSpec, eval_spec: EvalSpec, eval_data_seed: int) -> list[str]:
    command = [
        "python",
        "scripts/evaluate_variant.py",
        "--model-name",
        spec.model_name,
        "--variant",
        spec.variant,
        "--adapter-path",
        str(spec.artifact_dir),
        "--output-path",
        str(_result_path(spec, eval_spec)),
        "--config-name",
        eval_spec.config_name,
        "--split",
        eval_spec.split,
        "--batch-size",
        str(eval_spec.batch_size),
        "--max-prompt-length",
        str(eval_spec.max_prompt_length),
        "--max-new-tokens",
        str(eval_spec.max_new_tokens),
        "--seed",
        str(spec.seed),
        "--data-seed",
        str(eval_data_seed),
        "--study-tag",
        spec.study_tag,
        "--model-tag",
        spec.model_tag,
        "--progress-interval-batches",
        "50",
    ]
    if eval_spec.max_examples is not None:
        command.extend(["--max-examples", str(eval_spec.max_examples)])
    if spec.notes:
        command.extend(["--notes", spec.notes])
    return command


def _log_tail(path: Path, max_chars: int = 12000) -> str:
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _parse_int(token: str) -> int | None:
    value = token.strip()
    if not value or value.upper() == "N/A":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _nvidia_smi_query(*args: str) -> list[list[str]]:
    process = subprocess.run(
        ["nvidia-smi", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        return []
    rows: list[list[str]] = []
    for row in csv.reader(io.StringIO(process.stdout)):
        cleaned = [item.strip() for item in row]
        if cleaned and any(item for item in cleaned):
            rows.append(cleaned)
    return rows


def _detect_busy_gpus(gpus: list[int]) -> dict[int, str]:
    candidates = set(gpus)
    if not candidates:
        return {}

    gpu_rows = _nvidia_smi_query(
        "--query-gpu=index,uuid,memory.used",
        "--format=csv,noheader,nounits",
    )
    if not gpu_rows:
        return {}

    uuid_to_index: dict[str, int] = {}
    memory_used_by_gpu: dict[int, int] = {}
    for row in gpu_rows:
        if len(row) < 3:
            continue
        index = _parse_int(row[0])
        uuid = row[1]
        memory_used = _parse_int(row[2])
        if index is None:
            continue
        uuid_to_index[uuid] = index
        if memory_used is not None:
            memory_used_by_gpu[index] = memory_used

    busy: dict[int, str] = {}
    process_rows = _nvidia_smi_query(
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    )
    for row in process_rows:
        if len(row) < 4:
            continue
        gpu_uuid, pid, process_name, used_memory = row[:4]
        index = uuid_to_index.get(gpu_uuid)
        if index is None or index not in candidates or not pid or pid.upper() == "N/A":
            continue
        used_memory_mib = _parse_int(used_memory)
        used_memory_text = f"{used_memory_mib}MiB" if used_memory_mib is not None else used_memory
        reason = f"active compute pid={pid} name={process_name} mem={used_memory_text}"
        busy[index] = reason if index not in busy else f"{busy[index]}; {reason}"

    for gpu in sorted(candidates):
        if gpu in busy:
            continue
        memory_used = memory_used_by_gpu.get(gpu)
        if memory_used is not None and memory_used >= GPU_PRECHECK_USED_MEMORY_MIB:
            busy[gpu] = f"memory.used={memory_used}MiB without visible compute owner"

    return busy


def _is_gpu_resource_error(log_path: Path) -> bool:
    tail = _log_tail(log_path).lower()
    return any(re.search(pattern, tail) for pattern in GPU_RESOURCE_ERROR_PATTERNS)


def _run_step(command: list[str], gpu: int, log_path: Path) -> None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with log_path.open("w") as handle:
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if process.returncode != 0:
        if _is_gpu_resource_error(log_path):
            raise GPUResourceError(f"GPU {gpu} unavailable for command: {' '.join(command)}")
        raise RuntimeError(f"Command failed with code {process.returncode}: {' '.join(command)}")


def _should_train(spec: RunSpec, force_train: bool) -> bool:
    if force_train:
        return True
    return not (spec.artifact_dir / "train_metadata.json").exists()


def _should_eval(spec: RunSpec, eval_spec: EvalSpec, force_eval: bool) -> bool:
    if force_eval:
        return True
    return not _result_path(spec, eval_spec).exists()


def _dedupe_train_only_specs(specs: list[RunSpec]) -> list[RunSpec]:
    deduped: list[RunSpec] = []
    seen_artifacts: set[Path] = set()
    for spec in specs:
        artifact_dir = spec.artifact_dir
        if artifact_dir in seen_artifacts:
            continue
        seen_artifacts.add(artifact_dir)
        deduped.append(spec)
    return deduped


def _worker(
    gpu: int,
    gpus: list[int],
    tasks: "queue.Queue[RunSpec]",
    force_train: bool,
    force_eval: bool,
    skip_train: bool,
    skip_eval: bool,
    eval_data_seed: int,
    stop_event: threading.Event,
    disabled_gpus: set[int],
    state_lock: threading.Lock,
) -> None:
    while not stop_event.is_set():
        with state_lock:
            if gpu in disabled_gpus:
                return
        try:
            spec = tasks.get_nowait()
        except queue.Empty:
            return
        try:
            busy_reason = _detect_busy_gpus([gpu]).get(gpu)
            if busy_reason is not None:
                raise GPUResourceError(f"preflight bypass: {busy_reason}")
            print(
                f"[GPU {gpu}] start {spec.study_tag} {spec.variant} train={spec.train_label} seed={spec.seed}",
                flush=True,
            )
            if not skip_train and _should_train(spec, force_train):
                lock_path = _train_lock_path(spec)
                with lock_path.open("w") as handle:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                    if _should_train(spec, force_train):
                        _run_step(_train_command(spec), gpu, _log_path(spec, "train"))
            if not skip_eval:
                for eval_spec in spec.evals:
                    if _should_eval(spec, eval_spec, force_eval):
                        suffix = f"eval_{eval_spec.config_name}_{eval_spec.split}_{eval_spec.scope_tag}"
                        print(
                            f"[GPU {gpu}] eval  {spec.study_tag} {spec.variant} train={spec.train_label} "
                            f"seed={spec.seed} {eval_spec.config_name}/{eval_spec.split} scope={eval_spec.scope_tag}",
                            flush=True,
                        )
                        _run_step(
                            _eval_command(spec, eval_spec, eval_data_seed),
                            gpu,
                            _log_path(spec, suffix),
                        )
                        print(
                            f"[GPU {gpu}] done  {spec.study_tag} {spec.variant} train={spec.train_label} "
                            f"seed={spec.seed} {eval_spec.config_name}/{eval_spec.split} scope={eval_spec.scope_tag}",
                            flush=True,
                        )
            print(
                f"[GPU {gpu}] done  {spec.study_tag} {spec.variant} train={spec.train_label} seed={spec.seed}",
                flush=True,
            )
        except GPUResourceError as exc:
            tasks.put(spec)
            with state_lock:
                disabled_gpus.add(gpu)
                remaining = len([item for item in gpus if item not in disabled_gpus])
            print(
                f"[GPU {gpu}] unavailable, requeue {spec.study_tag} {spec.variant} train={spec.train_label} "
                f"seed={spec.seed}: {exc}",
                flush=True,
            )
            if remaining <= 0:
                stop_event.set()
                print("[suite] no remaining GPUs after automatic bypass", flush=True)
                raise
            return
        except Exception as exc:
            stop_event.set()
            print(f"[GPU {gpu}] failed: {exc}", flush=True)
            raise
        finally:
            tasks.task_done()


def build_suite(model_name: str, model_tag: str, eval_batch_size: int) -> list[RunSpec]:
    subset_evals = (
        EvalSpec(config_name="depth-3ext-NatLang", split="test", max_examples=4000, batch_size=eval_batch_size),
        EvalSpec(config_name="depth-5", split="test", max_examples=4000, batch_size=eval_batch_size),
    )
    full_transfer_evals = (
        EvalSpec(config_name="depth-3ext-NatLang", split="test", max_examples=None, batch_size=eval_batch_size),
        EvalSpec(config_name="depth-5", split="test", max_examples=None, batch_size=eval_batch_size),
        EvalSpec(config_name="NatLang", split="test", max_examples=None, batch_size=eval_batch_size),
        EvalSpec(config_name="depth-3", split="test", max_examples=None, batch_size=eval_batch_size),
        EvalSpec(config_name="birds-electricity", split="test", max_examples=None, batch_size=eval_batch_size),
    )

    specs: list[RunSpec] = []

    for train_max_examples, seeds in [
        (4096, (0, 1, 2)),
        (16384, (0,)),
        (32768, (0,)),
        (None, (0,)),
    ]:
        for variant in ["answer_only", "proof_only", "proco"]:
            for seed in seeds:
                specs.append(
                    RunSpec(
                        study_tag="maintrack",
                        model_name=model_name,
                        model_tag=model_tag,
                        variant=variant,
                        seed=seed,
                        train_max_examples=train_max_examples,
                        evals=subset_evals,
                        notes="main-track canonical suite",
                    )
                )

    for variant in ["proof_only", "proco_chain", "proco_witness", "proco"]:
        specs.append(
            RunSpec(
                study_tag="ablation",
                model_name=model_name,
                model_tag=model_tag,
                variant=variant,
                seed=0,
                train_max_examples=4096,
                evals=subset_evals,
                notes="ablation suite on abstention target components",
            )
        )

    for train_max_examples in [32768, None]:
        for variant in ["answer_only", "proof_only", "proco"]:
            specs.append(
                RunSpec(
                    study_tag="maintrack",
                    model_name=model_name,
                    model_tag=model_tag,
                    variant=variant,
                    seed=0,
                    train_max_examples=train_max_examples,
                    evals=full_transfer_evals,
                    notes="full transfer evaluation for strongest checkpoints",
                )
            )

    return specs


def _filter_specs(
    specs: list[RunSpec],
    study_tags: set[str] | None,
    variants: set[str] | None,
    train_labels: set[str] | None,
    seeds: set[int] | None,
) -> list[RunSpec]:
    filtered: list[RunSpec] = []
    for spec in specs:
        if study_tags is not None and spec.study_tag not in study_tags:
            continue
        if variants is not None and spec.variant not in variants:
            continue
        if train_labels is not None and spec.train_label not in train_labels:
            continue
        if seeds is not None and spec.seed not in seeds:
            continue
        filtered.append(spec)
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model-tag", default="qwen7b")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-data-seed", type=int, default=0)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--study-tags")
    parser.add_argument("--variants")
    parser.add_argument("--train-labels")
    parser.add_argument("--seeds")
    args = parser.parse_args()

    specs = build_suite(model_name=args.model_name, model_tag=args.model_tag, eval_batch_size=args.eval_batch_size)
    study_tags = set(args.study_tags.split(",")) if args.study_tags else None
    variants = set(args.variants.split(",")) if args.variants else None
    train_labels = set(args.train_labels.split(",")) if args.train_labels else None
    seeds = {int(token) for token in args.seeds.split(",")} if args.seeds else None
    specs = _filter_specs(specs, study_tags, variants, train_labels, seeds)
    if args.skip_eval and not args.skip_train:
        specs = _dedupe_train_only_specs(specs)
    if args.dry_run:
        for spec in specs:
            print(spec)
        return

    tasks: "queue.Queue[RunSpec]" = queue.Queue()
    for spec in specs:
        tasks.put(spec)

    requested_gpus = [int(token) for token in args.gpus.split(",") if token.strip()]
    preflight_busy = _detect_busy_gpus(requested_gpus)
    for gpu, reason in sorted(preflight_busy.items()):
        print(f"[suite] preflight bypass GPU {gpu}: {reason}", flush=True)
    gpus = [gpu for gpu in requested_gpus if gpu not in preflight_busy]
    if not gpus:
        print("[suite] no GPUs available after preflight bypass", flush=True)
        raise SystemExit(1)

    stop_event = threading.Event()
    disabled_gpus: set[int] = set()
    state_lock = threading.Lock()
    threads = [
        threading.Thread(
            target=_worker,
            args=(
                gpu,
                gpus,
                tasks,
                args.force_train,
                args.force_eval,
                args.skip_train,
                args.skip_eval,
                args.eval_data_seed,
                stop_event,
                disabled_gpus,
                state_lock,
            ),
            daemon=True,
        )
        for gpu in gpus
    ]

    start_time = time.time()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if stop_event.is_set():
        raise SystemExit(1)
    print(f"Suite complete in {(time.time() - start_time) / 3600.0:.2f} hours", flush=True)


if __name__ == "__main__":
    main()
