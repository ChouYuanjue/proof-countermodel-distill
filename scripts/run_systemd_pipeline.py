#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "state" / "systemd"
MARKER_DIR = STATE_DIR / "markers"
STATUS_PATH = STATE_DIR / "maintrack_pipeline_status.json"
LOCK_PATH = STATE_DIR / "maintrack_pipeline.lock"
RESULTS_DIR = ROOT / "results"
STATUS_HEARTBEAT_SECONDS = 30


@dataclass(frozen=True)
class Phase:
    name: str
    description: str
    command: tuple[str, ...]

    @property
    def marker_path(self) -> Path:
        return MARKER_DIR / f"{self.name}.done"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def ensure_state_dirs() -> None:
    MARKER_DIR.mkdir(parents=True, exist_ok=True)


def acquire_lock() -> object:
    handle = LOCK_PATH.open("w")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another pipeline instance is already running") from exc
    handle.write(f"pid={os.getpid()} started_at={utc_now()}\n")
    handle.flush()
    return handle


def shell_join(command: tuple[str, ...]) -> str:
    return " ".join(command)


def write_status(
    phases: list[Phase],
    *,
    state: str,
    current_phase: str | None,
    failed_phase: str | None = None,
    details: dict | None = None,
) -> None:
    completed = [phase.name for phase in phases if phase.marker_path.exists()]
    payload = {
        "updated_at": utc_now(),
        "state": state,
        "current_phase": current_phase,
        "failed_phase": failed_phase,
        "completed_phases": completed,
        "pending_phases": [phase.name for phase in phases if phase.name not in completed],
    }
    if details:
        payload.update(details)
    STATUS_PATH.write_text(json.dumps(payload, indent=2) + "\n")


def collect_active_progress(limit: int = 8) -> list[dict]:
    rows: list[dict] = []
    if not RESULTS_DIR.exists():
        return rows

    for path in sorted(RESULTS_DIR.glob("*.progress.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        if payload.get("completed"):
            continue
        staleness_seconds = max(0.0, time.time() - path.stat().st_mtime)
        if staleness_seconds > 6 * 3600:
            continue
        rows.append(
            {
                "progress_path": str(path),
                "output_path": payload.get("output_path"),
                "variant": payload.get("variant"),
                "config_name": payload.get("config_name"),
                "split": payload.get("split"),
                "eval_scope": payload.get("eval_scope"),
                "requested_max_examples": payload.get("requested_max_examples"),
                "processed_examples": payload.get("processed_examples"),
                "total_examples": payload.get("total_examples"),
                "processed_batches": payload.get("processed_batches"),
                "total_batches": payload.get("total_batches"),
                "pct_complete": payload.get("pct_complete"),
                "elapsed_seconds": payload.get("elapsed_seconds"),
                "examples_per_second": payload.get("examples_per_second"),
                "updated_at": payload.get("updated_at"),
                "staleness_seconds": round(staleness_seconds, 1),
            }
        )

    rows.sort(key=lambda row: (row.get("updated_at") or "", row.get("pct_complete") or 0.0), reverse=True)
    return rows[:limit]


def run_command(command: tuple[str, ...], *, phases: list[Phase], current_phase: str) -> None:
    print(f"$ {shell_join(command)}", flush=True)
    process = subprocess.Popen(command, cwd=ROOT)
    try:
        while True:
            returncode = process.poll()
            write_status(
                phases,
                state="running",
                current_phase=current_phase,
                details={
                    "current_command": list(command),
                    "active_progress": collect_active_progress(),
                },
            )
            if returncode is not None:
                if returncode != 0:
                    raise subprocess.CalledProcessError(returncode, command)
                return
            time.sleep(STATUS_HEARTBEAT_SECONDS)
    finally:
        if process.poll() is None:
            process.wait()


def mark_phase_done(phase: Phase, elapsed_seconds: float) -> None:
    payload = {
        "name": phase.name,
        "description": phase.description,
        "completed_at": utc_now(),
        "elapsed_seconds": round(elapsed_seconds, 1),
        "command": list(phase.command),
    }
    phase.marker_path.write_text(json.dumps(payload, indent=2) + "\n")


def suite_command(
    *,
    model_name: str,
    model_tag: str,
    gpus: str,
    study_tags: str,
    train_labels: str,
    seeds: str,
    variants: str,
    eval_batch_size: int,
    eval_data_seed: int,
    skip_train: bool = False,
    skip_eval: bool = False,
    force_eval: bool = False,
) -> tuple[str, ...]:
    command = [
        "python",
        "scripts/run_main_track_suite.py",
        "--model-name",
        model_name,
        "--model-tag",
        model_tag,
        "--study-tags",
        study_tags,
        "--train-labels",
        train_labels,
        "--seeds",
        seeds,
        "--variants",
        variants,
        "--gpus",
        gpus,
        "--eval-batch-size",
        str(eval_batch_size),
        "--eval-data-seed",
        str(eval_data_seed),
    ]
    if skip_train:
        command.append("--skip-train")
    if skip_eval:
        command.append("--skip-eval")
    if force_eval:
        command.append("--force-eval")
    return tuple(command)


def build_phases(args: argparse.Namespace) -> list[Phase]:
    phases = [
        Phase(
            name="maintrack_4k_train",
            description="Ensure 4k three-seed canonical checkpoints exist.",
            command=suite_command(
                model_name=args.model_name,
                model_tag=args.model_tag,
                gpus=args.gpus,
                study_tags="maintrack",
                train_labels="4096",
                seeds="0,1,2",
                variants="answer_only,proof_only,proco",
                eval_batch_size=args.eval_batch_size,
                eval_data_seed=args.eval_data_seed,
                skip_eval=True,
            ),
        ),
        Phase(
            name="maintrack_4k_eval",
            description="Run the fixed-subset 4k three-seed canonical evaluation.",
            command=suite_command(
                model_name=args.model_name,
                model_tag=args.model_tag,
                gpus=args.gpus,
                study_tags="maintrack",
                train_labels="4096",
                seeds="0,1,2",
                variants="answer_only,proof_only,proco",
                eval_batch_size=args.eval_batch_size,
                eval_data_seed=args.eval_data_seed,
                skip_train=True,
            ),
        ),
        Phase(
            name="refresh_after_maintrack_4k",
            description="Refresh tables, plots, summaries, and paper after 4k canonical runs.",
            command=("python", "scripts/refresh_artifacts.py", "--compile-paper"),
        ),
        Phase(
            name="ablation_4k_train",
            description="Train the 4k abstention-component ablations.",
            command=suite_command(
                model_name=args.model_name,
                model_tag=args.model_tag,
                gpus=args.gpus,
                study_tags="ablation",
                train_labels="4096",
                seeds="0",
                variants="proco_chain,proco_witness",
                eval_batch_size=args.eval_batch_size,
                eval_data_seed=args.eval_data_seed,
                skip_eval=True,
            ),
        ),
        Phase(
            name="ablation_4k_eval",
            description="Evaluate proof-only, ProCo, and the two 4k ablation variants.",
            command=suite_command(
                model_name=args.model_name,
                model_tag=args.model_tag,
                gpus=args.gpus,
                study_tags="ablation",
                train_labels="4096",
                seeds="0",
                variants="proof_only,proco_chain,proco_witness,proco",
                eval_batch_size=args.eval_batch_size,
                eval_data_seed=args.eval_data_seed,
                skip_train=True,
            ),
        ),
        Phase(
            name="refresh_after_ablation_4k",
            description="Refresh tables, plots, summaries, and paper after ablations.",
            command=("python", "scripts/refresh_artifacts.py", "--compile-paper"),
        ),
        Phase(
            name="maintrack_16k_train",
            description="Train the 16k canonical checkpoints.",
            command=suite_command(
                model_name=args.model_name,
                model_tag=args.model_tag,
                gpus=args.gpus,
                study_tags="maintrack",
                train_labels="16384",
                seeds="0",
                variants="answer_only,proof_only,proco",
                eval_batch_size=args.eval_batch_size,
                eval_data_seed=args.eval_data_seed,
                skip_eval=True,
            ),
        ),
        Phase(
            name="maintrack_16k_eval",
            description="Evaluate the 16k canonical checkpoints on the fixed subset.",
            command=suite_command(
                model_name=args.model_name,
                model_tag=args.model_tag,
                gpus=args.gpus,
                study_tags="maintrack",
                train_labels="16384",
                seeds="0",
                variants="answer_only,proof_only,proco",
                eval_batch_size=args.eval_batch_size,
                eval_data_seed=args.eval_data_seed,
                skip_train=True,
            ),
        ),
        Phase(
            name="refresh_after_maintrack_16k",
            description="Refresh tables, plots, summaries, and paper after the 16k runs.",
            command=("python", "scripts/refresh_artifacts.py", "--compile-paper"),
        ),
        Phase(
            name="maintrack_32k_train",
            description="Train the 32k canonical checkpoints.",
            command=suite_command(
                model_name=args.model_name,
                model_tag=args.model_tag,
                gpus=args.gpus,
                study_tags="maintrack",
                train_labels="32768",
                seeds="0",
                variants="answer_only,proof_only,proco",
                eval_batch_size=args.eval_batch_size,
                eval_data_seed=args.eval_data_seed,
                skip_eval=True,
            ),
        ),
        Phase(
            name="maintrack_32k_eval",
            description="Run the 32k evaluation bundle and fill any missing subset/full transfer results.",
            command=suite_command(
                model_name=args.model_name,
                model_tag=args.model_tag,
                gpus=args.gpus,
                study_tags="maintrack",
                train_labels="32768",
                seeds="0",
                variants="answer_only,proof_only,proco",
                eval_batch_size=args.eval_batch_size,
                eval_data_seed=args.eval_data_seed,
                skip_train=True,
            ),
        ),
        Phase(
            name="refresh_after_maintrack_32k",
            description="Refresh tables, plots, summaries, and paper after the 32k runs.",
            command=("python", "scripts/refresh_artifacts.py", "--compile-paper"),
        ),
        Phase(
            name="maintrack_full_train",
            description="Train the full-data canonical checkpoints.",
            command=suite_command(
                model_name=args.model_name,
                model_tag=args.model_tag,
                gpus=args.gpus,
                study_tags="maintrack",
                train_labels="full",
                seeds="0",
                variants="answer_only,proof_only,proco",
                eval_batch_size=args.eval_batch_size,
                eval_data_seed=args.eval_data_seed,
                skip_eval=True,
            ),
        ),
        Phase(
            name="maintrack_full_eval",
            description="Run the full-data evaluation bundle and fill any missing subset/full transfer results.",
            command=suite_command(
                model_name=args.model_name,
                model_tag=args.model_tag,
                gpus=args.gpus,
                study_tags="maintrack",
                train_labels="full",
                seeds="0",
                variants="answer_only,proof_only,proco",
                eval_batch_size=args.eval_batch_size,
                eval_data_seed=args.eval_data_seed,
                skip_train=True,
            ),
        ),
        Phase(
            name="refresh_after_maintrack_full",
            description="Refresh tables, plots, summaries, and paper after the full-data runs.",
            command=("python", "scripts/refresh_artifacts.py", "--compile-paper"),
        ),
    ]
    return phases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=os.environ.get("POCD_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"))
    parser.add_argument("--model-tag", default=os.environ.get("POCD_MODEL_TAG", "qwen7b"))
    parser.add_argument("--gpus", default=os.environ.get("POCD_GPUS", "0,1"))
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=int(os.environ.get("POCD_EVAL_BATCH_SIZE", "16")),
    )
    parser.add_argument(
        "--eval-data-seed",
        type=int,
        default=int(os.environ.get("POCD_EVAL_DATA_SEED", "0")),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_state_dirs()
    lock_handle = acquire_lock()
    phases = build_phases(args)
    write_status(phases, state="running", current_phase=None)
    current_phase: str | None = None

    try:
        for phase in phases:
            if phase.marker_path.exists():
                print(f"[skip] {phase.name}: {phase.description}", flush=True)
                write_status(phases, state="running", current_phase=None)
                continue

            current_phase = phase.name
            print(f"[run] {phase.name}: {phase.description}", flush=True)
            write_status(phases, state="running", current_phase=current_phase)
            start_time = time.time()
            run_command(phase.command, phases=phases, current_phase=current_phase)
            mark_phase_done(phase, time.time() - start_time)
            print(f"[done] {phase.name}", flush=True)
            current_phase = None
            write_status(phases, state="running", current_phase=None)

        write_status(phases, state="completed", current_phase=None)
        print("Pipeline complete.", flush=True)
    except subprocess.CalledProcessError as exc:
        write_status(phases, state="failed", current_phase=current_phase, failed_phase=current_phase)
        print(f"Pipeline failed with exit code {exc.returncode}.", flush=True)
        raise
    except Exception:
        write_status(phases, state="failed", current_phase=current_phase, failed_phase=current_phase)
        raise
    finally:
        try:
            lock_handle.close()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
