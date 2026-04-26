#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/runnel/proof-countermodel-distill"

source /home/runnel/miniconda3/etc/profile.d/conda.sh
conda activate tangut-nlp

mkdir -p "$ROOT/logs/systemd" "$ROOT/state/systemd/markers"
cd "$ROOT"

exec python "$ROOT/scripts/run_systemd_pipeline.py" "$@"
