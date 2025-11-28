#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt reportlab

python scripts/generate_dummy_pdf.py
python -m src.extract
python -m src.build_dataset
python -m src.finetune --output_dir models/test-sft --epochs 0 --batch_size 1 --grad_accum 1 || true
python -m src.ingest --model models/test-sft
python -m src.app --model models/test-sft
