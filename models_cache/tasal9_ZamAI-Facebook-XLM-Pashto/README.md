# ZamAI-Facebook-XLM-Pashto

Metadata

- license: MIT
- datasets: `tasal9/ZamAI-Pashto-Dataset-Cleaned`
- language: `ps` (Pashto)
- metrics: accuracy
- base_model: `FacebookAI/xlm-roberta-base`
- pipeline_tag: fill-mask
- library_name: transformers / adapter-transformers

Overview

This repository contains helper scripts to download and persist the base model `facebook/xlm-roberta-base` locally (into `./base_model/`) and to run a small fill-mask inference example. Large model files should be handled using Git LFS; `.gitattributes` at the repo root already includes common model file patterns.

Quick start

1. Create and activate a virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download and save the base model into `./base_model/`:

```bash
python download_base_model.py
```

4. Run the example inference (will load from `./base_model/` if present):

```bash
python inference.py
```

Files

- `download_base_model.py` — downloads the Hugging Face model and saves to `./base_model/`.
- `inference.py` — small script to run a fill-mask example.
- `requirements.txt` — Python dependencies.
- `.gitignore` — common ignores.

If you want me to download the base model into the repository now (it will download ~0.5–1.2 GB depending on files), tell me and I'll run the script and save the model into `./base_model/`
---

license: mit
datasets:

- tasal9/ZamAI-Pashto-Dataset-Cleaned
language:
- ps
metrics:
- accuracy
base_model:
- FacebookAI/xlm-roberta-base
pipeline_tag: fill-mask
library_name: adapter-transformers

---
