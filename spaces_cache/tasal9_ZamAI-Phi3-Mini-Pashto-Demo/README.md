---
title: ZamAI-Phi3-MiniPashto
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
emoji: 📚
colorFrom: blue
colorTo: green
pinned: false
---
![ZeroGPU](https://img.shields.io/badge/ZeroGPU-Enabled-blue)

## Model Overview

**Pashto-Edu-Phi3-SFT** is a language model fine-tuned to Pashto educational content, built on a small Microsoft Phi-3 variant. The pipeline automates collection, extraction, dataset generation, fine-tuning (via QLoRA/PEFT), and serving with a RAG-powered Gradio tutor interface.

- **Base Model:** Microsoft Phi-3 mini (e.g., phi-3-mini-4k-instruct)
- **Architecture:** Dense, decoder-only transformer
- **Fine-tuning:** Supervised Fine-Tuning (SFT) using QLoRA for efficient adaptation

## Pipeline Capabilities

1. **Collect Pashto School Textbooks:**  
   - Crawl and download PDFs from a configured website.
2. **Text Extraction and Cleaning:**  
   - Normalize and extract clean Pashto text from PDFs.
3. **Automatic Dataset Building:**  
   - Instruction/response pairs generated from textbook structure.
   - Optionally augment with chapter-based QA pairs.
4. **Model Fine-Tuning:**  
   - Fine-tune Phi-3 variant using QLoRA/PEFT.
5. **RAG-Enabled Tutor App:**  
   - Gradio interface enabling retrieval-augmented answers for curriculum Q&A.

## Directory Structure

```
data/
  raw_pdfs/        # Original downloaded PDFs
  processed/       # Extracted text & datasets
scripts/           # Utility scripts
src/
  config.py        # Pipeline configuration
  download.py      # PDF crawler/downloader
  extract.py       # Text extraction/normalization
  build_dataset.py # Dataset builder
  finetune.py      # QLoRA fine-tuning
  ingest.py        # RAG vector store builder
  app.py           # Gradio tutor interface
requirements.txt   # Dependencies
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 1. Set SOURCE_BASE_URL in src/config.py
# 2. Download PDFs
python -m src.download
# 3. Extract text
python -m src.extract
# 4. Build dataset
python -m src.build_dataset
# 5. Finetune the model
python -m src.finetune --output_dir models/pashto-phi3-sft
# 6. Build RAG vector store
python -m src.ingest --model models/pashto-phi3-sft
# 7. Launch Gradio app
python -m src.app --model models/pashto-phi3-sft
```

## Intended Use

- **Target Users:** Pashto-speaking learners, educators, and developers.
- **Primary Functions:**  
  - Answering curriculum questions.
  - Exploring textbook material.
  - Supplementing classroom instruction for remote learners.
- **RAG Integration:** Precise, context-aware responses by retrieving textbook passages.

## Training Data

- **Source:** Pashto school textbooks (PDFs)
- **Dataset:**  
  - Instruction: e.g., “Explain the concept of '[Section Title]' from lesson '[Lesson Number]'.”
  - Response: Text from the specified section.
- **Language:** Pashto

## Limitations and Biases

- **Domain-Specific:** Knowledge is limited to textbooks used for fine-tuning.
- **Potential Hallucinations:** May generate incorrect or unrelated information.
- **Extraction Errors:** Output quality depends on PDF text extraction accuracy.
- **Curriculum Bias:** Reflects the cultural/pedagogical stance of source materials.
- **Not a Teacher Replacement:** Tool is supplementary, not a substitute for educators.

## Ethical and Legal Notice

Use only with textbooks/PDFs that permit educational redistribution. Intended to facilitate curriculum access for learners excluded from physical classrooms.

## Evaluation

- **Current:** No formal evaluation notebook; accuracy should be judged on curriculum Q&A performance.
- **Future:** Add evaluation notebook and robust Pashto sentence segmentation.

## TODO

- Add PDF source URL after user provides it.
- Improve Pashto sentence segmentation.
- Implement evaluation notebook.

---

**Citation/Contact:**  
For issues, contributions, or questions, open an issue on this repo.

---

Let me know if you want additional sections, badges, or further customization!