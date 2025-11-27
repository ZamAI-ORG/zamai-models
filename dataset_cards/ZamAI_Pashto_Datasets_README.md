---
dataset_name:
  - tasal9/ZamAI_Pashto_Dataset
  - tasal9/ZamAi-Pashto-Datasets-V2
language:
  - ps
  - en
license: apache-2.0
task_categories:
  - text-generation
  - translation
  - question-answering
size_categories:
  - 10K<n<100K
tags:
  - pashto
  - afghanistan
  - instruction-following
  - prompt-completion
  - cultural-context
---

# 🇦🇫 ZamAI Pashto Instruction & Prompt Dataset

High-quality Pashto instruction data curated for conversational AI, translation, and cultural reasoning tasks. The dataset is published under two repository names (`tasal9/ZamAI_Pashto_Dataset` and `tasal9/ZamAi-Pashto-Datasets-V2`) and both share the same files.

## 📦 Contents

| File | Rows | Format | Schema |
|------|------:|--------|--------|
| `pashto_train_instruction.jsonl` | 25,785 | JSONL | `instruction`, `input`, `output` |
| `pashto_val_instruction.jsonl` | 2,865 | JSONL | `instruction`, `input`, `output` |
| `pashto_train_prompt_completion.jsonl` | 25,785 | JSONL | `prompt`, `completion` |
| `pashto_val_prompt_completion.jsonl` | 2,865 | JSONL | `prompt`, `completion` |
| `pashto_cleaned_train.csv` | 25,785 | CSV | 8 columns (title/text metadata + both schemas) |
| `pashto_cleaned_val.csv` | 2,865 | CSV | 8 columns |
| `pashto_cleaned_full_dataset.csv` | 28,650 | CSV | Full corpus |

The `.jsonl` files are mirrored in two synchronized schemas:

- **Instruction format** – preferred for supervised fine-tuning of instruction-following LLMs.
  ```json
  {
    "instruction": "Write a Pashto language article based on the provided title.",
    "input": "کابل کې د اشرف غني احمدزي په پلوۍ غونډه وشوه",
    "output": "..."
  }
  ```
- **Prompt/Completion format** – useful for legacy prompts or minimalist completion APIs.
  ```json
  {
    "prompt": "Title: کابل کې د اشرف غني احمدزي په پلوۍ غونډه وشوه\n\nWrite an article based on this title:",
    "completion": "..."
  }
  ```

## 🧠 Recommended Usage

### Instruction-tuned models
```python
from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files={
        "train": "https://huggingface.co/datasets/tasal9/ZamAi-Pashto-Datasets-V2/resolve/main/pashto_train_instruction.jsonl",
        "validation": "https://huggingface.co/datasets/tasal9/ZamAi-Pashto-Datasets-V2/resolve/main/pashto_val_instruction.jsonl"
    }
)
print(dataset["train"][0])
```

### Prompt/completion workflows
```python
from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files="https://huggingface.co/datasets/tasal9/ZamAi-Pashto-Datasets-V2/resolve/main/pashto_train_prompt_completion.jsonl",
    split="train"
)
print(dataset[0])
```

### Converting prompt/completion → instruction schema
If you need a single schema for the HF dataset viewer or training pipelines, convert the prompt records in-place before uploading:
```python
import json
from pathlib import Path

src = Path("pashto_train_prompt_completion.jsonl")
dst = Path("pashto_train_instruction_ready.jsonl")

def to_instruction(row):
    return {
        "instruction": "Write a Pashto article based on the provided title.",
        "input": row.get("prompt", ""),
        "output": row.get("completion", "")
    }

with src.open() as reader, dst.open("w", encoding="utf-8") as writer:
    for line in reader:
        writer.write(json.dumps(to_instruction(json.loads(line)), ensure_ascii=False) + "\n")
```
Upload the converted file (or define separate dataset configurations) to avoid schema-cast errors like `DatasetGenerationCastError`.

## 🚧 Known Issue & Fix
The HF dataset builder expects identical column names across all data files. Mixing `instruction/input/output` files with `prompt/completion` files inside the same configuration triggers:
```
DatasetGenerationCastError: ... 2 new columns {'completion','prompt'} and 3 missing columns {'output','instruction','input'}
```
**Fix:** either upload only one schema per configuration or convert the prompt/completion files using the script above.

## 📈 Quality Notes
- Articles drawn from Afghan news, opinion, and cultural sources.
- Balanced across political, educational, religious, and social topics.
- Cleaned duplicates and normalized Unicode punctuation.
- Preserves Pashto script; some records include English translations for evaluation.

## 📜 License & Attribution
- License: **Apache 2.0**
- Use requires attribution to ZamAI and the original Afghan media sources when known.
- Intended for research and community applications that respect Afghan culture and Islamic values.

## 📬 Contact
- **Maintainer:** tasal9 (Hugging Face)
- **Email:** tasal9@huggingface.co
- **Project:** [ZamAI Pro Models](https://github.com/tasal9/ZamAI-Pro-Models)

Contributions and issue reports are welcome! Feel free to open a PR or discussion if you create improved subsets or English translations.
