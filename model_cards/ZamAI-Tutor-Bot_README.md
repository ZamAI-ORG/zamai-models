---
language: ps
license: apache-2.0
tags:
- education
- pashto
- tutoring
- afghanistan
base_model: mistralai/Mistral-7B-Instruct-v0.2
pipeline_tag: text-generation
---

# 🇦🇫 ZamAI-Tutor-Bot

## Model Description
ZamAI Educational Tutor for Pashto learning

This model is part of the ZamAI (زم AI) project - Afghanistan's premier AI initiative for Pashto language processing.

## Model Details
- **Base Model**: mistralai/Mistral-7B-Instruct-v0.2
- **Language**: Pashto (ps)
- **Type**: text-generation
- **License**: Apache 2.0

## Usage

```python
from transformers import pipeline

# Initialize pipeline
pipe = pipeline("text-generation", model="tasal9/ZamAI-Tutor-Bot")

# Example usage
result = pipe("Your input text here")
print(result)
```

## Training Data
Trained on high-quality Pashto datasets with focus on Afghan cultural context and Islamic values.

## Limitations
- Optimized for Pashto language
- Cultural context may be specific to Afghanistan
- Requires internet connection for inference

## ZamAI Project
Part of the comprehensive ZamAI ecosystem:
- Voice Assistant
- Educational Tutor
- Business Automation
- Multilingual Embeddings

## Contact
For questions or collaboration: tasal9@huggingface.co

---
🇦🇫 **د افغانستان د AI پروژه** - ZamAI Project
