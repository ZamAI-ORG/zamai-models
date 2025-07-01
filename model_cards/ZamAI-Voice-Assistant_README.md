---
language: ps
license: apache-2.0
tags:
- conversational
- pashto
- voice
- afghanistan
base_model: microsoft/DialoGPT-medium
pipeline_tag: text-generation
---

# 🇦🇫 ZamAI-Voice-Assistant

## Model Description
ZamAI Voice Assistant for Pashto conversations

This model is part of the ZamAI (زم AI) project - Afghanistan's premier AI initiative for Pashto language processing.

## Model Details
- **Base Model**: microsoft/DialoGPT-medium
- **Language**: Pashto (ps)
- **Type**: text-generation
- **License**: Apache 2.0

## Usage

```python
from transformers import pipeline

# Initialize pipeline
pipe = pipeline("text-generation", model="tasal9/ZamAI-Voice-Assistant")

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
