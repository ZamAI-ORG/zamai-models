---
language: ps
license: apache-2.0
tags:
- embeddings
- pashto
- multilingual
- semantic-search
base_model: intfloat/e5-large-v2
pipeline_tag: feature-extraction
---

# 🇦🇫 ZamAI-Embeddings

## Model Description
ZamAI Multilingual Embeddings for Pashto

This model is part of the ZamAI (زم AI) project - Afghanistan's premier AI initiative for Pashto language processing.

## Model Details
- **Base Model**: intfloat/e5-large-v2
- **Language**: Pashto (ps)
- **Type**: feature-extraction
- **License**: Apache 2.0

## Usage

```python
from transformers import pipeline

# Initialize pipeline
pipe = pipeline("feature-extraction", model="tasal9/ZamAI-Embeddings")

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
