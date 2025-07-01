---
language: ps
license: apache-2.0
tags:
- speech-recognition
- pashto
- stt
- afghanistan
base_model: openai/whisper-base
pipeline_tag: automatic-speech-recognition
---

# 🇦🇫 ZamAI-STT-Processor

## Model Description
ZamAI Speech-to-Text for Pashto

This model is part of the ZamAI (زم AI) project - Afghanistan's premier AI initiative for Pashto language processing.

## Model Details
- **Base Model**: openai/whisper-base
- **Language**: Pashto (ps)
- **Type**: automatic-speech-recognition
- **License**: Apache 2.0

## Usage

```python
from transformers import pipeline

# Initialize pipeline
pipe = pipeline("automatic-speech-recognition", model="tasal9/ZamAI-STT-Processor")

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
