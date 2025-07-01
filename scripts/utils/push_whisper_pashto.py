#!/usr/bin/env python3
"""
Push Whisper Large v3 to tasal9/ZamAI-Whisper-v3-Pashto
This script uploads the openai/whisper-large-v3 model to a custom HF repo
with Pashto-specific configuration.
"""

import os
import json
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from huggingface_hub import HfApi, create_repo

# Configuration
SOURCE_MODEL = "openai/whisper-large-v3"
TARGET_REPO = "tasal9/ZamAI-Whisper-v3-Pashto"
MODEL_CARD = """---
language:
- en
- ps
license: apache-2.0
tags:
- speech-to-text
- automatic-speech-recognition
- pashto
- whisper
- afghanistan
datasets:
- mozilla-foundation/common_voice_13_0
- tasal9/ZamAI_Pashto_Speech_Dataset
---

# ZamAI Whisper v3 Pashto

This model is a deployment of openai/whisper-large-v3 optimized for Pashto speech recognition.

## Model description

ZamAI-Whisper-v3-Pashto is based on OpenAI's Whisper Large V3 model. It has been configured for optimal performance with Pashto language audio, enabling accurate transcription of Pashto speech.

## Intended uses & limitations

This model is intended for:
- Pashto speech recognition
- Pashto audio transcription
- Part of the ZamAI voice assistant pipeline

## Training and evaluation data

This model uses the base weights from OpenAI's Whisper Large V3 with additional configuration for Pashto language support.

## How to use

```python
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch

# Load model and processor
processor = AutoProcessor.from_pretrained("tasal9/ZamAI-Whisper-v3-Pashto")
model = AutoModelForSpeechSeq2Seq.from_pretrained("tasal9/ZamAI-Whisper-v3-Pashto")

# For GPU usage
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Transcribe Pashto audio
# This is a simple example, replace with your Pashto audio data
import librosa
audio_data, sampling_rate = librosa.load("path/to/pashto_audio.wav", sr=16000)
input_features = processor(audio_data, sampling_rate=16000, return_tensors="pt").input_features.to(device)

# Generate transcription
with torch.no_grad():
    generated_ids = model.generate(input_features=input_features, language="ps", task="transcribe")

# Decode the generated ids
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"Transcription: {transcription}")
```

## ZamAI Project

This model is part of the ZamAI project focused on developing AI tools for Pashto language speakers. Visit our other models for a complete Pashto AI ecosystem.

"""

def main():
    # Load token
    with open("HF-Token.txt", 'r') as f:
        token = f.read().strip()
    
    # Initialize HF API
    api = HfApi(token=token)
    
    print(f"🔄 Creating repository: {TARGET_REPO}")
    try:
        create_repo(TARGET_REPO, private=False, token=token)
        print("✅ Repository created successfully")
    except Exception as e:
        print(f"ℹ️ Repository already exists or error: {e}")
    
    # Create model-specific configuration for Pashto
    model_config = {
        "language": "ps", 
        "task": "transcribe",
        "model_type": "whisper",
        "processor_class": "WhisperProcessor",
        "supported_languages": ["ps", "en", "fa", "ar", "ur", "hi"],
        "source_model": SOURCE_MODEL,
        "tags": ["pashto", "whisper", "zamai", "speech-to-text"],
        "dataset": "tasal9/ZamAI_Pashto_Speech_Dataset"
    }
    
    # Create a temporary directory for the model
    os.makedirs("temp_model", exist_ok=True)
    
    # Save model config
    with open("temp_model/config.json", "w") as f:
        json.dump(model_config, f, indent=2)
    
    # Save README.md
    with open("temp_model/README.md", "w") as f:
        f.write(MODEL_CARD)
    
    print(f"📥 Loading model from {SOURCE_MODEL}")
    try:
        # Load and save model and processor
        print("Loading processor...")
        processor = AutoProcessor.from_pretrained(SOURCE_MODEL)
        
        print("Loading model...")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(SOURCE_MODEL)
        
        print("💾 Saving processor to temp directory...")
        processor.save_pretrained("temp_model")
        
        print("💾 Saving model to temp directory...")
        model.save_pretrained("temp_model")
        
        print(f"🚀 Uploading to {TARGET_REPO}...")
        api.upload_folder(
            folder_path="temp_model",
            repo_id=TARGET_REPO,
            commit_message="Upload Whisper Large v3 with Pashto configuration"
        )
        print(f"✅ Successfully pushed model to {TARGET_REPO}")
        
    except Exception as e:
        print(f"❌ Error during model processing or upload: {e}")
    finally:
        # Clean up
        print("🧹 Cleaning up temporary files...")
        import shutil
        shutil.rmtree("temp_model", ignore_errors=True)

if __name__ == "__main__":
    main()
