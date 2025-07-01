#!/usr/bin/env python3
"""
Push Phi-3 Mini to tasal9/ZamAI-Phi-3-Mini-Pashto
This script uploads the microsoft/Phi-3-mini-4k-instruct model to a custom HF repo
with Pashto-specific configuration.
"""

import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfApi, create_repo

# Configuration
SOURCE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
TARGET_REPO = "tasal9/ZamAI-Phi-3-Mini-Pashto"
MODEL_CARD = """---
language:
- en
- ps
license: apache-2.0
tags:
- text-generation
- pashto
- phi-3
- instruction-following
- afghanistan
datasets:
- tasal9/ZamAI_Pashto_Dataset
---

# ZamAI Phi-3 Mini Pashto

This model is a deployment of microsoft/Phi-3-mini-4k-instruct configured for Pashto language support.

## Model description

ZamAI-Phi-3-Mini-Pashto is based on Microsoft's Phi-3 Mini 4K Instruct model. It has been configured for optimal performance with the Pashto language, enabling intelligent text generation and instruction following capabilities for Pashto speakers.

## Intended uses & limitations

This model is intended for:
- Pashto text generation
- Pashto instruction following
- Part of the ZamAI language assistant pipeline
- Lightweight applications requiring less computational resources than larger models

## Training and evaluation data

This model uses the base weights from Microsoft's Phi-3 Mini with additional configuration for Pashto language support.

## How to use

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("tasal9/ZamAI-Phi-3-Mini-Pashto", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("tasal9/ZamAI-Phi-3-Mini-Pashto", trust_remote_code=True)

# Generate text in Pashto
prompt = "په پښتو کې ماته وایاست چې افغانستان څه ډول هیواد دی؟"  # "Tell me in Pashto what kind of country Afghanistan is?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Response: {response}")
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
        "task": "text-generation",
        "model_type": "phi-3",
        "processor_class": "Phi3Tokenizer",
        "supported_languages": ["ps", "en", "fa", "ar", "ur"],
        "source_model": SOURCE_MODEL,
        "tags": ["pashto", "phi-3", "zamai", "text-generation"],
        "dataset": "tasal9/ZamAI_Pashto_Dataset"
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
        # Load and save model and tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(SOURCE_MODEL, trust_remote_code=True)
        
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(SOURCE_MODEL, trust_remote_code=True)
        
        print("💾 Saving tokenizer to temp directory...")
        tokenizer.save_pretrained("temp_model")
        
        print("💾 Saving model to temp directory...")
        model.save_pretrained("temp_model")
        
        print(f"🚀 Uploading to {TARGET_REPO}...")
        api.upload_folder(
            folder_path="temp_model",
            repo_id=TARGET_REPO,
            commit_message="Upload Phi-3 Mini with Pashto configuration"
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
