#!/usr/bin/env python3
"""
Check which ZamAI models actually exist in HF Hub
"""

from huggingface_hub import HfApi
import sys

def read_hf_token():
    with open('/workspaces/ZamAI-Pro-Models/HF-Token.txt', 'r') as f:
        return f.read().strip()

def check_existing_models():
    token = read_hf_token()
    api = HfApi(token=token)
    
    # Models we want to check
    models_to_check = [
        "tasal9/ZamAI-LIama3-Pashto",
        "tasal9/pashto-base-bloom", 
        "tasal9/ZamAI-Mistral-7B-Pashto",
        "tasal9/ZamAI-Phi-3-Mini-Pashto",
        "tasal9/ZamAI-Whisper-v3-Pashto",
        "tasal9/Multilingual-ZamAI-Embeddings"
    ]
    
    existing_models = []
    missing_models = []
    
    print("🔍 Checking which models exist in HF Hub...")
    print("=" * 50)
    
    for model_id in models_to_check:
        try:
            # Try to get model info
            model_info = api.model_info(model_id)
            print(f"✅ {model_id} - EXISTS")
            existing_models.append(model_id)
        except Exception as e:
            print(f"❌ {model_id} - NOT FOUND")
            missing_models.append(model_id)
    
    print("\n" + "=" * 50)
    print(f"📊 Summary:")
    print(f"   ✅ Existing models: {len(existing_models)}")
    print(f"   ❌ Missing models: {len(missing_models)}")
    
    if existing_models:
        print(f"\n🟢 Models available for spaces:")
        for model in existing_models:
            print(f"   - {model}")
    
    if missing_models:
        print(f"\n🔴 Models NOT available (skip these):")
        for model in missing_models:
            print(f"   - {model}")
    
    return existing_models, missing_models

if __name__ == "__main__":
    existing, missing = check_existing_models()
