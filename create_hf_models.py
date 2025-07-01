#!/usr/bin/env python3
import os
from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch

def create_zamai_models():
    """Create all ZamAI models on HF Hub"""
    
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        print("❌ HUGGINGFACE_TOKEN not found!")
        return
        
    api = HfApi(token=token)
    
    models = {
        "ZamAI-Voice-Assistant": "microsoft/DialoGPT-medium",
        "ZamAI-Tutor-Bot": "mistralai/Mistral-7B-Instruct-v0.2", 
        "ZamAI-Business-Automation": "microsoft/DialoGPT-large",
        "ZamAI-Embeddings": "intfloat/e5-large-v2",
        "ZamAI-STT-Processor": "openai/whisper-base",
        "ZamAI-TTS-Generator": "microsoft/speecht5_tts"
    }
    
    for model_name, base_model in models.items():
        try:
            print(f"Creating {model_name}...")
            
            # Create repository
            repo_id = f"tasal9/{model_name}"
            create_repo(repo_id, token=token, exist_ok=True)
            
            # Upload model card
            model_card_path = f"/workspaces/ZamAI-Pro-Models/model_cards/{model_name}_README.md"
            if os.path.exists(model_card_path):
                api.upload_file(
                    path_or_fileobj=model_card_path,
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    token=token
                )
            
            print(f"✅ {model_name} created successfully!")
            
        except Exception as e:
            print(f"❌ Failed to create {model_name}: {e}")

if __name__ == "__main__":
    create_zamai_models()
