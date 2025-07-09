#!/usr/bin/env python3
"""
Upload trained model to HuggingFace Hub
Usage: python upload_model.py <model_path> <model_name>
"""

import sys
import os
from huggingface_hub import HfApi, create_repo, upload_folder
from pathlib import Path

def upload_model(model_path, model_name):
    """Upload a trained model to HF Hub"""
    
    # Read token
    with open('/workspaces/ZamAI-Pro-Models/HF-Token.txt', 'r') as f:
        token = f.read().strip()
    
    api = HfApi(token=token)
    
    print(f"📤 Uploading model: {model_name}")
    print(f"📂 From path: {model_path}")
    
    try:
        # Create repository
        create_repo(
            repo_id=model_name,
            token=token,
            exist_ok=True
        )
        print("✅ Repository created/exists")
        
        # Upload model files
        upload_folder(
            folder_path=model_path,
            repo_id=model_name,
            token=token,
            commit_message=f"Upload {model_name} model"
        )
        
        print(f"✅ Model uploaded successfully!")
        print(f"🌐 Available at: https://huggingface.co/{model_name}")
        
    except Exception as e:
        print(f"❌ Error uploading: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python upload_model.py <model_path> <model_name>")
        print("Example: python upload_model.py ./my_model tasal9/my-awesome-model")
        sys.exit(1)
    
    model_path = sys.argv[1]
    model_name = sys.argv[2]
    
    if not os.path.exists(model_path):
        print(f"❌ Model path does not exist: {model_path}")
        sys.exit(1)
    
    upload_model(model_path, model_name)
