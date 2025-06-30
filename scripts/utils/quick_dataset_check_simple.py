#!/usr/bin/env python3
"""
Quick Dataset Check - Simple version
"""

import json
from huggingface_hub import HfApi, login
import os

def quick_check():
    print("🇦🇫 Quick ZamAI Dataset Check")
    print("=" * 40)
    
    # Login
    token_path = "HF-Token.txt"
    if os.path.exists(token_path):
        with open(token_path, 'r') as f:
            token = f.read().strip()
        login(token=token)
        print("🔑 Logged in")
    
    # Check dataset
    api = HfApi()
    dataset_name = "tasal9/ZamAI_Pashto_Dataset"
    
    try:
        dataset_info = api.dataset_info(dataset_name)
        print(f"✅ Dataset: {dataset_name}")
        print(f"📊 Private: {dataset_info.private}")
        print(f"📈 Downloads: {dataset_info.downloads}")
        
        # List files
        files = api.list_repo_files(dataset_name, repo_type="dataset")
        print(f"📁 Files ({len(files)}):")
        
        training_files = []
        for file in files:
            print(f"   - {file}")
            if "instruction.jsonl" in file:
                training_files.append(file)
        
        print(f"\n🎯 Training files found: {training_files}")
        
        # Save quick config
        quick_config = {
            "dataset_name": dataset_name,
            "available_files": files,
            "training_files": training_files,
            "recommended_train": "pashto_train_instruction.jsonl",
            "recommended_val": "pashto_val_instruction.jsonl"
        }
        
        with open("quick_dataset_check.json", 'w') as f:
            json.dump(quick_config, f, indent=2)
        
        print(f"💾 Saved: quick_dataset_check.json")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    quick_check()
