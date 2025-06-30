#!/usr/bin/env python3
"""
Check dataset accessibility and metadata
"""
from huggingface_hub import HfApi, login
import os

def check_dataset_access():
    # Read HF token
    token_path = "HF-Token.txt"
    if os.path.exists(token_path):
        with open(token_path, 'r') as f:
            token = f.read().strip()
        print(f"🔑 Using HF token: {token[:10]}...")
        login(token=token)
    else:
        print("⚠️  No HF token found")
    
    api = HfApi()
    dataset_name = "tasal9/ZamAI_Pashto_Dataset"
    
    try:
        print(f"🔍 Checking dataset: {dataset_name}")
        
        # Get dataset info
        dataset_info = api.dataset_info(dataset_name)
        
        print(f"✅ Dataset found!")
        print(f"📊 Dataset info:")
        print(f"   - ID: {dataset_info.id}")
        print(f"   - Private: {dataset_info.private}")
        print(f"   - Downloads: {dataset_info.downloads}")
        print(f"   - Tags: {dataset_info.tags}")
        
        # List files
        try:
            files = api.list_repo_files(dataset_name, repo_type="dataset")
            print(f"📁 Files ({len(files)}):")
            for file in files[:10]:  # Show first 10 files
                print(f"   - {file}")
            if len(files) > 10:
                print(f"   ... and {len(files) - 10} more files")
        except Exception as e:
            print(f"❌ Error listing files: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error accessing dataset: {e}")
        return False

if __name__ == "__main__":
    check_dataset_access()
