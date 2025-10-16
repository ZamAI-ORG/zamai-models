#!/usr/bin/env python3
"""
Create ZamAI Training Hub Space with ZeroGPU
"""

import os
import shutil
from huggingface_hub import HfApi, create_repo, upload_folder
import tempfile

def create_training_space():
    # Read token
    with open('/workspaces/ZamAI-Pro-Models/HF-Token.txt', 'r') as f:
        token = f.read().strip()
    
    api = HfApi(token=token)
    space_name = "tasal9/zamai-training-hub"
    
    print(f"🔥 Creating training space: {space_name}")
    
    try:
        # Create repository
        create_repo(
            repo_id=space_name,
            repo_type="space",
            space_sdk="gradio",
            token=token,
            exist_ok=True
        )
        print("✅ Training space repository created")
        
        # Copy zerogpu files to temp directory
        source_dir = "/workspaces/ZamAI-Pro-Models/zerogpu_files"
        temp_dir = tempfile.mkdtemp()
        
        # Copy all files
        for item in os.listdir(source_dir):
            src_path = os.path.join(source_dir, item)
            dst_path = os.path.join(temp_dir, item)
            
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path)
            else:
                shutil.copy2(src_path, dst_path)
        
        print("✅ Files prepared for upload")
        
        # Upload folder
        upload_folder(
            folder_path=temp_dir,
            repo_id=space_name,
            repo_type="space",
            token=token,
            commit_message="Setup ZamAI training hub with ZeroGPU infrastructure"
        )
        
        print("✅ Files uploaded successfully")
        print(f"🌐 Training space: https://huggingface.co/spaces/{space_name}")
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        return space_name
        
    except Exception as e:
        print(f"❌ Error creating training space: {e}")
        return None

if __name__ == "__main__":
    print("🚀 Setting up ZamAI Training Hub")
    print("=" * 40)
    
    result = create_training_space()
    
    if result:
        print("\\n✅ Training Hub Setup Complete!")
        print("🎯 You can now:")
        print("1. Visit the training space to fine-tune models")
        print("2. Use ZeroGPU for GPU-accelerated training")
        print("3. Monitor training progress in real-time")
        print("4. Automatically upload trained models to HF Hub")
