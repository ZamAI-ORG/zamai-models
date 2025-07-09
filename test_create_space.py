#!/usr/bin/env python3
"""
Quick test to create one Hugging Face Space
"""

import os
from huggingface_hub import HfApi, create_repo, upload_file
import tempfile

def test_create_space():
    # Read token
    with open('/workspaces/ZamAI-Pro-Models/HF-Token.txt', 'r') as f:
        token = f.read().strip()
    
    api = HfApi(token=token)
    space_name = "tasal9/zamai-test-space"
    
    print(f"Creating test space: {space_name}")
    
    # Create README.md content
    readme_content = """---
title: ZamAI Test Space
emoji: 🧪
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.8.0
app_file: app.py
pinned: false
license: apache-2.0
hardware: cpu-basic
---

# 🧪 ZamAI Test Space

This is a test space for ZamAI.
"""
    
    # Create app.py content
    app_content = """import gradio as gr

def greet(name):
    return f"Hello {name}! This is ZamAI test space."

demo = gr.Interface(fn=greet, inputs="text", outputs="text")

if __name__ == "__main__":
    demo.launch()
"""
    
    # Create requirements.txt
    requirements_content = """gradio>=4.8.0
"""
    
    try:
        # Create repository
        create_repo(
            repo_id=space_name,
            repo_type="space",
            space_sdk="gradio",
            token=token,
            exist_ok=True
        )
        print("✅ Repository created")
        
        # Upload files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(readme_content)
            readme_path = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(app_content)
            app_path = f.name
            
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(requirements_content)
            req_path = f.name
        
        # Upload each file
        upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=space_name,
            repo_type="space",
            token=token
        )
        print("✅ README.md uploaded")
        
        upload_file(
            path_or_fileobj=app_path,
            path_in_repo="app.py",
            repo_id=space_name,
            repo_type="space",
            token=token
        )
        print("✅ app.py uploaded")
        
        upload_file(
            path_or_fileobj=req_path,
            path_in_repo="requirements.txt",
            repo_id=space_name,
            repo_type="space",
            token=token
        )
        print("✅ requirements.txt uploaded")
        
        # Clean up temp files
        os.unlink(readme_path)
        os.unlink(app_path)
        os.unlink(req_path)
        
        print(f"🎉 Space created successfully: https://huggingface.co/spaces/{space_name}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_create_space()
