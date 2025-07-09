#!/usr/bin/env python3
"""
Update existing spaces to use ZeroGPU based on user preference
"""

import os
from huggingface_hub import HfApi, upload_file
import tempfile

def read_hf_token():
    with open('/workspaces/ZamAI-Pro-Models/HF-Token.txt', 'r') as f:
        return f.read().strip()

def update_space_for_zerogpu(space_id, token):
    """Update a space to use ZeroGPU hardware and decorators"""
    
    print(f"🔄 Updating {space_id} for ZeroGPU...")
    
    try:
        api = HfApi(token=token)
        
        # Check current files
        files = api.list_repo_files(space_id, repo_type="space")
        
        # Read current README
        readme_path = api.hf_hub_download(
            repo_id=space_id,
            filename="README.md",
            repo_type="space",
            token=token
        )
        
        with open(readme_path, 'r', encoding='utf-8') as f:
            readme_content = f.read()
        
        # Update README to use zero-a10g hardware
        if "hardware:" in readme_content:
            # Replace existing hardware line
            lines = readme_content.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('hardware:'):
                    lines[i] = 'hardware: zero-a10g'
                    break
            readme_content = '\n'.join(lines)
        else:
            # Add hardware line after license
            lines = readme_content.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('license:'):
                    lines.insert(i + 1, 'hardware: zero-a10g')
                    break
            readme_content = '\n'.join(lines)
        
        # Upload updated README
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(readme_content)
            readme_temp = f.name
        
        upload_file(
            path_or_fileobj=readme_temp,
            path_in_repo="README.md",
            repo_id=space_id,
            repo_type="space",
            token=token,
            commit_message="Update to use ZeroGPU hardware"
        )
        
        # Clean up
        os.unlink(readme_temp)
        
        print(f"   ✅ Updated README with ZeroGPU hardware")
        
        # Update app.py if it exists and doesn't have spaces decorators
        if "app.py" in files:
            try:
                app_path = api.hf_hub_download(
                    repo_id=space_id,
                    filename="app.py",
                    repo_type="space",
                    token=token
                )
                
                with open(app_path, 'r', encoding='utf-8') as f:
                    app_content = f.read()
                
                # Check if already has spaces import and decorators
                if "import spaces" not in app_content and "@spaces.GPU" not in app_content:
                    # Add spaces import
                    if "import gradio as gr" in app_content:
                        app_content = app_content.replace(
                            "import gradio as gr",
                            "import gradio as gr\\nimport spaces"
                        )
                    
                    # Add @spaces.GPU decorator to main functions
                    # This is a basic addition - manual review recommended
                    if "def generate" in app_content or "def transcribe" in app_content or "def process" in app_content:
                        app_lines = app_content.split('\\n')
                        new_lines = []
                        
                        for i, line in enumerate(app_lines):
                            new_lines.append(line)
                            if line.strip().startswith('def ') and any(keyword in line for keyword in ['generate', 'transcribe', 'process', 'predict']):
                                # Add decorator before function
                                new_lines.insert(-1, '@spaces.GPU')
                        
                        app_content = '\\n'.join(new_lines)
                    
                    # Upload updated app.py
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                        f.write(app_content)
                        app_temp = f.name
                    
                    upload_file(
                        path_or_fileobj=app_temp,
                        path_in_repo="app.py",
                        repo_id=space_id,
                        repo_type="space",
                        token=token,
                        commit_message="Add ZeroGPU decorators for better performance"
                    )
                    
                    os.unlink(app_temp)
                    print(f"   ✅ Updated app.py with ZeroGPU decorators")
                else:
                    print(f"   ℹ️  app.py already has ZeroGPU support")
                    
            except Exception as e:
                print(f"   ⚠️  Could not update app.py: {e}")
        
        # Update requirements.txt to include spaces
        if "requirements.txt" in files:
            try:
                req_path = api.hf_hub_download(
                    repo_id=space_id,
                    filename="requirements.txt",
                    repo_type="space",
                    token=token
                )
                
                with open(req_path, 'r') as f:
                    req_content = f.read()
                
                if "spaces" not in req_content:
                    req_content += "\\nspaces\\n"
                    
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                        f.write(req_content)
                        req_temp = f.name
                    
                    upload_file(
                        path_or_fileobj=req_temp,
                        path_in_repo="requirements.txt",
                        repo_id=space_id,
                        repo_type="space",
                        token=token,
                        commit_message="Add spaces dependency for ZeroGPU"
                    )
                    
                    os.unlink(req_temp)
                    print(f"   ✅ Updated requirements.txt with spaces dependency")
                else:
                    print(f"   ℹ️  requirements.txt already has spaces dependency")
                    
            except Exception as e:
                print(f"   ⚠️  Could not update requirements.txt: {e}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error updating {space_id}: {e}")
        return False

def main():
    """Update key spaces to use ZeroGPU"""
    
    print("🚀 UPDATING ZAMAI SPACES FOR ZEROGPU")
    print("=" * 50)
    
    token = read_hf_token()
    api = HfApi(token=token)
    
    # Get user spaces
    spaces = list(api.list_spaces(author="tasal9"))
    
    # Priority spaces to update (training and inference heavy)
    priority_spaces = [
        "tasal9/zamai-training-hub",
        "tasal9/zamai-whisper-speech", 
        "tasal9/zamai-mistral-pashto",
        "tasal9/zamai-pashto-chat",
        "tasal9/ZamAI-Pashto-Multimodel-AI-Playground",
        "tasal9/HF-Inference"
    ]
    
    print(f"🎯 Priority spaces for ZeroGPU upgrade:")
    for space_id in priority_spaces:
        if any(s.id == space_id for s in spaces):
            print(f"   • {space_id}")
    
    updated_count = 0
    
    for space_id in priority_spaces:
        if any(s.id == space_id for s in spaces):
            if update_space_for_zerogpu(space_id, token):
                updated_count += 1
    
    print(f"\\n✅ Updated {updated_count} spaces for ZeroGPU")
    
    print(f"\\n🎯 ZEROGPU BENEFITS:")
    print("• Faster model loading and inference")
    print("• Better performance for large models") 
    print("• Improved user experience")
    print("• Support for GPU-intensive tasks")
    
    print(f"\\n📋 RECOMMENDED ZEROGPU USAGE:")
    print("• Training spaces: zero-a10g (GPU training)")
    print("• Large text models: zero-a10g (faster inference)")
    print("• Speech processing: zero-a10g (Whisper models)")
    print("• Embeddings: cpu-upgrade (sufficient for most cases)")

if __name__ == "__main__":
    main()
