#!/usr/bin/env python3
"""
Quick test to see if spaces were fixed
"""

from huggingface_hub import HfApi

def read_hf_token():
    with open('/workspaces/ZamAI-Pro-Models/HF-Token.txt', 'r') as f:
        return f.read().strip()

def check_space_fixed(space_id):
    """Check if a space has been fixed"""
    token = read_hf_token()
    api = HfApi(token=token)
    
    print(f"\n🔍 Checking {space_id}...")
    
    try:
        # Download app.py
        app_content = api.hf_hub_download(
            repo_id=space_id,
            filename="app.py",
            repo_type="space"
        )
        
        with open(app_content, 'r') as f:
            content = f.read()
        
        has_spaces_import = "import spaces" in content
        has_gpu_decorator = "@spaces.GPU" in content
        
        print(f"   - Has 'import spaces': {has_spaces_import}")
        print(f"   - Has '@spaces.GPU': {has_gpu_decorator}")
        
        # Check requirements.txt
        try:
            req_content = api.hf_hub_download(
                repo_id=space_id,
                filename="requirements.txt",
                repo_type="space"
            )
            
            with open(req_content, 'r') as f:
                req_text = f.read()
            
            has_spaces_requirement = "spaces" in req_text
            print(f"   - Has 'spaces' in requirements: {has_spaces_requirement}")
            
        except Exception as e:
            print(f"   - Could not check requirements.txt: {e}")
        
        return has_spaces_import and has_gpu_decorator
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

# Test a few spaces
test_spaces = [
    "tasal9/zamai-bloom-pashto",
    "tasal9/zamai-embeddings-multilingual",
    "tasal9/zamai-whisper-speech"
]

print("🧪 Testing if spaces were fixed...")
for space in test_spaces:
    check_space_fixed(space)
