#!/usr/bin/env python3
"""
Check ZamAI Hugging Face Setup Status
"""

from huggingface_hub import HfApi, list_models, list_spaces
import json

def read_hf_token():
    with open('/workspaces/ZamAI-Pro-Models/HF-Token.txt', 'r') as f:
        return f.read().strip()

def check_setup_status():
    token = read_hf_token()
    api = HfApi(token=token)
    
    print("🤗 ZamAI Hugging Face Hub Status")
    print("=" * 50)
    
    # Get user info
    user_info = api.whoami()
    username = user_info['name']
    print(f"👤 User: {username}")
    
    # Check models
    print("\\n📦 MODELS:")
    models = list(list_models(author=username, use_auth_token=token))
    print(f"Total Models: {len(models)}")
    
    for model in models:
        print(f"  ✅ {model.id}")
        print(f"     📅 Created: {model.created_at.strftime('%Y-%m-%d')}")
        print(f"     💾 Downloads: {model.downloads}")
    
    # Check spaces  
    print("\\n🚀 SPACES:")
    try:
        spaces = list(api.list_spaces(author=username))
        print(f"Total Spaces: {len(spaces)}")
        
        for space in spaces:
            print(f"  🌐 {space.id}")
            print(f"     📅 Created: {space.created_at.strftime('%Y-%m-%d')}")
            print(f"     🔗 https://huggingface.co/spaces/{space.id}")
    except Exception as e:
        print(f"Error checking spaces: {e}")
    
    # Expected spaces
    expected_spaces = [
        "zamai-pashto-chat",
        "zamai-bloom-pashto", 
        "zamai-mistral-pashto",
        "zamai-phi3-business",
        "zamai-whisper-speech",
        "zamai-embeddings"
    ]
    
    print("\\n📋 EXPECTED SPACES:")
    for space_name in expected_spaces:
        full_name = f"{username}/{space_name}"
        print(f"  📍 {full_name}")
    
    print("\\n🎯 QUICK ACCESS LINKS:")
    print("Models:")
    for model in models:
        print(f"  🔗 https://huggingface.co/{model.id}")
    
    print("\\nSpaces (Expected):")
    for space_name in expected_spaces:
        print(f"  🔗 https://huggingface.co/spaces/{username}/{space_name}")

if __name__ == "__main__":
    check_setup_status()
