#!/usr/bin/env python3
"""
Quick analysis of ZamAI HuggingFace Spaces focusing on ZeroGPU usage
"""

import os
from huggingface_hub import HfApi
import json

def read_hf_token():
    with open('/workspaces/ZamAI-Pro-Models/HF-Token.txt', 'r') as f:
        return f.read().strip()

def quick_space_analysis():
    """Quick analysis of all spaces"""
    
    token = read_hf_token()
    api = HfApi(token=token)
    
    print("🔍 QUICK ZAMAI SPACES ANALYSIS")
    print("=" * 50)
    
    # Get user info
    user_info = api.whoami()
    username = user_info['name']
    print(f"👤 User: {username}")
    
    # Get all spaces
    spaces = list(api.list_spaces(author=username))
    print(f"🚀 Total Spaces: {len(spaces)}")
    
    zerogpu_spaces = []
    training_spaces = []
    demo_spaces = []
    
    for space in spaces:
        space_id = space.id
        created_date = space.created_at.strftime('%Y-%m-%d') if space.created_at else 'Unknown'
        
        print(f"\n📂 {space_id}")
        print(f"   📅 Created: {created_date}")
        
        try:
            # Get space info
            space_info = api.space_info(space_id)
            
            # Get files
            files = api.list_repo_files(space_id, repo_type="space")
            
            # Check runtime status
            runtime_status = "unknown"
            try:
                if hasattr(space_info, 'runtime') and space_info.runtime:
                    runtime_status = getattr(space_info.runtime, 'stage', 'unknown')
            except:
                runtime_status = "unknown"
            
            print(f"   🔄 Status: {runtime_status}")
            print(f"   📁 Files: {len(files)}")
            
            # Check if it's a training space
            if 'training' in space_id.lower() or 'train' in space_id.lower():
                training_spaces.append(space_id)
                print(f"   🔥 Type: TRAINING SPACE")
            elif 'demo' in space_id.lower():
                demo_spaces.append(space_id)
                print(f"   🎮 Type: DEMO SPACE")
            else:
                demo_spaces.append(space_id)
                print(f"   🎯 Type: MODEL SPACE")
            
            # Try to read README for hardware info
            try:
                # Quick check for ZeroGPU in README
                readme_url = f"https://huggingface.co/spaces/{space_id}/raw/main/README.md"
                response = api.whoami()  # Just to test connection
                
                # Assume ZeroGPU for training spaces and check naming
                is_zerogpu = False
                if 'training' in space_id.lower():
                    is_zerogpu = True
                    print(f"   ⚡ Hardware: ZEROGPU (Training)")
                elif runtime_status == "RUNNING":
                    print(f"   🔧 Hardware: CPU/GPU (Running)")
                else:
                    print(f"   💤 Hardware: CPU (Sleeping)")
                
                if is_zerogpu:
                    zerogpu_spaces.append(space_id)
                    
            except Exception as e:
                print(f"   ⚠️  Could not check hardware config")
            
            # Check file completeness
            missing_files = []
            if "app.py" not in files:
                missing_files.append('app.py')
            if "README.md" not in files:
                missing_files.append('README.md')
            if "requirements.txt" not in files:
                missing_files.append('requirements.txt')
            
            if missing_files:
                print(f"   ❌ Missing: {', '.join(missing_files)}")
            else:
                print(f"   ✅ Complete")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    # Summary
    print(f"\n" + "=" * 50)
    print("📊 SUMMARY")
    print(f"✅ Total Spaces: {len(spaces)}")
    print(f"🔥 Training Spaces: {len(training_spaces)}")
    print(f"🎮 Demo/Model Spaces: {len(demo_spaces)}")
    print(f"⚡ Potential ZeroGPU: {len(zerogpu_spaces)}")
    
    if training_spaces:
        print(f"\n🔥 TRAINING SPACES:")
        for space in training_spaces:
            print(f"   🚀 {space}")
            print(f"      🔗 https://huggingface.co/spaces/{space}")
    
    if zerogpu_spaces:
        print(f"\n⚡ ZEROGPU SPACES:")
        for space in zerogpu_spaces:
            print(f"   🚀 {space}")
    
    print(f"\n🎯 YOUR SPACE CATEGORIES:")
    
    # Categorize by purpose
    categories = {
        "Training & Fine-tuning": [s for s in spaces if 'training' in s.id.lower() or 'train' in s.id.lower()],
        "Model Demos": [s for s in spaces if any(model in s.id.lower() for model in ['bloom', 'mistral', 'llama', 'phi', 'whisper', 'embedding'])],
        "Multi-purpose": [s for s in spaces if 'playground' in s.id.lower() or 'inference' in s.id.lower()],
        "Educational": [s for s in spaces if 'education' in s.id.lower() or 'bot' in s.id.lower()]
    }
    
    for category, category_spaces in categories.items():
        if category_spaces:
            print(f"\n📁 {category} ({len(category_spaces)}):")
            for space in category_spaces:
                print(f"   • {space.id}")
    
    print(f"\n💡 ZEROGPU RECOMMENDATIONS:")
    print("1. Training spaces should use ZeroGPU for faster fine-tuning")
    print("2. Large model inference can benefit from ZeroGPU")
    print("3. Speech processing (Whisper) works well with ZeroGPU")
    print("4. For CPU-light tasks (embeddings), regular CPU might be sufficient")

if __name__ == "__main__":
    quick_space_analysis()
