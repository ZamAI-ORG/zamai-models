#!/usr/bin/env python3
"""
Comprehensive audit of ZamAI Hugging Face setup
- Check existing models and spaces
- Identify what needs to be uploaded vs what's already there
- Check for local models that aren't uploaded
- Audit spaces for functionality
"""

import os
from huggingface_hub import HfApi, list_models, list_spaces
import json
from pathlib import Path

def read_hf_token():
    with open('/workspaces/ZamAI-Pro-Models/HF-Token.txt', 'r') as f:
        return f.read().strip()

def check_local_models():
    """Check for local model files that might need uploading"""
    local_models = []
    
    # Check common model directories
    model_dirs = [
        '/workspaces/ZamAI-Pro-Models/models',
        '/workspaces/ZamAI-Pro-Models/fine-tuning',
        '/workspaces/ZamAI-Pro-Models/zerogpu_files'
    ]
    
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            for root, dirs, files in os.walk(model_dir):
                # Look for model files
                model_files = [f for f in files if f.endswith(('.bin', '.safetensors', '.pth', '.pt', '.json'))]
                if model_files:
                    local_models.append({
                        'path': root,
                        'files': model_files
                    })
    
    return local_models

def audit_space_functionality(space_id, token):
    """Check if a space is working properly"""
    api = HfApi(token=token)
    
    try:
        # Get space info
        space_info = api.space_info(space_id)
        
        # Check if space has required files
        files = api.list_repo_files(space_id, repo_type="space")
        
        has_readme = "README.md" in files
        has_app = "app.py" in files
        has_requirements = "requirements.txt" in files
        
        # Safely get runtime info
        runtime_info = 'unknown'
        try:
            if hasattr(space_info, 'runtime') and space_info.runtime:
                runtime_info = getattr(space_info.runtime, 'stage', 'unknown')
        except:
            runtime_info = 'unknown'
        
        status = {
            'exists': True,
            'has_readme': has_readme,
            'has_app': has_app, 
            'has_requirements': has_requirements,
            'runtime': runtime_info,
            'sdk': getattr(space_info, 'sdk', 'unknown'),
            'files': files
        }
        
        return status
        
    except Exception as e:
        return {
            'exists': False,
            'error': str(e)
        }

def comprehensive_audit():
    token = read_hf_token()
    api = HfApi(token=token)
    
    print("🔍 COMPREHENSIVE ZAMAI HUGGING FACE AUDIT")
    print("=" * 60)
    
    # Get user info
    user_info = api.whoami()
    username = user_info['name']
    print(f"👤 User: {username}")
    
    # Check all models
    print(f"\n📦 EXISTING MODELS ON HUB:")
    models = list(list_models(author=username, token=token))
    
    existing_models = {}
    for model in models:
        print(f"  ✅ {model.id}")
        created_date = model.created_at.strftime('%Y-%m-%d') if model.created_at else 'Unknown'
        print(f"     📅 Created: {created_date}")
        print(f"     💾 Downloads: {model.downloads}")
        print(f"     ❤️  Likes: {model.likes}")
        
        existing_models[model.id] = {
            'created': model.created_at,
            'downloads': model.downloads,
            'likes': model.likes
        }
    
    # Check all spaces
    print(f"\n🚀 EXISTING SPACES ON HUB:")
    spaces = list(api.list_spaces(author=username))
    
    existing_spaces = {}
    for space in spaces:
        print(f"  🌐 {space.id}")
        space_created = space.created_at.strftime('%Y-%m-%d') if space.created_at else 'Unknown'
        print(f"     📅 Created: {space_created}")
        
        # Check space functionality
        space_status = audit_space_functionality(space.id, token)
        existing_spaces[space.id] = space_status
        
        if space_status['exists']:
            print(f"     📱 SDK: {space_status['sdk']}")
            print(f"     🔧 Status: {space_status['runtime']}")
            print(f"     📁 Files: {len(space_status['files'])}")
            
            # Check for issues
            issues = []
            if not space_status['has_app']:
                issues.append("Missing app.py")
            if not space_status['has_readme']:
                issues.append("Missing README.md")
            if not space_status['has_requirements']:
                issues.append("Missing requirements.txt")
                
            if issues:
                print(f"     ⚠️  Issues: {', '.join(issues)}")
            else:
                print(f"     ✅ All files present")
        else:
            print(f"     ❌ Error: {space_status['error']}")
    
    # Check local models
    print(f"\n💾 LOCAL MODEL FILES:")
    local_models = check_local_models()
    
    if local_models:
        for model in local_models:
            print(f"  📂 {model['path']}")
            print(f"     📄 Files: {', '.join(model['files'][:5])}{'...' if len(model['files']) > 5 else ''}")
    else:
        print("  ℹ️  No local model files found")
    
    # Check for models mentioned in configs but not on hub
    print(f"\n🔍 MISSING MODELS ANALYSIS:")
    
    # Expected models from your project
    expected_models = [
        "tasal9/zamai-pashto-chat-8b",
        "tasal9/zamai-translator-pashto-en", 
        "tasal9/zamai-qa-pashto",
        "tasal9/zamai-sentiment-pashto",
        "tasal9/zamai-dialogpt-pashto-v3"
    ]
    
    missing_models = []
    for expected in expected_models:
        if expected not in existing_models:
            missing_models.append(expected)
            print(f"  ❌ Missing: {expected}")
    
    if not missing_models:
        print("  ✅ All expected models found")
    
    # Identify problematic spaces
    print(f"\n⚠️  SPACES NEEDING ATTENTION:")
    problem_spaces = []
    
    for space_id, status in existing_spaces.items():
        if not status['exists']:
            problem_spaces.append((space_id, "Space not accessible"))
        elif not status['has_app']:
            problem_spaces.append((space_id, "Missing app.py"))
        elif not status['has_readme']:
            problem_spaces.append((space_id, "Missing README.md"))
    
    if problem_spaces:
        for space_id, issue in problem_spaces:
            print(f"  🔧 {space_id}: {issue}")
    else:
        print("  ✅ All spaces appear functional")
    
    # Summary and recommendations
    print(f"\n📋 AUDIT SUMMARY:")
    print(f"  📦 Models on Hub: {len(existing_models)}")
    print(f"  🚀 Spaces on Hub: {len(existing_spaces)}")
    print(f"  💾 Local model directories: {len(local_models)}")
    print(f"  ❌ Missing expected models: {len(missing_models)}")
    print(f"  ⚠️  Problematic spaces: {len(problem_spaces)}")
    
    print(f"\n🎯 RECOMMENDED ACTIONS:")
    
    if missing_models:
        print("  1. 🔄 Train and upload missing models:")
        for model in missing_models:
            print(f"     - {model}")
    
    if problem_spaces:
        print("  2. 🔧 Fix problematic spaces:")
        for space_id, issue in problem_spaces:
            print(f"     - {space_id}: {issue}")
    
    if local_models:
        print("  3. 📤 Check if local models need uploading")
    
    print("  4. 🧹 Clean up any duplicate spaces")
    print("  5. ✅ Verify all spaces are working correctly")
    
    # Save audit results
    audit_results = {
        'timestamp': '2025-07-09',
        'existing_models': existing_models,
        'existing_spaces': existing_spaces,
        'local_models': local_models,
        'missing_models': missing_models,
        'problem_spaces': problem_spaces
    }
    
    with open('/workspaces/ZamAI-Pro-Models/data/processed/hf_audit_results.json', 'w') as f:
        json.dump(audit_results, f, indent=2, default=str)
    
    print(f"\n💾 Audit results saved to: data/processed/hf_audit_results.json")

if __name__ == "__main__":
    comprehensive_audit()
