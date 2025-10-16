#!/usr/bin/env python3
"""
Check and Update Existing ZamAI Models
Analyzes current HF Hub models and creates only missing ones
"""

import os
import json
from huggingface_hub import HfApi, login, list_models

def load_hf_token():
    """Load HF token from file or environment"""
    token_file = "/workspaces/ZamAI-Pro-Models/HF-Token.txt"
    if os.path.exists(token_file):
        with open(token_file, 'r') as f:
            token = f.read().strip()
        return token
    return os.getenv("HUGGINGFACE_TOKEN")

def check_existing_models(username="tasal9"):
    """Check what models already exist on HF Hub"""
    token = load_hf_token()
    api = HfApi(token=token)
    
    try:
        # Get all models for the user
        models = list(api.list_models(author=username))
        existing_models = [model.id for model in models]
        
        print(f"🔍 Found {len(existing_models)} existing models:")
        for model in sorted(existing_models):
            print(f"   ✅ {model}")
        
        return existing_models
    
    except Exception as e:
        print(f"❌ Error checking existing models: {e}")
        return []

def get_target_models():
    """Get list of all target models we want to create"""
    return [
        "tasal9/zamai-pashto-chat-8b",
        "tasal9/zamai-translator-pashto-en", 
        "tasal9/zamai-qa-pashto",
        "tasal9/zamai-sentiment-pashto",
        "tasal9/zamai-dialogpt-pashto-v3",
        "tasal9/ZamAI-Voice-Assistant",
        "tasal9/ZamAI-Tutor-Bot",
        "tasal9/ZamAI-Business-Automation",
        "tasal9/ZamAI-Embeddings",
        "tasal9/ZamAI-STT-Processor",
        "tasal9/ZamAI-TTS-Generator"
    ]

def analyze_model_status():
    """Analyze which models exist and which are missing"""
    existing_models = check_existing_models()
    target_models = get_target_models()
    
    missing_models = []
    for target in target_models:
        if target not in existing_models:
            missing_models.append(target)
    
    print(f"\n📊 ANALYSIS RESULTS")
    print("=" * 50)
    print(f"🎯 Target models: {len(target_models)}")
    print(f"✅ Existing models: {len(existing_models)}")
    print(f"❌ Missing models: {len(missing_models)}")
    
    if missing_models:
        print(f"\n🔴 Missing models that need to be created:")
        for model in missing_models:
            print(f"   - {model}")
    else:
        print(f"\n🎉 All target models already exist!")
    
    return existing_models, missing_models

def check_model_health(model_id):
    """Check if a model is properly configured"""
    token = load_hf_token()
    api = HfApi(token=token)
    
    try:
        # Check if model has proper files
        files = api.list_repo_files(model_id, repo_type="model")
        
        has_readme = "README.md" in files
        has_config = "config.json" in files
        has_model_files = any(f.endswith(('.bin', '.safetensors', '.pt', '.pth')) for f in files)
        
        health_score = 0
        issues = []
        
        if has_readme:
            health_score += 30
        else:
            issues.append("Missing README.md")
        
        if has_config:
            health_score += 30
        else:
            issues.append("Missing config.json")
        
        if has_model_files:
            health_score += 40
        else:
            issues.append("Missing model weights")
        
        status = "🟢 Healthy" if health_score >= 90 else "🟡 Needs attention" if health_score >= 60 else "🔴 Issues"
        
        return {
            "model_id": model_id,
            "health_score": health_score,
            "status": status,
            "issues": issues,
            "files": files
        }
    
    except Exception as e:
        return {
            "model_id": model_id,
            "health_score": 0,
            "status": "🔴 Error",
            "issues": [f"Error accessing model: {e}"],
            "files": []
        }

def detailed_model_analysis():
    """Perform detailed analysis of all existing models"""
    existing_models, missing_models = analyze_model_status()
    
    if existing_models:
        print(f"\n🔍 DETAILED MODEL HEALTH CHECK")
        print("=" * 50)
        
        healthy_models = []
        unhealthy_models = []
        
        for model_id in existing_models:
            health_info = check_model_health(model_id)
            
            print(f"\n📋 {model_id}")
            print(f"   Status: {health_info['status']} ({health_info['health_score']}/100)")
            
            if health_info['issues']:
                print(f"   Issues:")
                for issue in health_info['issues']:
                    print(f"     - {issue}")
                unhealthy_models.append(health_info)
            else:
                print(f"   ✅ All checks passed")
                healthy_models.append(health_info)
        
        print(f"\n📊 HEALTH SUMMARY")
        print("=" * 30)
        print(f"🟢 Healthy models: {len(healthy_models)}")
        print(f"🔴 Unhealthy models: {len(unhealthy_models)}")
        
        if unhealthy_models:
            print(f"\n🔧 Models needing fixes:")
            for model in unhealthy_models:
                print(f"   - {model['model_id']}: {', '.join(model['issues'])}")
    
    return existing_models, missing_models

def save_analysis_report(existing_models, missing_models):
    """Save analysis report to file"""
    report = {
        "analysis_date": str(__import__('datetime').datetime.now()),
        "total_target_models": len(get_target_models()),
        "existing_models": len(existing_models),
        "missing_models": len(missing_models),
        "existing_model_list": existing_models,
        "missing_model_list": missing_models,
        "next_actions": [
            "Create missing models using create_all_models.py",
            "Fix unhealthy models if any",
            "Train models without weights",
            "Create demo spaces for all models"
        ]
    }
    
    report_path = "/workspaces/ZamAI-Pro-Models/data/processed/model_status_report.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Analysis report saved to: {report_path}")

def main():
    """Main function"""
    print("🇦🇫 ZamAI Models Status Check")
    print("=" * 50)
    
    # Load and verify token
    token = load_hf_token()
    if not token:
        print("❌ Hugging Face token not found!")
        return
    
    try:
        login(token=token)
        print("✅ Successfully connected to Hugging Face")
    except Exception as e:
        print(f"❌ Failed to connect to Hugging Face: {e}")
        return
    
    # Perform detailed analysis
    existing_models, missing_models = detailed_model_analysis()
    
    # Save report
    save_analysis_report(existing_models, missing_models)
    
    print(f"\n🎯 RECOMMENDATIONS")
    print("=" * 30)
    
    if missing_models:
        print(f"1. Run 'python create_all_models.py' to create {len(missing_models)} missing models")
    else:
        print(f"1. ✅ All target models exist")
    
    print(f"2. Check model health and fix any issues")
    print(f"3. Train models that don't have weights yet")
    print(f"4. Create demo spaces for all models")
    print(f"5. Test all models and spaces")

if __name__ == "__main__":
    main()
