#!/usr/bin/env python3
"""
Train and upload missing ZamAI models to HuggingFace Hub
"""

import os
import json
from huggingface_hub import HfApi, create_repo, upload_folder
from pathlib import Path

def read_hf_token():
    with open('/workspaces/ZamAI-Pro-Models/HF-Token.txt', 'r') as f:
        return f.read().strip()

def check_local_model_artifacts():
    """Check if we have any local model artifacts that need uploading"""
    
    print("🔍 Checking for local model artifacts...")
    
    # Check for saved model files
    model_dirs = [
        '/workspaces/ZamAI-Pro-Models/models',
        '/workspaces/ZamAI-Pro-Models/fine-tuning',
        '/workspaces/ZamAI-Pro-Models/data/processed'
    ]
    
    found_artifacts = []
    
    for base_dir in model_dirs:
        if os.path.exists(base_dir):
            for root, dirs, files in os.walk(base_dir):
                # Look for model files
                for file in files:
                    if file.endswith(('.bin', '.safetensors', '.pth', '.pt', '.onnx')):
                        found_artifacts.append({
                            'file': file,
                            'path': os.path.join(root, file),
                            'size': os.path.getsize(os.path.join(root, file))
                        })
    
    if found_artifacts:
        print(f"   📦 Found {len(found_artifacts)} model artifact files:")
        for artifact in found_artifacts:
            size_mb = artifact['size'] / (1024 * 1024)
            print(f"     • {artifact['file']} ({size_mb:.1f} MB)")
    else:
        print("   ℹ️  No local model artifacts found")
    
    return found_artifacts

def create_training_scripts():
    """Create training scripts for missing models"""
    
    missing_models = [
        {
            'name': 'zamai-pashto-chat-8b',
            'base_model': 'meta-llama/Llama-3.1-8B-Instruct',
            'task': 'conversational_ai',
            'description': 'Main Pashto conversational AI model'
        },
        {
            'name': 'zamai-translator-pashto-en',
            'base_model': 'Helsinki-NLP/opus-mt-en-mul',
            'task': 'translation',
            'description': 'Pashto-English translation model'
        },
        {
            'name': 'zamai-qa-pashto',
            'base_model': 'deepset/roberta-base-squad2',
            'task': 'question_answering',
            'description': 'Pashto question answering model'
        },
        {
            'name': 'zamai-sentiment-pashto',
            'base_model': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
            'task': 'sentiment_analysis',
            'description': 'Pashto sentiment analysis model'
        },
        {
            'name': 'zamai-dialogpt-pashto-v3',
            'base_model': 'microsoft/DialoGPT-medium',
            'task': 'dialogue_generation',
            'description': 'Advanced Pashto dialogue generation'
        }
    ]
    
    print("\\n📝 Creating training configurations for missing models...")
    
    training_dir = Path('/workspaces/ZamAI-Pro-Models/training_configs')
    training_dir.mkdir(exist_ok=True)
    
    for model_config in missing_models:
        config_file = training_dir / f"{model_config['name']}_config.json"
        
        config = {
            "model_name": f"tasal9/{model_config['name']}",
            "base_model": model_config['base_model'],
            "task": model_config['task'],
            "description": model_config['description'],
            "training_params": {
                "learning_rate": 2e-5,
                "batch_size": 4,
                "num_epochs": 3,
                "max_length": 512,
                "use_lora": True,
                "lora_r": 16,
                "lora_alpha": 32
            },
            "dataset": {
                "train_file": "data/pashto_training_data.json",
                "validation_split": 0.1
            }
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Created config: {config_file}")
    
    return missing_models

def create_model_upload_script():
    """Create a script to easily upload trained models"""
    
    upload_script = '''#!/usr/bin/env python3
"""
Upload trained model to HuggingFace Hub
Usage: python upload_model.py <model_path> <model_name>
"""

import sys
import os
from huggingface_hub import HfApi, create_repo, upload_folder
from pathlib import Path

def upload_model(model_path, model_name):
    """Upload a trained model to HF Hub"""
    
    # Read token
    with open('/workspaces/ZamAI-Pro-Models/HF-Token.txt', 'r') as f:
        token = f.read().strip()
    
    api = HfApi(token=token)
    
    print(f"📤 Uploading model: {model_name}")
    print(f"📂 From path: {model_path}")
    
    try:
        # Create repository
        create_repo(
            repo_id=model_name,
            token=token,
            exist_ok=True
        )
        print("✅ Repository created/exists")
        
        # Upload model files
        upload_folder(
            folder_path=model_path,
            repo_id=model_name,
            token=token,
            commit_message=f"Upload {model_name} model"
        )
        
        print(f"✅ Model uploaded successfully!")
        print(f"🌐 Available at: https://huggingface.co/{model_name}")
        
    except Exception as e:
        print(f"❌ Error uploading: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python upload_model.py <model_path> <model_name>")
        print("Example: python upload_model.py ./my_model tasal9/my-awesome-model")
        sys.exit(1)
    
    model_path = sys.argv[1]
    model_name = sys.argv[2]
    
    if not os.path.exists(model_path):
        print(f"❌ Model path does not exist: {model_path}")
        sys.exit(1)
    
    upload_model(model_path, model_name)
'''
    
    script_path = '/workspaces/ZamAI-Pro-Models/upload_model.py'
    with open(script_path, 'w') as f:
        f.write(upload_script)
    
    # Make it executable
    os.chmod(script_path, 0o755)
    
    print(f"\\n📤 Created model upload script: {script_path}")

def check_training_space_status():
    """Check if the training space is working"""
    
    token = read_hf_token()
    api = HfApi(token=token)
    
    training_spaces = [
        "tasal9/zamai-training-hub",
        "tasal9/ZamAI-Pashto-Education-Bot-Demo"
    ]
    
    print("\\n🔥 Checking training spaces...")
    
    for space_id in training_spaces:
        try:
            space_info = api.space_info(space_id)
            files = api.list_repo_files(space_id, repo_type="space")
            
            print(f"   🌐 {space_id}")
            print(f"      📁 Files: {len(files)}")
            print(f"      🔗 https://huggingface.co/spaces/{space_id}")
            
        except Exception as e:
            print(f"   ❌ {space_id}: {e}")

def main():
    print("🚀 ZAMAI MODEL TRAINING & UPLOAD SETUP")
    print("=" * 50)
    
    # Check for local artifacts
    artifacts = check_local_model_artifacts()
    
    # Create training configs
    missing_models = create_training_scripts()
    
    # Create upload script
    create_model_upload_script()
    
    # Check training spaces
    check_training_space_status()
    
    print("\\n" + "=" * 50)
    print("✅ SETUP COMPLETE!")
    
    print("\\n🎯 TO TRAIN MISSING MODELS:")
    print("1. Use your training space: https://huggingface.co/spaces/tasal9/zamai-training-hub")
    print("2. Or run training locally with the configs in /training_configs/")
    
    print("\\n📤 TO UPLOAD MODELS:")
    print("1. After training, use: python upload_model.py <model_path> <model_name>")
    print("2. Or upload directly from training space")
    
    print("\\n🔥 MISSING MODELS TO TRAIN:")
    for model in missing_models:
        print(f"   • tasal9/{model['name']} - {model['description']}")
    
    if artifacts:
        print("\\n💾 LOCAL ARTIFACTS FOUND:")
        print("   Check if these need to be uploaded to HF Hub")

if __name__ == "__main__":
    main()
