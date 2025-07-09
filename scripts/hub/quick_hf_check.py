"""
Quick HF Hub Model Checker
Check what models actually exist in your HF Hub account
"""

import os

def check_hf_models_simple():
    """Simple check of HF Hub models"""
    
    print("🔍 ZamAI HF Hub Quick Check")
    print("=" * 40)
    
    # Check if HF token exists
    token_path = '/workspaces/ZamAI-Pro-Models/HF-Token.txt'
    if not os.path.exists(token_path):
        print("❌ HF Token not found")
        print("Please create HF-Token.txt with your Hugging Face token")
        print("Get your token from: https://huggingface.co/settings/tokens")
        return
    
    # Try to import and use HF libraries
    try:
        from huggingface_hub import HfApi
        
        with open(token_path, 'r') as f:
            token = f.read().strip()
        
        api = HfApi(token=token)
        
        # Get user info
        try:
            user_info = api.whoami()
            username = user_info['name']
            print(f"✅ Connected as: {username}")
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            return
        
        # List user's models
        try:
            print(f"\n🔍 Checking models for {username}...")
            models = list(api.list_models(author=username))
            
            if models:
                print(f"✅ Found {len(models)} existing models:")
                for model in models:
                    print(f"  - {model.id}")
                    if model.downloads:
                        print(f"    Downloads: {model.downloads}")
                    if model.pipeline_tag:
                        print(f"    Type: {model.pipeline_tag}")
                    print()
            else:
                print("📝 No models found in your HF Hub account yet")
                print("This is normal for a new account or if you haven't uploaded models")
        
        except Exception as e:
            print(f"❌ Error listing models: {e}")
        
        # Expected models based on your configs
        print("\n🎯 EXPECTED MODELS (from your configs):")
        expected_models = [
            "tasal9/zamai-pashto-chat-8b",
            "tasal9/ZamAI-Mistral-7B-Pashto", 
            "tasal9/ZamAI-Whisper-v3-Pashto",
            "tasal9/ZamAI-LIama3-Pashto",
            "tasal9/ZamAI-Phi-3-Mini-Pashto",
            "tasal9/pashto-base-bloom",
            "tasal9/Multilingual-ZamAI-Embeddings"
        ]
        
        existing_model_ids = [model.id for model in models] if models else []
        
        for expected in expected_models:
            if expected in existing_model_ids:
                print(f"✅ {expected} - EXISTS")
            else:
                print(f"❌ {expected} - NEEDS TRAINING")
        
        # Training recommendation
        missing_count = len([m for m in expected_models if m not in existing_model_ids])
        if missing_count > 0:
            print(f"\n🚀 TRAINING NEEDED:")
            print(f"You need to train {missing_count} models")
            print("\nStart with the priority model:")
            print("python fine-tuning/train_pashto_chat.py")
        else:
            print("\n🎉 All expected models exist!")
        
    except ImportError:
        print("❌ huggingface_hub not installed")
        print("Run: pip install huggingface_hub")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    check_hf_models_simple()
