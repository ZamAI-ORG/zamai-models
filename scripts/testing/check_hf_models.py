#!/usr/bin/env python3
"""
Check and list models on Hugging Face Hub for ZamAI
"""

import os
from huggingface_hub import HfApi, list_models
import json

def check_hf_models():
    # Read the token
    with open('/workspaces/ZamAI-App/huggingface-models/HF-Token.txt', 'r') as f:
        token = f.read().strip()
    
    # Initialize HF API
    api = HfApi(token=token)
    
    # Get user info
    try:
        user_info = api.whoami()
        username = user_info['name']
        print(f"🤗 Hugging Face User: {username}")
        print("=" * 50)
        
        # List user's models
        models = list_models(author=username, use_auth_token=token)
        models_list = list(models)
        
        print(f"📊 Total Models Found: {len(models_list)}")
        print("\n🏷️ Your Models:")
        print("-" * 30)
        
        model_categories = {
            'chat': [],
            'translation': [],
            'text_generation': [],
            'sentiment': [],
            'qa': [],
            'other': []
        }
        
        for model in models_list:
            model_id = model.id
            print(f"📦 {model_id}")
            print(f"   📅 Created: {model.created_at}")
            print(f"   💾 Downloads: {model.downloads}")
            print(f"   ❤️  Likes: {model.likes}")
            
            # Try to get model info
            try:
                model_info = api.model_info(model_id)
                if hasattr(model_info, 'tags') and model_info.tags:
                    print(f"   🏷️  Tags: {', '.join(model_info.tags[:5])}")
                if hasattr(model_info, 'pipeline_tag') and model_info.pipeline_tag:
                    print(f"   🔧 Task: {model_info.pipeline_tag}")
            except:
                pass
            
            # Categorize models
            model_name_lower = model_id.lower()
            if 'chat' in model_name_lower or 'conversation' in model_name_lower:
                model_categories['chat'].append(model_id)
            elif 'translation' in model_name_lower or 'translate' in model_name_lower:
                model_categories['translation'].append(model_id)
            elif 'generation' in model_name_lower or 'gpt' in model_name_lower:
                model_categories['text_generation'].append(model_id)
            elif 'sentiment' in model_name_lower or 'emotion' in model_name_lower:
                model_categories['sentiment'].append(model_id)
            elif 'qa' in model_name_lower or 'question' in model_name_lower:
                model_categories['qa'].append(model_id)
            else:
                model_categories['other'].append(model_id)
            
            print("-" * 30)
        
        # Save model inventory
        inventory = {
            'username': username,
            'total_models': len(models_list),
            'categories': model_categories,
            'models': [
                {
                    'id': m.id,
                    'created_at': str(m.created_at),
                    'downloads': m.downloads,
                    'likes': m.likes
                } for m in models_list
            ]
        }
        
        with open('/workspaces/ZamAI-App/huggingface-models/model_inventory.json', 'w') as f:
            json.dump(inventory, f, indent=2, ensure_ascii=False)
        
        print("\n📋 Model Categories:")
        for category, models in model_categories.items():
            if models:
                print(f"  {category.upper()}: {len(models)} models")
                for model in models:
                    print(f"    - {model}")
        
        print(f"\n💾 Inventory saved to: model_inventory.json")
        
        return inventory
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    check_hf_models()
