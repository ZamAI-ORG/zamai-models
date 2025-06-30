#!/usr/bin/env python3
"""
Check and explore ZamAI Pashto Dataset
"""

from datasets import load_dataset
from huggingface_hub import HfApi
import json

def explore_dataset():
    print("🇦🇫 Exploring ZamAI Pashto Dataset")
    print("=" * 50)
    
    try:
        # Load your dataset
        dataset = load_dataset("tasal9/ZamAI_Pashto_Dataset")
        
        print(f"📊 Dataset loaded successfully!")
        print(f"📁 Splits available: {list(dataset.keys())}")
        
        # Check each split
        for split_name, split_data in dataset.items():
            print(f"\n📋 Split: {split_name}")
            print(f"   📊 Size: {len(split_data)} examples")
            print(f"   🏷️  Features: {split_data.features}")
            
            # Show first few examples
            if len(split_data) > 0:
                print(f"   📝 First example:")
                example = split_data[0]
                for key, value in example.items():
                    if isinstance(value, str) and len(value) > 100:
                        print(f"      {key}: {value[:100]}...")
                    else:
                        print(f"      {key}: {value}")
        
        return dataset
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

def create_dataset_config():
    """Create configuration for the dataset"""
    dataset_config = {
        "dataset_info": {
            "name": "ZamAI_Pashto_Dataset",
            "dataset_id": "tasal9/ZamAI_Pashto_Dataset",
            "description": "Custom Pashto language dataset for ZamAI training",
            "language": "Pashto (ps)",
            "use_cases": [
                "Chat model training",
                "Text generation",
                "Conversation fine-tuning",
                "Pashto language modeling"
            ]
        },
        "training_configs": {
            "chat_model": {
                "dataset_path": "tasal9/ZamAI_Pashto_Dataset",
                "text_column": "text",  # Adjust based on actual column name
                "target_column": "response",  # Adjust based on actual column name
                "max_length": 2048,
                "preprocessing": {
                    "add_special_tokens": True,
                    "format_as_chat": True,
                    "system_prompt": "تاسو د پښتو ژبې یو ماهر مرستیال یاست."
                }
            },
            "text_generation": {
                "dataset_path": "tasal9/ZamAI_Pashto_Dataset", 
                "text_column": "text",
                "max_length": 1024,
                "preprocessing": {
                    "add_special_tokens": False,
                    "format_as_chat": False
                }
            }
        }
    }
    
    # Save configuration
    with open('dataset_config.json', 'w', encoding='utf-8') as f:
        json.dump(dataset_config, f, indent=2, ensure_ascii=False)
    
    print("💾 Dataset configuration saved to: dataset_config.json")
    return dataset_config

if __name__ == "__main__":
    dataset = explore_dataset()
    if dataset:
        create_dataset_config()
        print("\n🎯 Dataset ready for training!")
    else:
        print("\n❌ Dataset exploration failed. Please check dataset availability.")
