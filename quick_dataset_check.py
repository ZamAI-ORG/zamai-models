#!/usr/bin/env python3
"""
Quick dataset exploration for ZamAI Pashto Dataset
"""
import os
from datasets import load_dataset
from huggingface_hub import HfApi
import json

def quick_dataset_check():
    dataset_name = "tasal9/ZamAI_Pashto_Dataset"
    
    try:
        print(f"🔍 Loading dataset: {dataset_name}")
        
        # Load dataset
        dataset = load_dataset(dataset_name)
        
        print(f"✅ Dataset loaded successfully!")
        print(f"📊 Dataset info:")
        print(f"   - Keys: {list(dataset.keys())}")
        
        for split_name, split_data in dataset.items():
            print(f"   - {split_name}: {len(split_data)} examples")
            if len(split_data) > 0:
                print(f"     - Columns: {split_data.column_names}")
                print(f"     - First example keys: {list(split_data[0].keys())}")
                
                # Show first example
                first_example = split_data[0]
                print(f"     - First example:")
                for key, value in first_example.items():
                    if isinstance(value, str) and len(value) > 100:
                        print(f"       {key}: {value[:100]}...")
                    else:
                        print(f"       {key}: {value}")
                print()
        
        # Generate config
        config = {
            "dataset_name": dataset_name,
            "splits": {},
            "columns": {},
            "total_examples": 0
        }
        
        for split_name, split_data in dataset.items():
            config["splits"][split_name] = len(split_data)
            config["total_examples"] += len(split_data)
            if len(split_data) > 0:
                config["columns"][split_name] = split_data.column_names
        
        # Save config
        config_path = "dataset_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Dataset config saved to: {config_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    quick_dataset_check()
