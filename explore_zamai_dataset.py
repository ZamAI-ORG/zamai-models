#!/usr/bin/env python3
"""
Load and explore ZamAI Pashto Dataset files
"""
import pandas as pd
import json
from huggingface_hub import hf_hub_download, login
import os

def explore_dataset_files():
    # Login with HF token
    token_path = "HF-Token.txt"
    if os.path.exists(token_path):
        with open(token_path, 'r') as f:
            token = f.read().strip()
        login(token=token)
    
    dataset_name = "tasal9/ZamAI_Pashto_Dataset"
    
    print(f"🇦🇫 Exploring ZamAI Pashto Dataset Files")
    print("=" * 50)
    
    # Download and examine different file types
    files_to_check = [
        "pashto_train_instruction.jsonl",
        "pashto_val_instruction.jsonl", 
        "pashto_cleaned_train.csv",
        "dataset_info.txt"
    ]
    
    dataset_info = {}
    
    for filename in files_to_check:
        try:
            print(f"\n📁 Examining: {filename}")
            print("-" * 30)
            
            # Download file
            file_path = hf_hub_download(
                repo_id=dataset_name,
                filename=filename,
                repo_type="dataset"
            )
            
            if filename.endswith('.jsonl'):
                # Read JSONL file
                data = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f):
                        if line_num >= 5:  # Only read first 5 lines
                            break
                        try:
                            data.append(json.loads(line.strip()))
                        except:
                            continue
                
                print(f"   📊 Format: JSONL")
                print(f"   📝 Sample entries: {len(data)}")
                
                if data:
                    print(f"   🔑 Keys: {list(data[0].keys())}")
                    print(f"   📖 First example:")
                    for key, value in data[0].items():
                        if isinstance(value, str) and len(value) > 100:
                            print(f"      {key}: {value[:100]}...")
                        else:
                            print(f"      {key}: {value}")
                
                dataset_info[filename] = {
                    "format": "jsonl",
                    "sample_count": len(data),
                    "keys": list(data[0].keys()) if data else []
                }
            
            elif filename.endswith('.csv'):
                # Read CSV file
                df = pd.read_csv(file_path)
                print(f"   📊 Format: CSV")
                print(f"   📏 Shape: {df.shape}")
                print(f"   🔑 Columns: {list(df.columns)}")
                print(f"   📖 First row:")
                for col in df.columns:
                    value = str(df.iloc[0][col])
                    if len(value) > 100:
                        print(f"      {col}: {value[:100]}...")
                    else:
                        print(f"      {col}: {value}")
                
                dataset_info[filename] = {
                    "format": "csv",
                    "shape": df.shape,
                    "columns": list(df.columns)
                }
            
            elif filename.endswith('.txt'):
                # Read text file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print(f"   📊 Format: Text")
                    print(f"   📏 Length: {len(content)} characters")
                    print(f"   📖 Content preview:")
                    print(f"      {content[:500]}...")
                
                dataset_info[filename] = {
                    "format": "text",
                    "length": len(content)
                }
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Save dataset exploration results
    config = {
        "dataset_name": dataset_name,
        "files_explored": dataset_info,
        "recommended_format": "instruction_jsonl",
        "training_files": {
            "train": "pashto_train_instruction.jsonl",
            "validation": "pashto_val_instruction.jsonl"
        }
    }
    
    with open("zamai_dataset_config.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Dataset exploration saved to: zamai_dataset_config.json")
    print(f"\n🎯 Recommended training setup:")
    print(f"   - Use instruction format JSONL files")
    print(f"   - Train: pashto_train_instruction.jsonl") 
    print(f"   - Validation: pashto_val_instruction.jsonl")

if __name__ == "__main__":
    explore_dataset_files()
