
from huggingface_hub import hf_hub_download
import pandas as pd
import json
import os

dataset_name = "tasal9/ZamAI_Pashto_Dataset"
files_to_check = [
    "pashto_train_prompt_completion.jsonl",
    "pashto_cleaned_train.csv"
]

print(f"🔍 Downloading and inspecting files from: {dataset_name}")

for filename in files_to_check:
    try:
        print(f"\n--- Inspecting {filename} ---")
        local_path = hf_hub_download(repo_id=dataset_name, filename=filename, repo_type="dataset")
        print(f"Downloaded to: {local_path}")
        
        if filename.endswith(".jsonl"):
            with open(local_path, 'r', encoding='utf-8') as f:
                print("First 3 lines:")
                for i in range(3):
                    line = f.readline()
                    if not line: break
                    data = json.loads(line)
                    print(f"Line {i+1} keys: {list(data.keys())}")
                    # print(f"Line {i+1} content: {str(data)[:100]}...")
                
                # Check for consistency
                f.seek(0)
                keys_set = set()
                for i, line in enumerate(f):
                    if not line.strip(): continue
                    try:
                        data = json.loads(line)
                        keys = tuple(sorted(data.keys()))
                        keys_set.add(keys)
                        if len(keys_set) > 1:
                            print(f"⚠️  Inconsistency found at line {i+1}!")
                            print(f"   Found keys: {keys}")
                            print(f"   Previous keys: {list(keys_set)}")
                            break
                    except json.JSONDecodeError:
                        print(f"❌ JSON Decode Error at line {i+1}")
                
                if len(keys_set) == 1:
                    print("✅ JSONL file has consistent keys.")

        elif filename.endswith(".csv"):
            df = pd.read_csv(local_path, nrows=5)
            print("Columns:", df.columns.tolist())
            print("First 2 rows:")
            print(df.head(2))
            
    except Exception as e:
        print(f"❌ Error inspecting {filename}: {e}")
