
from datasets import load_dataset
import pandas as pd

dataset_name = "tasal9/ZamAI_Pashto_Dataset"

print(f"🔍 Debugging dataset: {dataset_name}")

# Try loading instruction files
try:
    print("\n--- Loading Instruction Files ---")
    data_files = {
        "train": "pashto_train_instruction.jsonl",
        "validation": "pashto_val_instruction.jsonl"
    }
    ds_instruct = load_dataset(dataset_name, data_files=data_files)
    print("✅ Instruction files loaded successfully!")
    print(ds_instruct)
    print("Columns:", ds_instruct['train'].column_names)
except Exception as e:
    print(f"❌ Error loading instruction files: {e}")

# Try loading prompt-completion files
try:
    print("\n--- Loading Prompt-Completion Files ---")
    data_files = {
        "train": "pashto_train_prompt_completion.jsonl"
    }
    ds_prompt = load_dataset(dataset_name, data_files=data_files)
    print("✅ Prompt-completion files loaded successfully!")
    print(ds_prompt)
    print("Columns:", ds_prompt['train'].column_names)
except Exception as e:
    print(f"❌ Error loading prompt-completion files: {e}")

# Try loading CSV files
try:
    print("\n--- Loading CSV Files ---")
    data_files = {
        "train": "pashto_cleaned_train.csv",
        "validation": "pashto_cleaned_val.csv"
    }
    ds_csv = load_dataset(dataset_name, data_files=data_files)
    print("✅ CSV files loaded successfully!")
    print(ds_csv)
    print("Columns:", ds_csv['train'].column_names)
except Exception as e:
    print(f"❌ Error loading CSV files: {e}")
