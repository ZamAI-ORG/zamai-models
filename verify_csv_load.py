
from datasets import load_dataset

dataset_name = "tasal9/ZamAI_Pashto_Dataset"

print("Testing CSV loading with specific builder...")
try:
    ds = load_dataset("csv", data_files={"train": "pashto_cleaned_train.csv"})
    print("✅ CSV loaded successfully with 'csv' builder!")
    print(ds)
    print("Columns:", ds['train'].column_names)
except Exception as e:
    print(f"❌ CSV loading failed: {e}")
