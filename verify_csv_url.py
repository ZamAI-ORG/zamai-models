
from datasets import load_dataset

url = "https://huggingface.co/datasets/tasal9/ZamAI_Pashto_Dataset/resolve/main/pashto_cleaned_train.csv"

print(f"Testing CSV loading from URL: {url}")
try:
    ds = load_dataset("csv", data_files=url)
    print("✅ CSV loaded successfully from URL!")
    print(ds)
    print("Columns:", ds['train'].column_names)
except Exception as e:
    print(f"❌ CSV loading failed: {e}")
