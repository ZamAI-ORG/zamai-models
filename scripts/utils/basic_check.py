#!/usr/bin/env python3
import os
print("🧪 Basic Check")
print(f"Current directory: {os.getcwd()}")
print(f"Files here: {os.listdir('.')}")

# Check HF token
if os.path.exists("HF-Token.txt"):
    print("✅ HF Token found")
else:
    print("❌ HF Token not found")

# Check Python packages
try:
    import transformers
    print(f"✅ Transformers: {transformers.__version__}")
except:
    print("❌ Transformers not found")

try:
    import datasets
    print(f"✅ Datasets: {datasets.__version__}")
except:
    print("❌ Datasets not found")

try:
    from huggingface_hub import HfApi
    print("✅ Hugging Face Hub available")
except:
    print("❌ Hugging Face Hub not found")

print("🎯 Ready to proceed!")
