"""Download and save the base Hugging Face model into ./base_model/.

This script downloads `facebook/xlm-roberta-base` tokenizer and model and
saves them to the `./base_model/` folder so you can archive or push them with
Git LFS.
"""
import os
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForMaskedLM


MODEL_NAME = "facebook/xlm-roberta-base"
TARGET_DIR = Path("./base_model")


def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading tokenizer and model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)

    print(f"Saving to {TARGET_DIR.resolve()}...")
    tokenizer.save_pretrained(TARGET_DIR)
    model.save_pretrained(TARGET_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
