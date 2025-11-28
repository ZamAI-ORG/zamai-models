import os
from dataclasses import dataclass

@dataclass
class Config:
    # Base URL of the website hosting Pashto textbooks (set this!)
    SOURCE_BASE_URL: str = os.getenv("PASHTO_SOURCE_URL", "")
    OUTPUT_DIR: str = "data"
    RAW_PDF_DIR: str = "data/raw_pdfs"
    PROCESSED_DIR: str = "data/processed"
    DATASET_JSONL: str = "data/processed/dataset.jsonl"
    CHUNK_SIZE: int = 800  # characters
    CHUNK_OVERLAP: int = 120
    MODEL_NAME: str = "microsoft/phi-2"  # placeholder; replace with small phi-3 once available via HF
    MAX_SAMPLES_PER_PDF: int = 400

config = Config()
