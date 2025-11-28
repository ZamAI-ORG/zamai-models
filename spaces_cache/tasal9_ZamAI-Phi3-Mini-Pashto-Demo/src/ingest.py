import argparse
from pathlib import Path
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from .config import config

EMB_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'


def load_chunks():
    dataset_path = Path(config.DATASET_JSONL)
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            ex = json.loads(line)
            yield ex['output']


def build_index(output_dir: str):
    model = SentenceTransformer(EMB_MODEL)
    texts = list(load_chunks())
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, f"{output_dir}/vector.index")
    with open(f"{output_dir}/texts.json", 'w', encoding='utf-8') as f:
        json.dump(texts, f, ensure_ascii=False)
    print(f"Index built with {len(texts)} chunks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='Path to fine-tuned model (not used here yet)')
    ap.add_argument('--output_dir', default='rag_store')
    args = ap.parse_args()
    build_index(args.output_dir)

if __name__ == '__main__':
    main()
