import json
from pathlib import Path
import random
from .config import config

SYSTEM_PROMPT = "You are a helpful Pashto educational assistant for Afghan girls, explaining textbook content clearly and respectfully."

INSTRUCTION_TEMPLATES = [
    "په ساده ډول تشريح کړه: {snippet}",
    "دا برخه څه معنی لري؟ {snippet}",
    "مهم ټکي راوباسه: {snippet}",
]


def chunk_text(text: str, size: int, overlap: int):
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        yield chunk
        start += size - overlap


def build_examples():
    processed_dir = Path(config.PROCESSED_DIR)
    dataset_path = Path(config.DATASET_JSONL)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(dataset_path, 'w', encoding='utf-8') as out:
        for txt_file in processed_dir.glob('*.txt'):
            content = txt_file.read_text(encoding='utf-8')
            for chunk in chunk_text(content, config.CHUNK_SIZE, config.CHUNK_OVERLAP):
                if count >= config.MAX_SAMPLES_PER_PDF:
                    break
                snippet = chunk[:300].replace('\n', ' ')
                instr = random.choice(INSTRUCTION_TEMPLATES).format(snippet=snippet)
                record = {
                    "system": SYSTEM_PROMPT,
                    "instruction": instr,
                    "input": "",
                    "output": chunk.strip(),
                    "source": txt_file.name
                }
                out.write(json.dumps(record, ensure_ascii=False) + '\n')
                count += 1
            count = 0  # reset per PDF
    print(f"Dataset written to {dataset_path}")


def main():
    build_examples()

if __name__ == '__main__':
    main()
