import json
import sys
from pathlib import Path
from typing import List

from .config import config

try:
    import pypdf
except ImportError:
    print("Please install pypdf (pip install pypdf)", file=sys.stderr)
    raise

PASHTO_CHAR_RANGE = (0x0600, 0x06FF)  # rough; includes Arabic block


def is_pashto_text(s: str) -> bool:
    # Check if at least 30% chars are in Arabic block (heuristic)
    if not s:
        return False
    letters = [c for c in s if ord(c) >= PASHTO_CHAR_RANGE[0] and ord(c) <= PASHTO_CHAR_RANGE[1]]
    return len(letters) / max(1, len(s)) > 0.3


def clean_text(s: str) -> str:
    s = s.replace('\u200c', ' ')
    s = s.replace('\xa0', ' ')
    lines = [l.strip() for l in s.splitlines()]
    lines = [l for l in lines if l]
    return '\n'.join(lines)


def extract_pdf(path: Path) -> str:
    reader = pypdf.PdfReader(str(path))
    texts = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ''
        except Exception:
            t = ''
        t = clean_text(t)
        if is_pashto_text(t):
            texts.append(t)
    return '\n\n'.join(texts)


def write_text_file(pdf_path: Path, text: str):
    out_dir = Path(config.PROCESSED_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / (pdf_path.stem + '.txt')
    out_file.write_text(text, encoding='utf-8')
    print(f"Wrote {out_file}")


def main():
    raw_dir = Path(config.RAW_PDF_DIR)
    if not raw_dir.exists():
        print("No raw_pdfs directory.")
        return
    for pdf in raw_dir.glob('*.pdf'):
        try:
            text = extract_pdf(pdf)
            if text.strip():
                write_text_file(pdf, text)
            else:
                print(f"No Pashto text found in {pdf}")
        except Exception as e:
            print(f"Failed {pdf}: {e}", file=sys.stderr)

if __name__ == '__main__':
    main()
