import os
import re
import sys
import time
import urllib.parse
from pathlib import Path
from typing import List
import shutil
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup

from .config import config
HEADERS = {"User-Agent": "Mozilla/5.0 (Educational Pashto Access)"}

def fetch_html(url: str) -> str:
    req = Request(url, headers=HEADERS)
    with urlopen(req, timeout=30) as r:
        return r.read().decode('utf-8', errors='replace')
    return r.text

def find_pdf_links(base_url: str) -> List[str]:
    if not base_url:
        print("Config SOURCE_BASE_URL empty. Set PASHTO_SOURCE_URL env var or edit config.py", file=sys.stderr)
        return []
    html = fetch_html(base_url)
    soup = BeautifulSoup(html, 'html.parser')
    pdfs = set()
    for a in soup.find_all('a', href=True):
        href = a['href']
        if href.lower().endswith('.pdf'):
            full = urllib.parse.urljoin(base_url, href)
            pdfs.add(full)
    return sorted(pdfs)
def download_pdf(url: str, out_dir: str) -> Path:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    filename = urllib.parse.unquote(url.split('/')[-1]) or f"file_{int(time.time()*1000)}.pdf"
    dest = Path(out_dir) / filename
    if dest.exists():
        print(f"Skip existing {dest}")
        return dest
    req = Request(url, headers=HEADERS)
    with urlopen(req, timeout=120) as r, open(dest, 'wb') as f:
        shutil.copyfileobj(r, f)
    print(f"Downloaded {dest}")
    return dest
    return dest

def main():
    pdf_links = find_pdf_links(config.SOURCE_BASE_URL)
    if not pdf_links:
        print("No PDF links found or base URL not set.")
        return
    for link in pdf_links:
        try:
            download_pdf(link, config.RAW_PDF_DIR)
        except Exception as e:
            print(f"Failed {link}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
