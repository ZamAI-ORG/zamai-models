import json, random, sys
from pathlib import Path

p = Path('data/processed/dataset.jsonl')
if not p.exists():
    print('Dataset not found, run build_dataset first', file=sys.stderr)
    raise SystemExit(1)

lines = p.read_text(encoding='utf-8').splitlines()
record = json.loads(random.choice(lines))
print('\n--- SAMPLE RECORD ---')
for k,v in record.items():
    print(f'{k}: {v[:120]}...')
