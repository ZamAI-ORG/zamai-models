#!/usr/bin/env python3
"""Normalize ZamAI prompt/completion JSONL files to the instruction schema.

This utility converts rows that look like::

    {"prompt": "Title: ...", "completion": "..."}

into the instruction-following structure expected by most training scripts::

    {
        "instruction": "Write a Pashto article based on the provided title.",
        "input": "Title: ...",
        "output": "..."
    }

Example usage::

    python scripts/datasets/normalize_prompt_completion.py \
        --source data/raw/pashto_train_prompt_completion.jsonl \
        --destination data/processed/pashto_train_instruction_ready.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

INSTRUCTION_COLUMNS = ("instruction", "input", "output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize prompt/completion JSONL files")
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Path to the prompt/completion JSONL file",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        help="Path for the normalized instruction JSONL file. Defaults to '<source>_instruction.jsonl'.",
    )
    parser.add_argument(
        "--instruction-text",
        default="Write a Pashto article based on the provided title.",
        help="Fallback instruction string when the source row does not contain an instruction field.",
    )
    parser.add_argument(
        "--prompt-field",
        default="prompt",
        help="Field name containing the prompt/input text.",
    )
    parser.add_argument(
        "--completion-field",
        default="completion",
        help="Field name containing the completion/response text.",
    )
    parser.add_argument(
        "--instruction-field",
        default="instruction",
        help="Optional field to read an already-provided instruction from the source row.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="File encoding for both source and destination. Default: utf-8.",
    )
    return parser.parse_args()


def normalize_records(
    source: Path,
    destination: Path,
    instruction_text: str,
    prompt_field: str,
    completion_field: str,
    instruction_field: str,
    encoding: str,
) -> int:
    """Convert the source JSONL file into instruction format.

    Returns the number of records written.
    """

    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    total = 0

    with source.open("r", encoding=encoding) as reader, destination.open(
        "w", encoding=encoding
    ) as writer:
        for line_number, line in enumerate(reader, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {source}:{line_number}: {exc.msg}"
                ) from exc

            prompt_value = record.get(prompt_field, "")
            completion_value = record.get(completion_field, "")
            instruction_value = record.get(instruction_field) or instruction_text

            normalized = {
                "instruction": instruction_value,
                "input": prompt_value,
                "output": completion_value,
            }

            writer.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            total += 1

    return total


def main() -> None:
    args = parse_args()
    destination = args.destination or args.source.with_name(
        f"{args.source.stem}_instruction.jsonl"
    )

    count = normalize_records(
        source=args.source,
        destination=destination,
        instruction_text=args.instruction_text,
        prompt_field=args.prompt_field,
        completion_field=args.completion_field,
        instruction_field=args.instruction_field,
        encoding=args.encoding,
    )

    print(
        f"✅ Converted {count} rows from {args.source} to instruction schema at {destination}"
    )


if __name__ == "__main__":
    main()
