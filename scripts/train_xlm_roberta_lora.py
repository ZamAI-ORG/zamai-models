#!/usr/bin/env python3
"""Run a short PEFT/LoRA masked-language-modeling job on XLM-R."""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LoRA adapter on Pashto data")
    arg_configs = (
        (("--dataset",), {"default": "tasal9/ZamAi-Pashto-Datasets-V2", "help": "Dataset repo id"}),
        (("--dataset-split",), {"default": "train", "help": "Dataset split (default: train)"}),
        (("--base-model",), {"default": "FacebookAI/xlm-roberta-base", "help": "Base model checkpoint"}),
        (("--output-dir",), {"default": "adapters/xlm_roberta_pashto_lora_trained", "help": "Adapter output directory"}),
        (("--max-steps",), {"type": int, "default": 200, "help": "Maximum training steps"}),
        (("--per-device-batch",), {"type": int, "default": 4, "help": "Per-device train batch size"}),
        (("--lr",), {"type": float, "default": 2e-4, "help": "Learning rate"}),
        (("--max-train-samples",), {"type": int, "default": 5000, "help": "Optional cap on training samples"}),
        (("--max-seq-length",), {"type": int, "default": 512, "help": "Sequence length for tokenization"}),
        (("--push-to-hub",), {"action": "store_true", "help": "Upload adapter to tasal9/ZamAI-Facebook-XLM-Pashto"}),
        (("--token-path",), {"default": "HF-Token.txt", "help": "HF token path (used when pushing)"}),
    )
    for names, kwargs in arg_configs:
        parser.add_argument(*names, **kwargs)
    return parser.parse_args()


def build_text(example: dict) -> str:
    parts = []
    if example.get("instruction"):
        parts.append(f"Instruction: {example['instruction']}")
    if example.get("input"):
        parts.append(f"Input: {example['input']}")
    if example.get("output"):
        parts.append(f"Output: {example['output']}")
    return " \n".join(parts)


def load_model_and_tokenizer(base_model: str):
    print("🔡 Loading tokenizer and base model ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForMaskedLM.from_pretrained(base_model)
    return tokenizer, model


def prepare_dataset(args: argparse.Namespace, tokenizer):
    print(f"📚 Loading dataset {args.dataset} ({args.dataset_split}) ...")
    dataset = load_dataset(
        args.dataset,
        split=args.dataset_split,
        verification_mode="no_checks",
    )

    def tokenize(batch):
        keys = list(batch.keys())
        rows = []
        for idx in range(len(batch[keys[0]])):
            row = {key: batch[key][idx] for key in keys}
            rows.append(build_text(row))
        return tokenizer(rows, truncation=True, max_length=args.max_seq_length)

    print("✂️  Tokenizing dataset ...")
    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
    )

    if args.max_train_samples and len(tokenized) > args.max_train_samples:
        tokenized = tokenized.shuffle(seed=42).select(range(args.max_train_samples))

    return tokenized


def push_adapter(args: argparse.Namespace, output_dir: Path) -> None:
    from huggingface_hub import HfApi

    token = Path(args.token_path).read_text().strip()
    api = HfApi(token=token)
    print("☁️  Uploading adapter to Hugging Face ...")
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id="tasal9/ZamAI-Facebook-XLM-Pashto",
        repo_type="model",
        path_in_repo="adapters/pashto-lora",
        commit_message="Update Pashto LoRA adapter",
    )
    print("✅ Upload complete")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_model_and_tokenizer(args.base_model)
    tokenized = prepare_dataset(args, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["query", "value"],
        bias="none",
    )

    print("🪝 Injecting LoRA adapters ...")
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "trainer"),
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_batch,
        max_steps=args.max_steps,
        logging_steps=10,
        save_strategy="no",
        gradient_accumulation_steps=1,
        bf16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    print("🚂 Starting training ...")
    trainer.train()

    print(f"💾 Saving adapter to {output_dir} ...")
    model.save_pretrained(output_dir, safe_serialization=True)
    (output_dir / "README.md").write_text(
        "LoRA adapter fine-tuned on tasal9/ZamAi-Pashto-Datasets-V2 using transformers+PEFT.\n"
    )

    if args.push_to_hub:
        push_adapter(args, output_dir)


if __name__ == "__main__":
    main()
