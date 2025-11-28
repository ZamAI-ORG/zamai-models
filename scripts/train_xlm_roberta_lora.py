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
    parser.add_argument(
        "--dataset",
        default="tasal9/ZamAi-Pashto-Datasets-V2",
        help="Dataset repo id to train on",
    )
    parser.add_argument(
        "--dataset-split",
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--base-model",
        default="FacebookAI/xlm-roberta-base",
        help="Base model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        default="adapters/xlm_roberta_pashto_lora_trained",
        help="Where to save the trained adapter",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum training steps (keeps the run short)",
    )
    parser.add_argument(
        "--per-device-batch",
        type=int,
        default=4,
        help="Per-device train batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=5000,
        help="Optional cap on number of training samples to keep run lightweight",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload the resulting adapter to tasal9/ZamAI-Facebook-XLM-Pashto",
    )
    parser.add_argument(
        "--token-path",
        default="HF-Token.txt",
        help="Path to file containing the HF token (used when pushing)",
    )
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


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📚 Loading dataset {args.dataset} ({args.dataset_split}) ...")
    dataset = load_dataset(
        args.dataset,
        split=args.dataset_split,
        verification_mode="no_checks",
    )

    print("🔡 Loading tokenizer and base model ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForMaskedLM.from_pretrained(args.base_model)

    def tokenize(batch):
        keys = list(batch.keys())
        rows = []
        for idx in range(len(batch[keys[0]])):
            row = {key: batch[key][idx] for key in keys}
            rows.append(build_text(row))
        return tokenizer(rows, truncation=True, max_length=512)

    print("✂️  Tokenizing dataset ...")
    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names,
    )

    if args.max_train_samples and len(tokenized) > args.max_train_samples:
        tokenized = tokenized.shuffle(seed=42).select(range(args.max_train_samples))

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


if __name__ == "__main__":
    main()
+
+    main()
