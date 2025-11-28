import argparse
import json
from pathlib import Path
from dataclasses import dataclass

import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from .config import config

@dataclass
class FinetuneArgs:
    dataset_path: str
    output_dir: str
    model_name: str = config.MODEL_NAME
    lr: float = 2e-4
    epochs: int = 1
    batch_size: int = 2
    grad_accum: int = 8
    max_length: int = 1024

PROMPT_FORMAT = """<|system|>\n{system}\n<|user|>\n{instruction}\n<|assistant|>\n{output}\n"""

def load_jsonl_dataset(path: str):
    # Use HF datasets from local jsonl
    return load_dataset('json', data_files=path, split='train')


def tokenize_function(example, tokenizer, max_length):
    text = PROMPT_FORMAT.format(**example)
    toks = tokenizer(text, truncation=True, max_length=max_length, add_special_tokens=True)
    return {"input_ids": toks["input_ids"], "attention_mask": toks["attention_mask"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default=config.DATASET_JSONL)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--model_name', default=config.MODEL_NAME)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--grad_accum', type=int, default=8)
    parser.add_argument('--max_length', type=int, default=1024)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_jsonl_dataset(args.dataset_path)
    ds = ds.map(lambda ex: tokenize_function(ex, tokenizer, args.max_length))

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        load_in_8bit=True,
        device_map='auto'
    )

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=['q_proj','k_proj','v_proj','o_proj'], lora_dropout=0.05, bias='none', task_type='CAUSAL_LM')
    model = get_peft_model(model, lora_config)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy='epoch',
        bf16=torch.cuda.is_bf16_supported(),
        report_to=[]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == '__main__':
    main()
