#!/usr/bin/env python3
"""
Training script for ZamAI-QA-Pashto
Base model: microsoft/DialoGPT-medium
"""

import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

def train_zamai_qa_pashto():
    """Train ZamAI-QA-Pashto model"""
    
    print("🚀 Starting ZamAI-QA-Pashto training...")
    
    # Model configuration
    model_name = "microsoft/DialoGPT-medium"
    output_dir = "./outputs/zamai-qa-pashto"
    hub_model_id = "tasal9/zamai-qa-pashto"
    
    # Load tokenizer and model
    print("📥 Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    
    # Load dataset
    print("📊 Loading dataset...")
    try:
        dataset = load_dataset("tasal9/ZamAI_Pashto_Dataset")
    except:
        print("⚠️ Using dummy dataset for testing...")
        dataset = load_dataset("imdb", split="train[:100]")
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length",
            max_length=512
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=4,
        per_device_train_batch_size=3,
        per_device_eval_batch_size=3,
        learning_rate=5e-5,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        save_total_limit=2,
        prediction_loss_only=True,
        remove_unused_columns=False,
        push_to_hub=True,
        hub_model_id=hub_model_id,
        hub_strategy="checkpoint",
        report_to="none"
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset.select(range(100)) if len(tokenized_dataset) > 100 else tokenized_dataset,
    )
    
    # Train the model
    print("🏋️ Starting training...")
    trainer.train()
    
    # Save the model
    print("💾 Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Push to hub
    print("🌐 Pushing to Hub...")
    trainer.push_to_hub()
    
    print("✅ Training completed successfully!")
    print(f"🌐 Model available at: https://huggingface.co/{hub_model_id}")

if __name__ == "__main__":
    train_zamai_qa_pashto()
