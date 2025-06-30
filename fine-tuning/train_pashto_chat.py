#!/usr/bin/env python3
"""
ZamAI Pashto Model Fine-tuning Script
Fine-tune language models for Pashto chat and text generation
"""

import os
import json
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import wandb

class PashtoModelTrainer:
    def __init__(self, config_path="configs/pashto_chat_config.json"):
        """Initialize the Pashto model trainer"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.model_name = self.config["base_model"]
        self.output_dir = self.config["output_dir"]
        self.dataset_name = self.config.get("dataset_name")
        self.dataset_path = self.config.get("dataset_path")
        self.dataset_format = self.config.get("dataset_format", "conversation")
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer"""
        print(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()
        
    def setup_lora(self):
        """Setup LoRA for efficient fine-tuning"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config["lora"]["rank"],
            lora_alpha=self.config["lora"]["alpha"],
            lora_dropout=self.config["lora"]["dropout"],
            target_modules=self.config["lora"]["target_modules"],
            bias="none"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
    def prepare_dataset(self):
        """Prepare the Pashto dataset"""
        print("Loading Pashto dataset...")
        
        if self.dataset_name:
            # Load from Hugging Face Hub
            print(f"Loading dataset from HF Hub: {self.dataset_name}")
            
            if self.dataset_format == "instruction":
                # Load instruction-following format (ZamAI dataset)
                train_file = self.config.get("train_file", "pashto_train_instruction.jsonl")
                validation_file = self.config.get("validation_file", "pashto_val_instruction.jsonl")
                
                dataset = load_dataset(
                    self.dataset_name,
                    data_files={
                        "train": train_file,
                        "validation": validation_file
                    }
                )
                
                train_dataset = dataset["train"]
                eval_dataset = dataset["validation"]
            else:
                # Load standard format
                dataset = load_dataset(self.dataset_name, split="train")
                # Split for evaluation
                split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
                train_dataset = split_dataset["train"]
                eval_dataset = split_dataset["test"]
                
        elif self.dataset_path and self.dataset_path.endswith('.json'):
            # Load local JSON dataset
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            full_dataset = Dataset.from_list(data)
            split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset = split_dataset["train"]
            eval_dataset = split_dataset["test"]
        else:
            raise ValueError("Either dataset_name or dataset_path must be provided")
            
        # Tokenize the datasets
        def tokenize_function(examples):
            if self.dataset_format == "instruction":
                # Format instruction-following data
                formatted_texts = []
                for i in range(len(examples["instruction"])):
                    formatted = self.format_instruction_data(
                        examples["instruction"][i],
                        examples["input"][i],
                        examples["output"][i]
                    )
                    formatted_texts.append(formatted)
            else:
                # Format conversation data
                formatted_texts = []
                for conv in examples["conversation"]:
                    formatted = self.format_pashto_conversation(conv)
                    formatted_texts.append(formatted)
                
            return self.tokenizer(
                formatted_texts,
                truncation=True,
                padding=False,
                max_length=self.config["max_length"],
                return_overflowing_tokens=False
            )
        
        train_tokenized = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        eval_tokenized = eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=eval_dataset.column_names
        )
        
        return train_tokenized, eval_tokenized
    
    def format_instruction_data(self, instruction, input_text, output_text):
        """Format instruction-following data for training"""
        # Create instruction-following format compatible with Llama
        if input_text and input_text.strip():
            # With input
            formatted = f"<|system|>\nYou are a helpful AI assistant that can understand and generate Pashto text. Follow the instructions carefully.\n<|user|>\n{instruction}\n\nInput: {input_text}\n<|assistant|>\n{output_text}<|end_of_text|>"
        else:
            # Without input
            formatted = f"<|system|>\nYou are a helpful AI assistant that can understand and generate Pashto text. Follow the instructions carefully.\n<|user|>\n{instruction}\n<|assistant|>\n{output_text}<|end_of_text|>"
        
        return formatted
    
    def format_pashto_conversation(self, conversation):
        """Format conversation data for Pashto chat training"""
        system_prompt = """تاسو د پښتو ژبې یو ګټور مرستیال یاست. د افغان کلتور په درناوي سره ځواب ورکړئ.
د اسلامي ارزښتونو درناوی وکړئ او د پښتو ادبياتو څخه استفاده وکړئ."""
        
        formatted = f"<|system|>\n{system_prompt}\n"
        
        for turn in conversation:
            if turn["role"] == "user":
                formatted += f"<|user|>\n{turn['content']}\n"
            elif turn["role"] == "assistant":
                formatted += f"<|assistant|>\n{turn['content']}\n"
                
        formatted += "<|end|>"
        return formatted
    
    def train(self):
        """Main training function"""
        print("Starting Pashto model training...")
        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Setup LoRA
        self.setup_lora()
        
        # Prepare datasets
        train_dataset, eval_dataset = self.prepare_dataset()
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM, not masked LM
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.config["training"]["epochs"],
            per_device_train_batch_size=self.config["training"]["batch_size"],
            gradient_accumulation_steps=self.config["training"]["gradient_accumulation"],
            learning_rate=self.config["training"]["learning_rate"],
            warmup_steps=self.config["training"]["warmup_steps"],
            logging_steps=self.config["training"]["logging_steps"],
            save_steps=self.config["training"]["save_steps"],
            eval_steps=self.config["training"]["eval_steps"],
            evaluation_strategy="steps",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=True,
            dataloader_pin_memory=False,
            report_to="wandb" if self.config.get("use_wandb", False) else None,
            run_name=f"zamai-pashto-{self.config['model_version']}"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Start training
        trainer.train()
        
        # Save the final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"Training completed! Model saved to {self.output_dir}")
        
    def push_to_hub(self):
        """Push the trained model to Hugging Face Hub"""
        if self.config.get("push_to_hub", False):
            print("Pushing model to Hugging Face Hub...")
            
            self.model.push_to_hub(
                self.config["hub_model_id"],
                private=self.config.get("private_repo", False)
            )
            
            self.tokenizer.push_to_hub(
                self.config["hub_model_id"],
                private=self.config.get("private_repo", False)
            )
            
            print(f"Model pushed to: {self.config['hub_model_id']}")

def main():
    """Main function to run the training"""
    # Initialize wandb if configured
    config_path = "configs/pashto_chat_config.json"
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        if config.get("use_wandb", False):
            wandb.init(
                project="zamai-pashto-models",
                name=f"pashto-chat-{config['model_version']}",
                config=config
            )
    
    # Start training
    trainer = PashtoModelTrainer(config_path)
    trainer.train()
    trainer.push_to_hub()
    
    print("🇦🇫 ZamAI Pashto model training completed successfully!")

if __name__ == "__main__":
    main()
