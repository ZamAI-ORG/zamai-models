from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, 
    DataCollatorForLanguageModeling
)
from huggingface_hub import HfApi
import torch
import os

class ZamAITutorTrainer:
    def __init__(self, base_model="mistralai/Mistral-7B-Instruct-v0.2"):
        self.base_model = base_model
        self.output_model = "tasal9/ZamAI-Tutor-Bot"
        self.token = os.getenv("HUGGINGFACE_TOKEN")
        
    def prepare_dataset(self):
        """Load and prepare Pashto educational dataset"""
        # Load your Pashto dataset
        dataset = load_dataset("tasal9/ZamAI_Pashto_Dataset", token=self.token)
        
        # Filter for educational content
        def is_educational(example):
            keywords = ["ښوونځی", "زده کړه", "درس", "پوښتنه", "ځواب"]
            return any(keyword in example.get("instruction", "") for keyword in keywords)
        
        educational_dataset = dataset.filter(is_educational)
        
        return educational_dataset
    
    def format_prompts(self, examples):
        """Format prompts for educational context"""
        prompts = []
        for instruction, output in zip(examples["instruction"], examples["output"]):
            prompt = f"""### د ښوونې پوښتنه:
{instruction}

### ځواب:
{output}"""
            prompts.append(prompt)
        return {"text": prompts}
    
    def train(self):
        """Train the ZamAI Tutor Bot"""
        print("🏫 Starting ZamAI Tutor Bot training...")
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Prepare dataset
        dataset = self.prepare_dataset()
        tokenized_dataset = dataset.map(
            self.format_prompts,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        # Tokenize
        def tokenize(examples):
            return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)
        
        tokenized_dataset = tokenized_dataset.map(tokenize, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./zamai-tutor-training",
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            learning_rate=2e-5,
            fp16=True,
            logging_steps=10,
            save_steps=500,
            report_to="none",
            push_to_hub=True,
            hub_model_id=self.output_model,
            hub_token=self.token
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        
        # Train and push
        trainer.train()
        trainer.push_to_hub()
        
        print(f"✅ ZamAI Tutor Bot trained and pushed to {self.output_model}")

if __name__ == "__main__":
    trainer = ZamAITutorTrainer()
    trainer.train()
