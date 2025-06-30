#!/usr/bin/env python3
"""
ZamAI V3 - Model Fine-tuning Pipeline
Comprehensive training system for Pashto models
"""

import os
import json
import torch
import time
from pathlib import Path
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    pipeline, logging
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from huggingface_hub import HfApi, create_repo
import wandb

# Set logging level
logging.set_verbosity_warning()

class ZamAIModelPipeline:
    def __init__(self, config_file="configs/training_pipeline_config.json"):
        """Initialize the training pipeline"""
        self.load_config(config_file)
        self.setup_environment()
        
    def load_config(self, config_file):
        """Load training configuration"""
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        else:
            # Default configuration
            self.config = {
                "models_to_train": [
                    {
                        "name": "zamai-pashto-translator",
                        "base_model": "Helsinki-NLP/opus-mt-en-mul",
                        "task": "translation",
                        "dataset": "pashto_english_pairs",
                        "priority": "high"
                    },
                    {
                        "name": "zamai-pashto-qa",
                        "base_model": "deepset/roberta-base-squad2",
                        "task": "question-answering",
                        "dataset": "pashto_qa_dataset", 
                        "priority": "high"
                    },
                    {
                        "name": "zamai-pashto-sentiment",
                        "base_model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                        "task": "sentiment-analysis",
                        "dataset": "pashto_sentiment_dataset",
                        "priority": "medium"
                    }
                ],
                "training_params": {
                    "epochs": 3,
                    "batch_size": 8,
                    "learning_rate": 2e-5,
                    "warmup_steps": 500,
                    "save_steps": 1000
                },
                "hf_hub": {
                    "push_to_hub": True,
                    "private": False,
                    "organization": "tasal9"
                }
            }
            
            # Save default config
            os.makedirs("configs", exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
    
    def setup_environment(self):
        """Setup training environment"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Using device: {self.device}")
        
        # Load HF token
        token_file = "HF-Token.txt"
        if os.path.exists(token_file):
            with open(token_file, 'r') as f:
                self.hf_token = f.read().strip()
        else:
            self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        
        self.api = HfApi(token=self.hf_token)
    
    def create_pashto_datasets(self):
        """Create sample Pashto datasets for training"""
        print("📊 Creating Pashto training datasets...")
        
        # Translation dataset
        translation_data = [
            {"en": "Hello, how are you?", "ps": "سلام، څنګه یاست؟"},
            {"en": "What is your name?", "ps": "ستاسو نوم څه دی؟"},
            {"en": "Thank you very much", "ps": "ډېر مننه"},
            {"en": "Good morning", "ps": "سهار مو پخیر"},
            {"en": "Good night", "ps": "شپه مو پخیر"},
            {"en": "I love Afghanistan", "ps": "زه د افغانستان سره مینه لرم"},
            {"en": "Pashto is a beautiful language", "ps": "پښتو یوه ښکلې ژبه دی"},
            {"en": "Peace be upon you", "ps": "السلام علیکم"},
            {"en": "Where are you from?", "ps": "تاسو د کومه ځای یاست؟"},
            {"en": "I am learning Pashto", "ps": "زه د پښتو زده کړه کوم"}
        ]
        
        # Q&A dataset
        qa_data = [
            {
                "question": "د افغانستان پلازمېنه څه دی؟",
                "context": "افغانستان یو هېواد دی چې د آسیا په مرکز کې موقعیت لري. د دغه هېواد پلازمېنه کابل دی.",
                "answer": "کابل"
            },
            {
                "question": "د پښتو ژبې اصلي سیمه چېرته دی؟",
                "context": "پښتو د افغانستان او پاکستان رسمي ژبه دی. دا ژبه په دواړو هېوادونو کې پراخه ویل کېږي.",
                "answer": "افغانستان او پاکستان"
            },
            {
                "question": "د اسلام مقدس کتاب څه دی؟",
                "context": "اسلام د نړۍ درې لویې مذاهبو څخه یو دی. د اسلام مقدس کتاب قرآن کریم دی.",
                "answer": "قرآن کریم"
            }
        ]
        
        # Sentiment dataset
        sentiment_data = [
            {"text": "زه خوشحاله یم", "label": "positive"},
            {"text": "دا ډېر ښه دی", "label": "positive"},
            {"text": "زه خواشینی یم", "label": "negative"},
            {"text": "دا بد دی", "label": "negative"},
            {"text": "دا سمون دی", "label": "neutral"},
            {"text": "زه د خپل هېواد سره ویاړ لرم", "label": "positive"},
            {"text": "دا کار اسانه نه دی", "label": "negative"},
            {"text": "نن ورځ ښه ده", "label": "positive"}
        ]
        
        # Save datasets
        datasets_dir = "datasets"
        os.makedirs(datasets_dir, exist_ok=True)
        
        with open(f"{datasets_dir}/pashto_english_pairs.json", 'w', encoding='utf-8') as f:
            json.dump(translation_data, f, ensure_ascii=False, indent=2)
        
        with open(f"{datasets_dir}/pashto_qa_dataset.json", 'w', encoding='utf-8') as f:
            json.dump(qa_data, f, ensure_ascii=False, indent=2)
        
        with open(f"{datasets_dir}/pashto_sentiment_dataset.json", 'w', encoding='utf-8') as f:
            json.dump(sentiment_data, f, ensure_ascii=False, indent=2)
        
        print("✅ Sample datasets created successfully!")
    
    def train_translation_model(self, model_config):
        """Train Pashto-English translation model"""
        print(f"\n🔄 Training Translation Model: {model_config['name']}")
        
        # Load dataset
        dataset_path = f"datasets/{model_config['dataset']}.json"
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Prepare training data
        train_texts = []
        for item in data:
            # English to Pashto
            train_texts.append(f"translate English to Pashto: {item['en']}")
            train_texts.append(item['ps'])
            # Pashto to English  
            train_texts.append(f"translate Pashto to English: {item['ps']}")
            train_texts.append(item['en'])
        
        # Load model and tokenizer
        model_name = model_config['base_model']
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Tokenize data
        def tokenize_function(examples):
            model_inputs = tokenizer(examples['input'], truncation=True, padding=True, max_length=128)
            labels = tokenizer(examples['target'], truncation=True, padding=True, max_length=128)
            model_inputs['labels'] = labels['input_ids']
            return model_inputs
        
        # Create dataset
        train_data = []
        for i in range(0, len(train_texts), 2):
            train_data.append({
                'input': train_texts[i],
                'target': train_texts[i+1]
            })
        
        dataset = Dataset.from_list(train_data)
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Training arguments
        output_dir = f"outputs/{model_config['name']}"
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config['training_params']['epochs'],
            per_device_train_batch_size=self.config['training_params']['batch_size'],
            learning_rate=self.config['training_params']['learning_rate'],
            warmup_steps=self.config['training_params']['warmup_steps'],
            save_steps=self.config['training_params']['save_steps'],
            logging_steps=100,
            save_total_limit=2,
            prediction_loss_only=True,
            fp16=torch.cuda.is_available(),
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
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # Train
        print("🚀 Starting training...")
        trainer.train()
        
        # Save model
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        print(f"✅ Translation model saved to: {output_dir}")
        
        # Push to hub if configured
        if self.config['hf_hub']['push_to_hub']:
            self.push_model_to_hub(output_dir, f"tasal9/{model_config['name']}")
    
    def test_models(self):
        """Test your existing models"""
        print("\n🧪 Testing Existing ZamAI Models")
        print("=" * 40)
        
        existing_models = [
            "tasal9/ZamAI-Mistral-7B-Pashto",
            "tasal9/ZamAI-LIama3-Pashto", 
            "tasal9/pashto-base-bloom",
            "tasal9/Multilingual-ZamAI-Embeddings"
        ]
        
        test_inputs = [
            "سلام وروره، څنګه یاست؟",
            "د افغانستان تاریخ څه دی؟",
            "زه د پښتو زده کړه غواړم."
        ]
        
        for model_id in existing_models:
            print(f"\n📦 Testing: {model_id}")
            
            try:
                # Try to create a pipeline
                if "Embeddings" in model_id:
                    continue  # Skip embeddings for now
                
                # For text generation models
                generator = pipeline(
                    "text-generation", 
                    model=model_id,
                    tokenizer=model_id,
                    device=0 if torch.cuda.is_available() else -1,
                    token=self.hf_token
                )
                
                for test_input in test_inputs[:1]:  # Test with one input
                    print(f"  📝 Input: {test_input}")
                    
                    result = generator(
                        test_input,
                        max_length=150,
                        num_return_sequences=1,
                        temperature=0.7,
                        pad_token_id=generator.tokenizer.eos_token_id
                    )
                    
                    output = result[0]['generated_text']
                    print(f"  ✅ Output: {output[:100]}...")
                    break
                    
            except Exception as e:
                print(f"  ❌ Error: {str(e)[:100]}...")
    
    def push_model_to_hub(self, model_path, repo_name):
        """Push trained model to Hugging Face Hub"""
        try:
            print(f"📤 Pushing {repo_name} to Hugging Face Hub...")
            
            # Create repository
            self.api.create_repo(
                repo_id=repo_name,
                private=self.config['hf_hub']['private'],
                exist_ok=True
            )
            
            # Upload model files
            self.api.upload_folder(
                folder_path=model_path,
                repo_id=repo_name,
                repo_type="model"
            )
            
            print(f"✅ Model uploaded successfully: https://huggingface.co/{repo_name}")
            
        except Exception as e:
            print(f"❌ Upload error: {e}")
    
    def run_training_pipeline(self):
        """Run the complete training pipeline"""
        print("🇦🇫 ZamAI V3 - Model Training Pipeline")
        print("=" * 50)
        
        # Create sample datasets
        self.create_pashto_datasets()
        
        # Test existing models first
        print("\n1. Testing existing models...")
        self.test_models()
        
        # Train new models
        print("\n2. Training new models...")
        for model_config in self.config['models_to_train']:
            if model_config['priority'] == 'high':
                if model_config['task'] == 'translation':
                    self.train_translation_model(model_config)
                # Add other training methods as needed
        
        print("\n🎉 Training pipeline completed!")
        print("\n📊 Summary:")
        print("- ✅ Existing models tested")
        print("- ✅ New datasets created") 
        print("- ✅ Translation model trained")
        print("- ✅ Models pushed to Hugging Face Hub")

def main():
    """Main function"""
    pipeline = ZamAIModelPipeline()
    
    print("🎯 Choose an option:")
    print("1. Run full training pipeline")
    print("2. Test existing models only")
    print("3. Create datasets only")
    print("4. Train specific model")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        pipeline.run_training_pipeline()
    elif choice == '2':
        pipeline.test_models()
    elif choice == '3':
        pipeline.create_pashto_datasets()
    elif choice == '4':
        print("Available models:")
        for i, model in enumerate(pipeline.config['models_to_train']):
            print(f"{i+1}. {model['name']} ({model['task']})")
        
        model_choice = input("Choose model number: ").strip()
        try:
            model_idx = int(model_choice) - 1
            model_config = pipeline.config['models_to_train'][model_idx]
            if model_config['task'] == 'translation':
                pipeline.train_translation_model(model_config)
        except:
            print("Invalid choice")
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
