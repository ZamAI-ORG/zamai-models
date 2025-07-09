"""
ZeroGPU Training Setup for ZamAI Models
Create Hugging Face Spaces for training with ZeroGPU
"""

import os
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file
import tempfile

class ZeroGPUTrainingSetup:
    """Setup training on HF Spaces with ZeroGPU"""
    
    def __init__(self, token_path="/workspaces/ZamAI-Pro-Models/HF-Token.txt"):
        with open(token_path, 'r') as f:
            self.token = f.read().strip()
        self.api = HfApi(token=self.token)
        self.username = self.api.whoami()['name']
    
    def create_training_space(self, model_config, space_name):
        """Create a training space for a specific model"""
        
        # Create space repository
        space_id = f"{self.username}/{space_name}"
        
        try:
            self.api.create_repo(
                repo_id=space_id,
                repo_type="space",
                space_sdk="gradio",
                space_hardware="zero-a10g",  # ZeroGPU hardware
                private=True
            )
            print(f"✅ Created space: {space_id}")
        except Exception as e:
            if "already exists" in str(e):
                print(f"⚠️ Space {space_id} already exists")
            else:
                print(f"❌ Error creating space: {e}")
                return None
        
        # Create training script for the space
        training_script = self.generate_zerogpu_training_script(model_config)
        
        # Create requirements.txt for the space
        requirements = self.generate_space_requirements()
        
        # Create README.md for the space
        readme = self.generate_space_readme(model_config, space_name)
        
        # Upload files to space
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save files locally first
            script_path = os.path.join(temp_dir, "app.py")
            req_path = os.path.join(temp_dir, "requirements.txt")
            readme_path = os.path.join(temp_dir, "README.md")
            config_path = os.path.join(temp_dir, "config.json")
            
            with open(script_path, 'w') as f:
                f.write(training_script)
            with open(req_path, 'w') as f:
                f.write(requirements)
            with open(readme_path, 'w') as f:
                f.write(readme)
            with open(config_path, 'w') as f:
                json.dump(model_config, f, indent=2)
            
            # Upload to space
            self.api.upload_file(
                path_or_fileobj=script_path,
                path_in_repo="app.py",
                repo_id=space_id,
                repo_type="space"
            )
            self.api.upload_file(
                path_or_fileobj=req_path,
                path_in_repo="requirements.txt",
                repo_id=space_id,
                repo_type="space"
            )
            self.api.upload_file(
                path_or_fileobj=readme_path,
                path_in_repo="README.md",
                repo_id=space_id,
                repo_type="space"
            )
            self.api.upload_file(
                path_or_fileobj=config_path,
                path_in_repo="config.json",
                repo_id=space_id,
                repo_type="space"
            )
        
        return f"https://huggingface.co/spaces/{space_id}"
    
    def generate_zerogpu_training_script(self, model_config):
        """Generate training script optimized for ZeroGPU"""
        
        model_id = model_config.get('hub_model_id', 'zamai-model')
        base_model = model_config.get('base_model', 'meta-llama/Llama-3.1-8B-Instruct')
        dataset_name = model_config.get('dataset_name', 'tasal9/ZamAI_Pashto_Dataset')
        
        script = f'''import gradio as gr
import spaces
import torch
import json
import os
import time
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import wandb
from huggingface_hub import HfApi

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

class ZamAITrainer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.training_status = "Not started"
        
    @spaces.GPU(duration=120)  # Request GPU for 2 hours
    def setup_model(self):
        """Setup model and tokenizer with ZeroGPU"""
        try:
            self.training_status = "Loading model..."
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("{base_model}")
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model in 4-bit for memory efficiency
            self.model = AutoModelForCausalLM.from_pretrained(
                "{base_model}",
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            
            # Setup LoRA
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=config['lora']['rank'],
                lora_alpha=config['lora']['alpha'],
                lora_dropout=config['lora']['dropout'],
                target_modules=config['lora']['target_modules']
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.training_status = "Model loaded successfully"
            
            return "✅ Model setup complete"
            
        except Exception as e:
            self.training_status = f"Error: {{e}}"
            return f"❌ Error: {{e}}"
    
    @spaces.GPU(duration=180)  # Request GPU for 3 hours for training
    def start_training(self, wandb_key=""):
        """Start training with ZeroGPU"""
        try:
            if self.model is None:
                return "❌ Model not loaded. Setup model first."
            
            self.training_status = "Preparing training..."
            
            # Setup wandb if key provided
            if wandb_key:
                wandb.login(key=wandb_key)
                wandb.init(
                    project="zamai-zerogpu-training",
                    name="{model_id}",
                    config=config
                )
            
            # Load dataset
            self.training_status = "Loading dataset..."
            dataset = load_dataset("{dataset_name}")
            train_dataset = dataset['train']
            eval_dataset = dataset.get('validation', None)
            
            # Tokenize dataset
            def tokenize_function(examples):
                return self.tokenizer(
                    examples['text'] if 'text' in examples else examples['instruction'],
                    truncation=True,
                    padding=True,
                    max_length=config['max_length']
                )
            
            train_dataset = train_dataset.map(tokenize_function, batched=True)
            if eval_dataset:
                eval_dataset = eval_dataset.map(tokenize_function, batched=True)
            
            # Training arguments optimized for ZeroGPU
            training_args = TrainingArguments(
                output_dir="./outputs",
                num_train_epochs=config['training']['epochs'],
                per_device_train_batch_size=1,  # Small batch for memory
                gradient_accumulation_steps=16,  # Increase accumulation
                learning_rate=config['training']['learning_rate'],
                warmup_steps=config['training']['warmup_steps'],
                logging_steps=config['training']['logging_steps'],
                save_steps=config['training']['save_steps'],
                eval_steps=config['training']['eval_steps'] if eval_dataset else None,
                evaluation_strategy="steps" if eval_dataset else "no",
                save_strategy="steps",
                load_best_model_at_end=True if eval_dataset else False,
                dataloader_drop_last=True,
                fp16=True,  # Use mixed precision
                push_to_hub=True,
                hub_model_id="{model_id}",
                hub_token=os.getenv("HF_TOKEN"),
                report_to="wandb" if wandb_key else None,
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
            
            # Create trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
            )
            
            self.training_status = "Training started..."
            
            # Start training
            self.trainer.train()
            
            # Save and push to hub
            self.trainer.save_model()
            self.trainer.push_to_hub()
            
            self.training_status = "Training completed successfully!"
            
            return "✅ Training completed and model pushed to Hub!"
            
        except Exception as e:
            self.training_status = f"Training error: {{e}}"
            return f"❌ Training error: {{e}}"
    
    def get_status(self):
        """Get current training status"""
        return self.training_status

# Initialize trainer
trainer = ZamAITrainer()

# Gradio interface
with gr.Blocks(title="ZamAI ZeroGPU Training") as demo:
    gr.Markdown("# 🚀 ZamAI Model Training on ZeroGPU")
    gr.Markdown(f"Training model: **{model_id}**")
    gr.Markdown(f"Base model: **{base_model}**")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("## Step 1: Setup Model")
            setup_btn = gr.Button("🔧 Setup Model", variant="primary")
            setup_output = gr.Textbox(label="Setup Status", interactive=False)
            
        with gr.Column():
            gr.Markdown("## Step 2: Start Training")
            wandb_key = gr.Textbox(
                label="WandB API Key (optional)", 
                type="password",
                placeholder="Enter your WandB key for experiment tracking"
            )
            train_btn = gr.Button("🏃 Start Training", variant="secondary")
            train_output = gr.Textbox(label="Training Status", interactive=False)
    
    gr.Markdown("## Status Monitor")
    status_display = gr.Textbox(label="Current Status", interactive=False)
    refresh_btn = gr.Button("🔄 Refresh Status")
    
    # Event handlers
    setup_btn.click(trainer.setup_model, outputs=setup_output)
    train_btn.click(trainer.start_training, inputs=wandb_key, outputs=train_output)
    refresh_btn.click(trainer.get_status, outputs=status_display)
    
    # Auto-refresh status every 10 seconds
    demo.load(trainer.get_status, outputs=status_display, every=10)

if __name__ == "__main__":
    demo.launch()
'''
        
        return script
    
    def generate_space_requirements(self):
        """Generate requirements.txt for the space"""
        requirements = """
torch>=2.0.0
transformers>=4.36.0
datasets>=2.14.0
accelerate>=0.24.0
peft>=0.6.0
bitsandbytes>=0.41.0
gradio>=4.8.0
wandb>=0.16.0
huggingface-hub>=0.19.0
spaces>=0.19.0
"""
        return requirements.strip()
    
    def generate_space_readme(self, model_config, space_name):
        """Generate README.md for the space"""
        model_id = model_config.get('hub_model_id', 'zamai-model')
        base_model = model_config.get('base_model', 'meta-llama/Llama-3.1-8B-Instruct')
        
        readme = f"""---
title: {space_name}
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.8.0
app_file: app.py
pinned: false
license: apache-2.0
duplicated_from: spaces/template
hardware: zero-a10g
---

# 🚀 ZamAI Model Training on ZeroGPU

This space trains the **{model_id}** model using ZeroGPU infrastructure.

## Model Details
- **Base Model**: {base_model}
- **Target Model**: {model_id}
- **Training Method**: LoRA fine-tuning
- **Hardware**: ZeroGPU (A10G)

## How to Use

1. **Setup Model**: Click "Setup Model" to load the base model
2. **Start Training**: Click "Start Training" to begin fine-tuning
3. **Monitor**: Watch the status updates in real-time

## Features
- ✅ Automatic model uploading to HF Hub
- ✅ WandB integration for experiment tracking
- ✅ Memory-optimized training with 4-bit quantization
- ✅ ZeroGPU automatic resource management

## Training Configuration
The model uses optimized settings for ZeroGPU including:
- 4-bit quantization for memory efficiency
- Small batch sizes with gradient accumulation
- Mixed precision training (FP16)
- Automatic checkpointing and hub uploads

Training typically takes 2-3 hours depending on dataset size.
"""
        return readme

def setup_all_training_spaces():
    """Setup training spaces for all ZamAI models"""
    
    print("🚀 Setting up ZeroGPU Training Spaces for ZamAI Models")
    print("=" * 60)
    
    setup = ZeroGPUTrainingSetup()
    
    # Load model configurations
    configs_dir = Path("/workspaces/ZamAI-Pro-Models/configs")
    models_dir = Path("/workspaces/ZamAI-Pro-Models/models")
    
    training_spaces = []
    
    # 1. Pashto Chat Model (Priority 1)
    pashto_config_path = configs_dir / "pashto_chat_config.json"
    if pashto_config_path.exists():
        with open(pashto_config_path, 'r') as f:
            config = json.load(f)
        
        space_name = "zamai-pashto-chat-training"
        space_url = setup.create_training_space(config, space_name)
        if space_url:
            training_spaces.append({
                'model': config['hub_model_id'],
                'space_name': space_name,
                'space_url': space_url,
                'priority': 'HIGH'
            })
    
    # 2. Other models from models directory
    priority_models = {
        'mistral-7b-pashto': 'HIGH',
        'llama3-pashto': 'MEDIUM',
        'phi3-mini-pashto': 'MEDIUM',
        'bloom-pashto': 'LOW'
    }
    
    for category_dir in models_dir.iterdir():
        if category_dir.is_dir():
            for model_file in category_dir.glob("*.json"):
                try:
                    with open(model_file, 'r') as f:
                        model_config = json.load(f)
                    
                    # Skip if no model_id
                    if 'model_id' not in model_config:
                        continue
                    
                    # Create simplified config for training
                    training_config = {
                        'hub_model_id': model_config['model_id'],
                        'base_model': model_config.get('training_config', {}).get('base_model', 'meta-llama/Llama-3.1-8B-Instruct'),
                        'dataset_name': 'tasal9/ZamAI_Pashto_Dataset',
                        'max_length': 2048,
                        'lora': {
                            'rank': 32,
                            'alpha': 64,
                            'dropout': 0.1,
                            'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj"]
                        },
                        'training': {
                            'epochs': 2,
                            'learning_rate': 1e-5,
                            'warmup_steps': 100,
                            'logging_steps': 10,
                            'save_steps': 200,
                            'eval_steps': 200
                        }
                    }
                    
                    space_name = f"zamai-{model_file.stem}-training"
                    space_url = setup.create_training_space(training_config, space_name)
                    
                    if space_url:
                        priority = priority_models.get(model_file.stem, 'LOW')
                        training_spaces.append({
                            'model': model_config['model_id'],
                            'space_name': space_name,
                            'space_url': space_url,
                            'priority': priority
                        })
                
                except Exception as e:
                    print(f"⚠️ Skipping {model_file}: {e}")
    
    # Display results
    print(f"\n📊 TRAINING SPACES CREATED: {len(training_spaces)}")
    print("-" * 50)
    
    # Sort by priority
    priority_order = {'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
    training_spaces.sort(key=lambda x: priority_order.get(x['priority'], 4))
    
    for space in training_spaces:
        print(f"🎯 {space['priority']}: {space['model']}")
        print(f"   Space: {space['space_name']}")
        print(f"   URL: {space['space_url']}")
        print()
    
    # Save training plan
    plan = {
        'generated_at': time.time(),
        'total_spaces': len(training_spaces),
        'training_spaces': training_spaces
    }
    
    plan_path = "/workspaces/ZamAI-Pro-Models/zerogpu_training_plan.json"
    with open(plan_path, 'w') as f:
        json.dump(plan, f, indent=2)
    
    print(f"💾 Training plan saved to: {plan_path}")
    
    print("\n🚀 NEXT STEPS:")
    print("1. Visit the HIGH priority space URLs above")
    print("2. Click 'Setup Model' in each space")
    print("3. Click 'Start Training' to begin fine-tuning")
    print("4. Monitor progress in the space interface")
    print("5. Models will auto-upload to your HF Hub when complete")
    
    return training_spaces

if __name__ == "__main__":
    setup_all_training_spaces()
