"""
Quick ZeroGPU Training Space Creator
Create a single training space for your priority model
"""

import os
import json
import tempfile
from huggingface_hub import HfApi

def create_pashto_chat_training_space():
    """Create training space for the main Pashto chat model"""
    
    print("🚀 Creating ZeroGPU Training Space for Pashto Chat Model")
    print("=" * 60)
    
    # Load HF token
    token_path = "/workspaces/ZamAI-Pro-Models/HF-Token.txt"
    if not os.path.exists(token_path):
        print("❌ HF Token not found")
        return None
    
    with open(token_path, 'r') as f:
        token = f.read().strip()
    
    api = HfApi(token=token)
    username = api.whoami()['name']
    
    # Load Pashto chat config
    config_path = "/workspaces/ZamAI-Pro-Models/configs/pashto_chat_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    space_name = "zamai-pashto-chat-zerogpu-training"
    space_id = f"{username}/{space_name}"
    
    print(f"Creating space: {space_id}")
    
    # Create space
    try:
        api.create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="gradio",
            space_hardware="zero-a10g",
            private=False  # Make it public so you can access it easily
        )
        print(f"✅ Space created: {space_id}")
    except Exception as e:
        if "already exists" in str(e):
            print(f"⚠️ Space already exists: {space_id}")
        else:
            print(f"❌ Error creating space: {e}")
            return None
    
    # Create training app
    app_py = f'''import gradio as gr
import spaces
import torch
import json
import os
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import wandb

# Model configuration
MODEL_ID = "{config['hub_model_id']}"
BASE_MODEL = "{config['base_model']}"
DATASET_NAME = "{config['dataset_name']}"

class ZamAITrainer:
    def __init__(self):
        self.status = "Ready to start"
        self.model = None
        self.tokenizer = None
    
    @spaces.GPU(duration=120)
    def setup_model(self):
        try:
            self.status = "Loading tokenizer..."
            yield self.status
            
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.status = "Loading model (this may take a few minutes)..."
            yield self.status
            
            self.model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_4bit=True,
                trust_remote_code=True
            )
            
            self.status = "Setting up LoRA..."
            yield self.status
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r={config['lora']['rank']},
                lora_alpha={config['lora']['alpha']},
                lora_dropout={config['lora']['dropout']},
                target_modules={config['lora']['target_modules']}
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
            self.status = "✅ Model setup complete! Ready to train."
            yield self.status
            
        except Exception as e:
            self.status = f"❌ Setup error: {{str(e)}}"
            yield self.status
    
    @spaces.GPU(duration=240)  # 4 hours for training
    def start_training(self, wandb_key):
        try:
            if self.model is None:
                self.status = "❌ Please setup model first"
                yield self.status
                return
            
            # Setup wandb if key provided
            if wandb_key.strip():
                self.status = "Setting up WandB..."
                yield self.status
                wandb.login(key=wandb_key.strip())
                wandb.init(project="zamai-models", name=MODEL_ID.split('/')[-1])
            
            self.status = "Loading dataset..."
            yield self.status
            
            dataset = load_dataset(DATASET_NAME)
            train_dataset = dataset['train']
            
            # Simple tokenization
            def tokenize_function(examples):
                # Handle both instruction and conversation formats
                if 'instruction' in examples:
                    texts = [f"### Instruction\\n{{inst}}\\n\\n### Response\\n{{resp}}" 
                            for inst, resp in zip(examples['instruction'], examples['output'])]
                elif 'text' in examples:
                    texts = examples['text']
                else:
                    texts = [str(ex) for ex in examples]
                
                return self.tokenizer(
                    texts,
                    truncation=True,
                    padding=True,
                    max_length={config['max_length']},
                    return_tensors="pt"
                )
            
            self.status = "Tokenizing dataset..."
            yield self.status
            
            train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
            
            self.status = "Setting up training..."
            yield self.status
            
            training_args = TrainingArguments(
                output_dir="./outputs",
                num_train_epochs={config['training']['epochs']},
                per_device_train_batch_size=1,
                gradient_accumulation_steps=8,
                learning_rate={config['training']['learning_rate']},
                warmup_steps={config['training']['warmup_steps']},
                logging_steps=10,
                save_steps=100,
                fp16=True,
                push_to_hub=True,
                hub_model_id=MODEL_ID,
                hub_token=os.getenv("HF_TOKEN"),
                report_to="wandb" if wandb_key.strip() else None,
                dataloader_drop_last=True,
            )
            
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
            
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=data_collator,
            )
            
            self.status = "🏃 Training started! This will take 2-3 hours..."
            yield self.status
            
            trainer.train()
            
            self.status = "💾 Saving model..."
            yield self.status
            
            trainer.save_model()
            trainer.push_to_hub()
            
            self.status = f"🎉 Training complete! Model uploaded to {{MODEL_ID}}"
            yield self.status
            
        except Exception as e:
            self.status = f"❌ Training error: {{str(e)}}"
            yield self.status

trainer = ZamAITrainer()

with gr.Blocks(title="ZamAI Pashto Chat Training") as demo:
    gr.Markdown("# 🚀 ZamAI Pashto Chat Model Training")
    gr.Markdown(f"**Model**: `{{MODEL_ID}}`")
    gr.Markdown(f"**Base**: `{{BASE_MODEL}}`")
    gr.Markdown(f"**Dataset**: `{{DATASET_NAME}}`")
    
    gr.Markdown("## Instructions")
    gr.Markdown("1. Click **Setup Model** (takes ~5 minutes)")
    gr.Markdown("2. Enter your WandB key (optional)")
    gr.Markdown("3. Click **Start Training** (takes ~2-3 hours)")
    gr.Markdown("4. Monitor progress below")
    
    with gr.Row():
        setup_btn = gr.Button("🔧 Setup Model", variant="primary")
        train_btn = gr.Button("🏃 Start Training", variant="secondary")
    
    wandb_key = gr.Textbox(
        label="WandB API Key (optional)",
        type="password",
        placeholder="Get from: https://wandb.ai/settings"
    )
    
    status_box = gr.Textbox(
        label="Training Status",
        value="Ready to start",
        interactive=False,
        lines=3
    )
    
    setup_btn.click(trainer.setup_model, outputs=status_box)
    train_btn.click(trainer.start_training, inputs=wandb_key, outputs=status_box)

if __name__ == "__main__":
    demo.launch()
'''
    
    # Create requirements.txt
    requirements = """torch>=2.0.0
transformers>=4.36.0
datasets>=2.14.0
accelerate>=0.24.0
peft>=0.6.0
bitsandbytes>=0.41.0
gradio>=4.8.0
spaces>=0.19.0
wandb>=0.16.0
huggingface-hub>=0.19.0"""
    
    # Create README.md
    readme = f"""---
title: ZamAI Pashto Chat Training
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.8.0
app_file: app.py
pinned: false
license: apache-2.0
hardware: zero-a10g
---

# 🚀 ZamAI Pashto Chat Model Training

This space trains the **{config['hub_model_id']}** model using ZeroGPU.

## Model Details
- **Base Model**: {config['base_model']}
- **Target Model**: {config['hub_model_id']}
- **Dataset**: {config['dataset_name']}
- **Training Method**: LoRA fine-tuning

## Usage
1. Click "Setup Model" to load the base model
2. Click "Start Training" to begin fine-tuning
3. Monitor progress in real-time

Training takes approximately 2-3 hours and the model will be automatically uploaded to your HuggingFace Hub.
"""
    
    # Upload files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Write files
        app_path = os.path.join(temp_dir, "app.py")
        req_path = os.path.join(temp_dir, "requirements.txt")
        readme_path = os.path.join(temp_dir, "README.md")
        
        with open(app_path, 'w') as f:
            f.write(app_py)
        with open(req_path, 'w') as f:
            f.write(requirements)
        with open(readme_path, 'w') as f:
            f.write(readme)
        
        # Upload to space
        api.upload_file(
            path_or_fileobj=app_path,
            path_in_repo="app.py",
            repo_id=space_id,
            repo_type="space"
        )
        api.upload_file(
            path_or_fileobj=req_path,
            path_in_repo="requirements.txt",
            repo_id=space_id,
            repo_type="space"
        )
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=space_id,
            repo_type="space"
        )
    
    space_url = f"https://huggingface.co/spaces/{space_id}"
    
    print("✅ Training space created successfully!")
    print(f"🔗 Space URL: {space_url}")
    print()
    print("🚀 NEXT STEPS:")
    print(f"1. Visit: {space_url}")
    print("2. Wait for the space to build (~2-3 minutes)")
    print("3. Click 'Setup Model' (takes ~5 minutes)")
    print("4. Click 'Start Training' (takes ~2-3 hours)")
    print("5. Your model will be automatically uploaded to HF Hub!")
    
    return space_url

if __name__ == "__main__":
    create_pashto_chat_training_space()
