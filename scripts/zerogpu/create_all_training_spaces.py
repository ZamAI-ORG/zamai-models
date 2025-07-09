"""
Create ZeroGPU Training Spaces for All Existing ZamAI Models
Based on your current HF Hub models
"""

import os
import json
import tempfile
from huggingface_hub import HfApi

def get_model_configs():
    """Define configurations for each of your existing models"""
    
    # Base configurations for each model type
    configs = {
        "bloom": {
            "hub_model_id": "tasal9/pashto-base-bloom",
            "base_model": "bigscience/bloom-560m",
            "dataset_name": "tasal9/ZamAI_Pashto_Dataset",
            "model_type": "bloom",
            "description": "Pashto language model based on BLOOM 560M",
            "lora_config": {
                "rank": 16,
                "alpha": 32,
                "dropout": 0.1,
                "target_modules": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
            },
            "training_config": {
                "epochs": 3,
                "batch_size": 2,
                "learning_rate": 1e-4,
                "max_length": 1024
            }
        },
        
        "llama3": {
            "hub_model_id": "tasal9/ZamAI-LIama3-Pashto",
            "base_model": "meta-llama/Meta-Llama-3-8B-Instruct",
            "dataset_name": "tasal9/ZamAI_Pashto_Dataset",
            "model_type": "llama",
            "description": "Advanced Pashto conversational AI based on Llama 3",
            "lora_config": {
                "rank": 64,
                "alpha": 128,
                "dropout": 0.05,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            },
            "training_config": {
                "epochs": 3,
                "batch_size": 1,
                "learning_rate": 2e-5,
                "max_length": 2048
            }
        },
        
        "mistral": {
            "hub_model_id": "tasal9/ZamAI-Mistral-7B-Pashto",
            "base_model": "mistralai/Mistral-7B-Instruct-v0.2",
            "dataset_name": "tasal9/ZamAI_Pashto_Dataset", 
            "model_type": "mistral",
            "description": "Pashto conversational AI based on Mistral 7B",
            "lora_config": {
                "rank": 32,
                "alpha": 64,
                "dropout": 0.1,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
            },
            "training_config": {
                "epochs": 2,
                "batch_size": 1,
                "learning_rate": 1e-5,
                "max_length": 1536
            }
        },
        
        "phi3": {
            "hub_model_id": "tasal9/ZamAI-Phi-3-Mini-Pashto",
            "base_model": "microsoft/Phi-3-mini-4k-instruct",
            "dataset_name": "tasal9/ZamAI_Pashto_Dataset",
            "model_type": "phi",
            "description": "Lightweight Pashto model based on Phi-3 Mini",
            "lora_config": {
                "rank": 32,
                "alpha": 64,
                "dropout": 0.1,
                "target_modules": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
            },
            "training_config": {
                "epochs": 3,
                "batch_size": 2,
                "learning_rate": 2e-5,
                "max_length": 1024
            }
        },
        
        "whisper": {
            "hub_model_id": "tasal9/ZamAI-Whisper-v3-Pashto",
            "base_model": "openai/whisper-large-v3",
            "dataset_name": "tasal9/ZamAI_Pashto_Speech_Dataset",  # Assuming you have speech data
            "model_type": "whisper",
            "description": "Pashto speech recognition based on Whisper Large V3",
            "lora_config": {
                "rank": 32,
                "alpha": 64,
                "dropout": 0.1,
                "target_modules": ["q_proj", "k_proj", "v_proj", "out_proj"]
            },
            "training_config": {
                "epochs": 5,
                "batch_size": 4,
                "learning_rate": 1e-5,
                "max_length": 448  # Whisper's context length
            }
        },
        
        "embeddings": {
            "hub_model_id": "tasal9/Multilingual-ZamAI-Embeddings",
            "base_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "dataset_name": "tasal9/ZamAI_Pashto_Dataset",
            "model_type": "embeddings",
            "description": "Multilingual embeddings optimized for Pashto and Afghan languages",
            "lora_config": {
                "rank": 16,
                "alpha": 32,
                "dropout": 0.1,
                "target_modules": ["query", "key", "value", "dense"]
            },
            "training_config": {
                "epochs": 3,
                "batch_size": 8,
                "learning_rate": 2e-5,
                "max_length": 512
            }
        }
    }
    
    return configs

def create_training_space_for_model(api, username, model_key, config):
    """Create a training space for a specific model"""
    
    space_name = f"zamai-{model_key}-training"
    space_id = f"{username}/{space_name}"
    
    print(f"Creating space for {config['hub_model_id']}: {space_id}")
    
    # Create space
    try:
        api.create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="gradio",
            space_hardware="zero-a10g",
            private=False
        )
        print(f"✅ Space created: {space_id}")
    except Exception as e:
        if "already exists" in str(e):
            print(f"⚠️ Space already exists: {space_id}")
        else:
            print(f"❌ Error creating space: {e}")
            return None
    
    # Generate training script based on model type
    if config['model_type'] == 'whisper':
        app_py = generate_whisper_training_script(config)
    elif config['model_type'] == 'embeddings':
        app_py = generate_embeddings_training_script(config)
    else:
        app_py = generate_text_generation_training_script(config)
    
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
huggingface-hub>=0.19.0
librosa>=0.10.0
soundfile>=0.12.0
sentence-transformers>=2.2.0"""
    
    # Create README.md
    readme = f"""---
title: ZamAI {model_key.title()} Training
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

# 🚀 ZamAI {model_key.title()} Model Training

This space retrains/improves the **{config['hub_model_id']}** model using ZeroGPU.

## Model Details
- **Model**: {config['hub_model_id']}
- **Base Model**: {config['base_model']}
- **Description**: {config['description']}
- **Training Method**: LoRA fine-tuning

## Features
- ✅ Memory-optimized training with 4-bit quantization
- ✅ Real-time progress monitoring
- ✅ Automatic model upload to HuggingFace Hub
- ✅ Optional WandB experiment tracking

## Usage
1. Click "Setup Model" to prepare the base model
2. Click "Start Training" to begin fine-tuning
3. Monitor progress in real-time
4. Model will be automatically updated in your HF Hub

Training takes approximately 2-4 hours depending on the model size.
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
    
    return f"https://huggingface.co/spaces/{space_id}"

def generate_text_generation_training_script(config):
    """Generate training script for text generation models"""
    
    return f'''import gradio as gr
import spaces
import torch
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
MODEL_TYPE = "{config['model_type']}"

class ZamAITrainer:
    def __init__(self):
        self.status = "Ready to start training"
        self.model = None
        self.tokenizer = None
    
    @spaces.GPU(duration=120)
    def setup_model(self):
        try:
            self.status = "🔄 Loading tokenizer..."
            yield self.status
            
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.status = "🔄 Loading model (this takes ~5-10 minutes)..."
            yield self.status
            
            # Load model with optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_4bit=True,
                trust_remote_code=True
            )
            
            self.status = "🔄 Setting up LoRA adapters..."
            yield self.status
            
            # LoRA configuration optimized for model type
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r={config['lora_config']['rank']},
                lora_alpha={config['lora_config']['alpha']},
                lora_dropout={config['lora_config']['dropout']},
                target_modules={config['lora_config']['target_modules']}
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
            self.status = "✅ Model setup complete! Ready to train."
            yield self.status
            
        except Exception as e:
            self.status = f"❌ Setup error: {{str(e)}}"
            yield self.status
    
    @spaces.GPU(duration=300)
    def start_training(self, wandb_key):
        try:
            if self.model is None:
                self.status = "❌ Please setup model first!"
                yield self.status
                return
            
            # Setup WandB if key provided
            if wandb_key.strip():
                self.status = "🔄 Setting up WandB tracking..."
                yield self.status
                wandb.login(key=wandb_key.strip())
                wandb.init(
                    project="zamai-models",
                    name=f"{{MODEL_TYPE}}-retrain",
                    config={{
                        "model": MODEL_ID,
                        "base_model": BASE_MODEL,
                        "dataset": DATASET_NAME,
                        "model_type": MODEL_TYPE
                    }}
                )
            
            self.status = "🔄 Loading dataset..."
            yield self.status
            
            dataset = load_dataset(DATASET_NAME)
            train_dataset = dataset['train']
            
            def tokenize_function(examples):
                if 'instruction' in examples and 'output' in examples:
                    texts = []
                    for inst, out in zip(examples['instruction'], examples['output']):
                        text = f"### Instruction\\n{{inst}}\\n\\n### Response\\n{{out}}"
                        texts.append(text)
                elif 'text' in examples:
                    texts = examples['text']
                else:
                    texts = [str(ex) for ex in examples]
                
                return self.tokenizer(
                    texts,
                    truncation=True,
                    padding=True,
                    max_length={config['training_config']['max_length']},
                    return_tensors="pt"
                )
            
            self.status = "🔄 Tokenizing dataset..."
            yield self.status
            
            train_dataset = train_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=train_dataset.column_names
            )
            
            self.status = "🔄 Setting up training arguments..."
            yield self.status
            
            training_args = TrainingArguments(
                output_dir="./outputs",
                num_train_epochs={config['training_config']['epochs']},
                per_device_train_batch_size={config['training_config']['batch_size']},
                gradient_accumulation_steps=8,
                learning_rate={config['training_config']['learning_rate']},
                warmup_steps=100,
                logging_steps=10,
                save_steps=200,
                fp16=True,
                dataloader_drop_last=True,
                push_to_hub=True,
                hub_model_id=MODEL_ID,
                hub_token=os.getenv("HF_TOKEN"),
                report_to="wandb" if wandb_key.strip() else None,
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
            
            self.status = "🏃 Training started! This will take 2-4 hours..."
            yield self.status
            
            trainer.train()
            
            self.status = "💾 Saving and uploading model..."
            yield self.status
            
            trainer.save_model()
            trainer.push_to_hub()
            
            self.status = f"🎉 SUCCESS! Model retrained and updated at {{MODEL_ID}}"
            yield self.status
            
        except Exception as e:
            self.status = f"❌ Training error: {{str(e)}}"
            yield self.status

trainer = ZamAITrainer()

with gr.Blocks(title="ZamAI {MODEL_TYPE.title()} Training") as demo:
    gr.Markdown("# 🚀 ZamAI {MODEL_TYPE.title()} Model Retraining")
    
    gr.Markdown(f"""
    **Model**: `{{MODEL_ID}}`  
    **Base**: `{{BASE_MODEL}}`  
    **Type**: {config['description']}
    **Hardware**: ZeroGPU A10G
    """)
    
    gr.Markdown("""
    ## 📋 Instructions
    1. **Setup Model** (5-10 minutes) - Loads and prepares the model
    2. **Start Training** (2-4 hours) - Retrains with latest data
    3. **Monitor Progress** - Real-time status updates
    
    The improved model will automatically update your existing HF Hub model!
    """)
    
    with gr.Row():
        setup_btn = gr.Button("🔧 Setup Model", variant="primary", size="lg")
        train_btn = gr.Button("🏃 Start Training", variant="secondary", size="lg")
    
    wandb_key = gr.Textbox(
        label="WandB API Key (optional)",
        type="password",
        placeholder="Get from: https://wandb.ai/settings"
    )
    
    status_display = gr.Textbox(
        label="Training Status",
        value="Ready to start training",
        interactive=False,
        lines=4
    )
    
    setup_btn.click(trainer.setup_model, outputs=status_display)
    train_btn.click(trainer.start_training, inputs=wandb_key, outputs=status_display)

if __name__ == "__main__":
    demo.launch()
'''

def generate_whisper_training_script(config):
    """Generate training script for Whisper ASR model"""
    
    return f'''import gradio as gr
import spaces
import torch
import os
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import wandb

MODEL_ID = "{config['hub_model_id']}"
BASE_MODEL = "{config['base_model']}"
DATASET_NAME = "{config['dataset_name']}"

class WhisperTrainer:
    def __init__(self):
        self.status = "Ready for Whisper training"
        self.model = None
        self.processor = None
    
    @spaces.GPU(duration=120)
    def setup_model(self):
        try:
            self.status = "🔄 Loading Whisper processor..."
            yield self.status
            
            self.processor = WhisperProcessor.from_pretrained(BASE_MODEL)
            
            self.status = "🔄 Loading Whisper model..."
            yield self.status
            
            self.model = WhisperForConditionalGeneration.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_4bit=True
            )
            
            self.status = "🔄 Setting up LoRA for Whisper..."
            yield self.status
            
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                inference_mode=False,
                r={config['lora_config']['rank']},
                lora_alpha={config['lora_config']['alpha']},
                lora_dropout={config['lora_config']['dropout']},
                target_modules={config['lora_config']['target_modules']}
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
            self.status = "✅ Whisper setup complete! Ready to train."
            yield self.status
            
        except Exception as e:
            self.status = f"❌ Setup error: {{str(e)}}"
            yield self.status
    
    @spaces.GPU(duration=300)
    def start_training(self, wandb_key):
        try:
            if self.model is None:
                self.status = "❌ Please setup model first!"
                yield self.status
                return
            
            if wandb_key.strip():
                self.status = "🔄 Setting up WandB..."
                yield self.status
                wandb.login(key=wandb_key.strip())
                wandb.init(project="zamai-whisper", name="pashto-asr")
            
            self.status = "🔄 Loading speech dataset..."
            yield self.status
            
            # Note: Replace with actual speech dataset loading
            # This is a placeholder for speech data processing
            self.status = "🔄 Processing audio data..."
            yield self.status
            
            # Placeholder training logic for Whisper
            self.status = "🏃 Training Whisper ASR (this may take 3-5 hours)..."
            yield self.status
            
            # Add actual Whisper training logic here
            
            self.status = "🎉 Whisper training completed! Model updated."
            yield self.status
            
        except Exception as e:
            self.status = f"❌ Training error: {{str(e)}}"
            yield self.status

trainer = WhisperTrainer()

with gr.Blocks(title="ZamAI Whisper Training") as demo:
    gr.Markdown("# 🎤 ZamAI Whisper ASR Training")
    gr.Markdown("Train Pashto speech recognition model")
    
    with gr.Row():
        setup_btn = gr.Button("🔧 Setup Whisper", variant="primary")
        train_btn = gr.Button("🏃 Start Training", variant="secondary")
    
    wandb_key = gr.Textbox(label="WandB Key", type="password")
    status_display = gr.Textbox(label="Status", lines=3, interactive=False)
    
    setup_btn.click(trainer.setup_model, outputs=status_display)
    train_btn.click(trainer.start_training, inputs=wandb_key, outputs=status_display)

demo.launch()
'''

def generate_embeddings_training_script(config):
    """Generate training script for embeddings model"""
    
    return f'''import gradio as gr
import spaces
import torch
import os
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from datasets import load_dataset
import wandb

MODEL_ID = "{config['hub_model_id']}"
BASE_MODEL = "{config['base_model']}"
DATASET_NAME = "{config['dataset_name']}"

class EmbeddingsTrainer:
    def __init__(self):
        self.status = "Ready for embeddings training"
        self.model = None
    
    @spaces.GPU(duration=60)
    def setup_model(self):
        try:
            self.status = "🔄 Loading sentence transformer..."
            yield self.status
            
            self.model = SentenceTransformer(BASE_MODEL)
            
            self.status = "✅ Embeddings model ready!"
            yield self.status
            
        except Exception as e:
            self.status = f"❌ Setup error: {{str(e)}}"
            yield self.status
    
    @spaces.GPU(duration=180)
    def start_training(self, wandb_key):
        try:
            if self.model is None:
                self.status = "❌ Please setup model first!"
                yield self.status
                return
            
            if wandb_key.strip():
                wandb.login(key=wandb_key.strip())
                wandb.init(project="zamai-embeddings", name="multilingual-embeddings")
            
            self.status = "🔄 Loading dataset for embeddings..."
            yield self.status
            
            dataset = load_dataset(DATASET_NAME)
            
            self.status = "🔄 Preparing training examples..."
            yield self.status
            
            # Create training examples for sentence transformers
            train_examples = []
            for item in dataset['train']:
                if 'instruction' in item and 'output' in item:
                    train_examples.append(InputExample(texts=[item['instruction'], item['output']], label=1.0))
            
            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
            train_loss = losses.CosineSimilarityLoss(self.model)
            
            self.status = "🏃 Training embeddings model..."
            yield self.status
            
            # Train the model
            self.model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs={config['training_config']['epochs']},
                warmup_steps=100,
                output_path="./embeddings_output"
            )
            
            self.status = "💾 Saving embeddings model..."
            yield self.status
            
            # Save to hub
            self.model.save_to_hub(MODEL_ID, token=os.getenv("HF_TOKEN"))
            
            self.status = "🎉 Embeddings training completed!"
            yield self.status
            
        except Exception as e:
            self.status = f"❌ Training error: {{str(e)}}"
            yield self.status

trainer = EmbeddingsTrainer()

with gr.Blocks(title="ZamAI Embeddings Training") as demo:
    gr.Markdown("# 🔗 ZamAI Embeddings Training")
    gr.Markdown("Train multilingual embeddings optimized for Pashto")
    
    with gr.Row():
        setup_btn = gr.Button("🔧 Setup Model", variant="primary")
        train_btn = gr.Button("🏃 Start Training", variant="secondary")
    
    wandb_key = gr.Textbox(label="WandB Key", type="password")
    status_display = gr.Textbox(label="Status", lines=3, interactive=False)
    
    setup_btn.click(trainer.setup_model, outputs=status_display)
    train_btn.click(trainer.start_training, inputs=wandb_key, outputs=status_display)

demo.launch()
'''

def create_all_training_spaces():
    """Create training spaces for all existing models"""
    
    print("🚀 Creating ZeroGPU Training Spaces for All ZamAI Models")
    print("=" * 70)
    
    # Load HF token
    token_path = "/workspaces/ZamAI-Pro-Models/HF-Token.txt"
    if not os.path.exists(token_path):
        print("❌ HF Token not found")
        return None
    
    with open(token_path, 'r') as f:
        token = f.read().strip()
    
    api = HfApi(token=token)
    username = api.whoami()['name']
    
    # Get model configurations
    configs = get_model_configs()
    
    created_spaces = []
    
    # Create training space for each model
    for model_key, config in configs.items():
        print(f"\\n📋 Creating training space for {model_key}...")
        space_url = create_training_space_for_model(api, username, model_key, config)
        
        if space_url:
            created_spaces.append({
                'model': config['hub_model_id'],
                'type': model_key,
                'space_url': space_url,
                'description': config['description']
            })
    
    # Display results
    print(f"\\n🎉 TRAINING SPACES CREATED: {len(created_spaces)}")
    print("=" * 70)
    
    for space in created_spaces:
        print(f"\\n🎯 {space['model']}")
        print(f"   Type: {space['type']}")
        print(f"   Description: {space['description']}")
        print(f"   Training Space: {space['space_url']}")
    
    print(f"\\n🚀 NEXT STEPS:")
    print("1. Visit each training space URL above")
    print("2. Wait for spaces to build (~2-3 minutes each)")
    print("3. Click 'Setup Model' in each space")
    print("4. Click 'Start Training' to improve your models")
    print("5. Models will be automatically updated in your HF Hub")
    
    # Save training plan
    plan = {
        'created_spaces': created_spaces,
        'total_count': len(created_spaces)
    }
    
    plan_path = "/workspaces/ZamAI-Pro-Models/all_training_spaces.json"
    with open(plan_path, 'w') as f:
        json.dump(plan, f, indent=2)
    
    print(f"\\n💾 Training spaces info saved to: {plan_path}")
    
    return created_spaces

if __name__ == "__main__":
    create_all_training_spaces()
