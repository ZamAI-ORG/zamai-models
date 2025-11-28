# 🚀 ZeroGPU Training Setup Guide for ZamAI Models

This guide will help you set up ZeroGPU training spaces for all your ZamAI models on HuggingFace Spaces.

## 📋 Overview

We've created ready-to-upload files for **7 training spaces**:

1. **bloom-pashto** - Pashto BLOOM model training
2. **llama3-pashto** - Pashto Llama3 model training  
3. **mistral-pashto** - Pashto Mistral model training
4. **phi3-pashto** - Pashto Phi-3 model training
5. **mt5-pashto** - Pashto MT5 translation (English↔Pashto) training
6. **whisper-pashto** - Pashto Whisper ASR training
7. **embeddings-multilingual** - Multilingual embeddings training

### 🆕 mT5 Translation Space Highlights
- 🔁 Built-in EN↔PS translation demo powered by `tasal9/ZamAI-mT5-Pashto`
- ⚙️ LoRA training pipeline that wraps `google/mt5-base` with ZeroGPU-friendly defaults
- 🧹 Automatic dataset column detection for `input/output`, `en/ps`, or `prompt/completion` schemas
- ☁️ Optional push-to-hub so fine-tuned adapters land back in your model repo immediately

## 🎯 Quick Setup (Manual Upload)

### For Each Model:

1. **Create a New Space:**
   - Go to https://huggingface.co/new-space
   - Choose a name: `zamai-[model-name]-training` (e.g., `zamai-llama3-pashto-training`)
   - Set SDK to **Gradio**
   - Set Hardware to **ZeroGPU - A10G** 
   - Make it Public (recommended)

2. **Upload Files:**
   - Navigate to the corresponding folder in `zerogpu_files/[model-name]/`
   - Upload all 3 files: `app.py`, `requirements.txt`, `README.md`

3. **Wait for Build:**
   - The space will automatically build and deploy
   - Wait for the "Running" status

4. **Start Training:**
   - Visit your space URL
   - Use the "Test" tab to verify the model works
   - Use the "Training" tab to fine-tune with your datasets

#### app.py (Main training script):
```python
import gradio as gr
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

# Configuration
MODEL_ID = "tasal9/zamai-pashto-chat-8b"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_NAME = "tasal9/ZamAI_Pashto_Dataset"

class ZamAITrainer:
    def __init__(self):
        self.status = "Ready to start training"
        self.model = None
        self.tokenizer = None
    
    @spaces.GPU(duration=120)  # 2 hours for setup
    def setup_model(self):
        try:
            self.status = "🔄 Loading tokenizer..."
            yield self.status
            
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.status = "🔄 Loading model (this takes ~5 minutes)..."
            yield self.status
            
            # Load model with 4-bit quantization for memory efficiency
            self.model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_4bit=True,
                trust_remote_code=True
            )
            
            self.status = "🔄 Setting up LoRA adapters..."
            yield self.status
            
            # LoRA configuration
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=64,  # rank
                lora_alpha=128,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
            
            self.model = get_peft_model(self.model, lora_config)

            self.model.print_trainable_parameters()
            
            self.status = "✅ Model setup complete! Ready to train."
            yield self.status
            
        except Exception as e:
            self.status = f"❌ Setup error: {str(e)}"
            yield self.status
    
    @spaces.GPU(duration=300)  # 5 hours for training (generous time)
    def start_training(self, wandb_key):
        try:
            if self.model is None:
                self.status = "❌ Please setup model first!"
                yield self.status
                return
            
            # Optional WandB setup
            if wandb_key.strip():
                self.status = "🔄 Setting up WandB tracking..."
                yield self.status
                wandb.login(key=wandb_key.strip())
                wandb.init(
                    project="zamai-models",
                    name="pashto-chat-8b",
                    config={
                        "model": MODEL_ID,
                        "base_model": BASE_MODEL,
                        "dataset": DATASET_NAME
                    }
                )
            
            self.status = "🔄 Loading dataset..."
            yield self.status
            
            # Load dataset
            dataset = load_dataset(DATASET_NAME)
            train_dataset = dataset['train']
            
            # Tokenization function
            def tokenize_function(examples):
                # Handle instruction format
                if 'instruction' in examples and 'output' in examples:
                    texts = []
                    for inst, out in zip(examples['instruction'], examples['output']):
                        text = f"### Instruction\\n{inst}\\n\\n### Response\\n{out}"
                        texts.append(text)
                elif 'text' in examples:
                    texts = examples['text']
                else:
                    texts = [str(ex) for ex in examples]
                
                return self.tokenizer(
                    texts,
                    truncation=True,
                    padding=True,
                    max_length=2048,
                    return_tensors="pt"
                )
            
            self.status = "🔄 Tokenizing dataset (this may take a while)..."
            yield self.status
            
            # Tokenize dataset
            train_dataset = train_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=train_dataset.column_names
            )
            
            self.status = "🔄 Setting up training arguments..."
            yield self.status
            
            # Training arguments optimized for ZeroGPU
            training_args = TrainingArguments(
                output_dir="./outputs",
                num_train_epochs=3,
                per_device_train_batch_size=1,  # Small batch for memory
                gradient_accumulation_steps=8,   # Increase effective batch size
                learning_rate=2e-5,
                warmup_steps=500,
                logging_steps=10,
                save_steps=200,
                fp16=True,  # Mixed precision for speed
                dataloader_drop_last=True,
                push_to_hub=True,
                hub_model_id=MODEL_ID,
                hub_token=os.getenv("HF_TOKEN"),
                report_to="wandb" if wandb_key.strip() else None,
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
            
            # Create trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=data_collator,
            )
            
            self.status = "🏃 Training started! This will take 2-4 hours. Check back later..."
            yield self.status
            
            # Start training
            trainer.train()
            
            self.status = "💾 Saving and uploading model..."
            yield self.status
            
            # Save and upload
            trainer.save_model()
            trainer.push_to_hub()
            
            self.status = f"🎉 SUCCESS! Model trained and uploaded to {MODEL_ID}"
            yield self.status
            
        except Exception as e:
            self.status = f"❌ Training error: {str(e)}"
            yield self.status

# Initialize trainer
trainer = ZamAITrainer()

# Gradio interface
with gr.Blocks(title="ZamAI Pashto Chat Training") as demo:
    gr.Markdown("# 🚀 ZamAI Pashto Chat Model Training on ZeroGPU")
    
    gr.Markdown(f"""
    **Training Model**: `{MODEL_ID}`  
    **Base Model**: `{BASE_MODEL}`  
    **Dataset**: `{DATASET_NAME}`  
    **Hardware**: ZeroGPU A10G
    """)
    
    gr.Markdown("""
    ## 📋 Instructions
    1. **Setup Model** (5-10 minutes) - Loads the base model and sets up LoRA
    2. **Start Training** (2-4 hours) - Fine-tunes the model on Pashto data
    3. **Monitor Progress** - Watch the status updates below
    
    The trained model will automatically be uploaded to your HuggingFace Hub!
    """)
    
    with gr.Row():
        setup_btn = gr.Button("🔧 Setup Model", variant="primary", size="lg")
        train_btn = gr.Button("🏃 Start Training", variant="secondary", size="lg")
    
    with gr.Row():
        wandb_key = gr.Textbox(
            label="WandB API Key (optional - for experiment tracking)",
            type="password",
            placeholder="Get from: https://wandb.ai/settings",
            scale=3
        )
        gr.Markdown("*Optional: Add your WandB key to track training metrics*", scale=1)
    
    status_display = gr.Textbox(
        label="Training Status",
        value="Ready to start training",
        interactive=False,
        lines=3
    )
    
    # Event handlers
    setup_btn.click(trainer.setup_model, outputs=status_display)
    train_btn.click(trainer.start_training, inputs=wandb_key, outputs=status_display)

if __name__ == "__main__":
    demo.launch()
```

#### requirements.txt:
```
torch>=2.0.0
transformers>=4.36.0
datasets>=2.14.0
accelerate>=0.24.0
peft>=0.6.0
bitsandbytes>=0.41.0
gradio>=4.8.0
spaces>=0.19.0
wandb>=0.16.0
huggingface-hub>=0.19.0
```

#### README.md:
```markdown
---
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

This space trains a Pashto conversational AI model using ZeroGPU infrastructure.

## Features
- ✅ Automatic model setup and LoRA fine-tuning
- ✅ Memory-optimized training with 4-bit quantization
- ✅ Real-time progress monitoring
- ✅ Automatic model upload to HuggingFace Hub
- ✅ Optional WandB experiment tracking

## Usage
1. Click "Setup Model" to prepare the base model
2. Optionally add your WandB API key for tracking
3. Click "Start Training" to begin fine-tuning
4. Wait 2-4 hours for training to complete
5. Your model will be available at `tasal9/zamai-pashto-chat-8b`

Training uses Meta's Llama 3.1 8B as the base model and fine-tunes it on Pashto conversational data using LoRA adapters.
```

## 🎯 Option 2: Automated Script (If Token Works)

If your HF token is working, you can run:

```bash
python scripts/zerogpu/create_training_space.py
```

This will automatically create the space and upload all files.

## 🚀 Quick Start Steps

1. **Check your HF token** has write permissions and ZeroGPU access
2. **Create the space manually** using the files above
3. **Visit your space** at `https://huggingface.co/spaces/YOUR_USERNAME/zamai-pashto-chat-training`
4. **Wait for build** (~2-3 minutes)
5. **Click "Setup Model"** (~5-10 minutes)
6. **Click "Start Training"** (~2-4 hours)
7. **Your model will be ready** at `tasal9/zamai-pashto-chat-8b`!

## ⚡ Benefits of ZeroGPU Training

- ✅ **No local GPU needed** - Uses HF's A10G GPUs
- ✅ **Automatic scaling** - Resources allocated on demand
- ✅ **Built-in monitoring** - Real-time progress tracking
- ✅ **Auto-upload** - Models pushed to Hub automatically
- ✅ **Cost-effective** - Part of your HF Pro subscription
- ✅ **Reliable** - Professional infrastructure

## 📊 Expected Results

After training completes, you'll have:
- ✅ `tasal9/zamai-pashto-chat-8b` - Your trained Pashto chat model
- ✅ Ready for inference via HF Inference API
- ✅ Ready for deployment to Inference Endpoints
- ✅ Available for integration in your applications

This is **significantly easier** than local training and leverages HF's professional infrastructure!
