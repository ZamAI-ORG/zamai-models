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
                        text = f"### Instruction\n{inst}\n\n### Response\n{out}"
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
