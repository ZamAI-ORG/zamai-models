#!/usr/bin/env python3
"""
🇦🇫 ZamAI Master Training Hub
Create a comprehensive training space using your best dataset
"""

from huggingface_hub import HfApi
import os

def create_master_training_hub():
    """Create the ultimate ZamAI training space"""
    
    with open("/workspaces/ZamAI-Pro-Models/HF-Token.txt", "r") as f:
        token = f.read().strip()
    
    api = HfApi(token=token)
    space_id = "tasal9/zamai-master-training-hub"
    
    print(f"🚀 Creating Master Training Hub: {space_id}")
    
    try:
        # Create space with A10G hardware
        api.create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="gradio",
            space_hardware="a10g-small",
            exist_ok=True
        )
        
        # Ultimate training app
        master_app = '''import gradio as gr
import os
import spaces
from huggingface_hub import HfApi
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
import torch
import json

class ZamAIMasterTrainer:
    def __init__(self):
        self.api = HfApi(token=os.getenv("HF_TOKEN"))
        self.username = "tasal9"
        
    @spaces.GPU
    def ultimate_training(self, base_model, dataset_choice, model_name, learning_rate, 
                         epochs, batch_size, lora_rank, max_length):
        """Ultimate ZamAI training with all optimizations"""
        try:
            yield "🚀 ZamAI Master Trainer initializing..."
            
            # Dataset mapping
            datasets = {
                "ZamAI Main Dataset": "tasal9/ZamAI_Pashto_Dataset",
                "High Quality Dataset": "tasal9/ZamAI-Pashto-High-Qualituly-Dataset", 
                "Cleaned Dataset": "tasal9/ZamAI-Pashto-Dataset-Cleaned",
                "Original Pashto": "tasal9/Pashto_Dataset"
            }
            
            dataset_id = datasets[dataset_choice]
            yield f"📚 Loading dataset: {dataset_id}"
            
            # Load dataset
            dataset = load_dataset(dataset_id)
            yield f"✅ Dataset loaded: {len(dataset['train'])} examples"
            
            # Load model
            yield f"🤖 Loading base model: {base_model}"
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # LoRA setup
            yield f"⚙️ Setting up LoRA (rank={lora_rank})..."
            lora_config = LoraConfig(
                r=int(lora_rank),
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            model = get_peft_model(model, lora_config)
            trainable_params = model.print_trainable_parameters()
            yield f"📊 Trainable parameters: {trainable_params}"
            
            # Advanced data preprocessing
            yield "🔄 Advanced data preprocessing..."
            
            def advanced_format(examples):
                formatted = []
                instructions = examples.get("instruction", examples.get("input", []))
                outputs = examples.get("output", examples.get("response", []))
                
                for inst, out in zip(instructions, outputs):
                    # ZamAI specialized prompt format
                    prompt = f"""### د ZamAI هوښیار ماډل:
د لاندې پوښتنې لپاره په پښتو کې یو بشپړ او دقیق ځواب ورکړئ:

### پوښتنه:
{inst}

### ځواب:
{out}<|endoftext|>"""
                    formatted.append(prompt)
                
                return {"text": formatted}
            
            dataset = dataset.map(advanced_format, batched=True)
            
            def tokenize_function(examples):
                return tokenizer(
                    examples["text"],
                    truncation=True,
                    padding=True,
                    max_length=int(max_length)
                )
            
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            
            # Advanced training setup
            output_model = f"{model_name}-{base_model.split('/')[-1]}-lora"
            
            training_args = TrainingArguments(
                output_dir=f"./zamai-outputs/{output_model}",
                num_train_epochs=int(epochs),
                per_device_train_batch_size=int(batch_size),
                gradient_accumulation_steps=8,
                warmup_steps=200,
                learning_rate=float(learning_rate),
                fp16=True,
                logging_steps=5,
                save_steps=100,
                eval_steps=100,
                evaluation_strategy="steps",
                save_total_limit=3,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                push_to_hub=True,
                hub_model_id=f"{self.username}/{output_model}",
                hub_token=os.getenv("HF_TOKEN"),
                report_to="none",
                dataloader_pin_memory=False,
                remove_unused_columns=False
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            )
            
            # Split for evaluation
            if len(tokenized_dataset["train"]) > 100:
                train_test = tokenized_dataset["train"].train_test_split(test_size=0.1)
                train_dataset = train_test["train"]
                eval_dataset = train_test["test"]
            else:
                train_dataset = tokenized_dataset["train"]
                eval_dataset = None
            
            # Trainer with callbacks
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator,
                tokenizer=tokenizer,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if eval_dataset else []
            )
            
            # Start training
            yield "🏋️ Starting advanced LoRA training..."
            yield f"📊 Training on {len(train_dataset)} examples"
            if eval_dataset:
                yield f"📊 Evaluating on {len(eval_dataset)} examples"
            
            trainer.train()
            
            # Save model
            yield "💾 Saving trained model..."
            trainer.save_model()
            
            # Push to hub
            yield "📤 Uploading to Hugging Face Hub..."
            trainer.push_to_hub(commit_message=f"ZamAI {output_model} - Advanced LoRA Training")
            
            # Generate sample
            yield "🧪 Testing model with sample generation..."
            model.eval()
            test_prompt = "سلام وروره! زه د افغانستان د"
            inputs = tokenizer(test_prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            sample_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            yield f"🧪 Sample output: {sample_output}"
            
            # Final success message
            yield f"""
✅ ZamAI Master Training Completed Successfully!

🎯 **Model Details:**
- **Name**: tasal9/{output_model}
- **Base**: {base_model}
- **Dataset**: {dataset_id}
- **Method**: LoRA Fine-tuning
- **Parameters**: {trainable_params}

🚀 **Model URL**: https://huggingface.co/tasal9/{output_model}

🔥 **Ready for use in your ZamAI applications!**
"""
            
        except Exception as e:
            yield f"❌ Training failed: {str(e)}"
            yield f"📋 Error details: Please check your configuration and try again."

trainer = ZamAIMasterTrainer()

# Ultimate Gradio interface
with gr.Blocks(title="🇦🇫 ZamAI Master Training Hub", theme=gr.themes.Soft()) as demo:
    gr.HTML("""
    <div style="text-align: center; padding: 20px;">
        <h1>🇦🇫 ZamAI Master Training Hub</h1>
        <h3>Advanced LoRA Fine-tuning for Pashto AI Models</h3>
        <p>Train state-of-the-art ZamAI models with your datasets</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎯 Model Configuration")
            
            base_model = gr.Dropdown(
                choices=[
                    "meta-llama/Llama-3.1-8B-Instruct",
                    "meta-llama/Llama-2-7b-chat-hf",
                    "mistralai/Mistral-7B-Instruct-v0.2",
                    "microsoft/phi-3-mini-4k-instruct",
                    "microsoft/DialoGPT-medium"
                ],
                label="🤖 Base Model",
                value="meta-llama/Llama-3.1-8B-Instruct",
                info="Choose the foundation model to fine-tune"
            )
            
            dataset_choice = gr.Dropdown(
                choices=[
                    "ZamAI Main Dataset",
                    "High Quality Dataset", 
                    "Cleaned Dataset",
                    "Original Pashto"
                ],
                label="📚 Training Dataset",
                value="High Quality Dataset",
                info="Select your ZamAI dataset"
            )
            
            model_name = gr.Textbox(
                label="📝 Output Model Name",
                value="zamai-pashto-ultimate-v1",
                placeholder="zamai-model-name",
                info="Name for your trained model"
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Training Parameters")
            
            learning_rate = gr.Slider(
                minimum=1e-5,
                maximum=5e-4,
                value=2e-4,
                step=1e-5,
                label="📈 Learning Rate",
                info="Controls training speed"
            )
            
            epochs = gr.Slider(
                minimum=1,
                maximum=10,
                value=3,
                step=1,
                label="🔄 Training Epochs",
                info="Number of training cycles"
            )
            
            batch_size = gr.Slider(
                minimum=1,
                maximum=4,
                value=1,
                step=1,
                label="📦 Batch Size",
                info="Samples per training step"
            )
            
            lora_rank = gr.Slider(
                minimum=8,
                maximum=64,
                value=16,
                step=8,
                label="🎯 LoRA Rank",
                info="Adaptation complexity"
            )
            
            max_length = gr.Slider(
                minimum=256,
                maximum=2048,
                value=512,
                step=128,
                label="📏 Max Sequence Length",
                info="Maximum token length"
            )
    
    train_btn = gr.Button(
        "🚀 Start Ultimate Training", 
        variant="primary", 
        size="lg",
        elem_classes=["big-button"]
    )
    
    with gr.Row():
        progress = gr.Textbox(
            label="📊 Training Progress & Results",
            lines=25,
            interactive=False,
            show_copy_button=True,
            max_lines=30
        )
    
    train_btn.click(
        trainer.ultimate_training,
        inputs=[base_model, dataset_choice, model_name, learning_rate, epochs, batch_size, lora_rank, max_length],
        outputs=progress
    )
    
    gr.HTML("""
    <div style="margin-top: 20px; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
        <h3>🔥 Advanced Features:</h3>
        <ul>
            <li><strong>LoRA Fine-tuning</strong>: Memory-efficient training</li>
            <li><strong>A10G GPU</strong>: High-performance training</li>
            <li><strong>Early Stopping</strong>: Prevents overfitting</li>
            <li><strong>Auto-Evaluation</strong>: Built-in validation</li>
            <li><strong>Hub Integration</strong>: Automatic model upload</li>
            <li><strong>ZamAI Optimization</strong>: Pashto-specific enhancements</li>
        </ul>
    </div>
    """)

if __name__ == "__main__":
    demo.launch()
'''
        
        # Upload the master app
        api.upload_file(
            path_or_fileobj=master_app.encode(),
            path_in_repo="app.py",
            repo_id=space_id,
            repo_type="space",
            token=token
        )
        
        # Advanced requirements
        requirements = '''gradio>=4.0.0
transformers>=4.35.0
torch>=2.0.0
datasets>=2.14.0
huggingface_hub>=0.16.0
accelerate>=0.20.0
peft>=0.4.0
spaces>=0.19.0
bitsandbytes>=0.41.0
scipy>=1.9.0
'''
        
        api.upload_file(
            path_or_fileobj=requirements.encode(),
            path_in_repo="requirements.txt",
            repo_id=space_id,
            repo_type="space",
            token=token
        )
        
        print(f"✅ Master Training Hub created: https://huggingface.co/spaces/{space_id}")
        
    except Exception as e:
        print(f"❌ Failed to create master hub: {e}")

if __name__ == "__main__":
    create_master_training_hub()
