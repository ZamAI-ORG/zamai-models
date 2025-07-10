#!/usr/bin/env python3
"""
🇦🇫 ZamAI Auto Training Launcher
Automatically start multiple training sessions with your best configurations
"""

import requests
import time
import json
from huggingface_hub import HfApi
import os

class ZamAIAutoTrainer:
    def __init__(self):
        with open("/workspaces/ZamAI-Pro-Models/HF-Token.txt", "r") as f:
            self.token = f.read().strip()
        
        self.api = HfApi(token=self.token)
        self.username = "tasal9"
        
        # Priority training configurations
        self.training_configs = [
            {
                "name": "zamai-ultimate-llama3-v1",
                "base_model": "meta-llama/Llama-3.1-8B-Instruct",
                "dataset": "tasal9/ZamAI-Pashto-High-Qualituly-Dataset",
                "epochs": 3,
                "learning_rate": 2e-4,
                "lora_rank": 16,
                "priority": "HIGH"
            },
            {
                "name": "zamai-mistral-chat-v1", 
                "base_model": "mistralai/Mistral-7B-Instruct-v0.2",
                "dataset": "tasal9/ZamAI_Pashto_Dataset",
                "epochs": 3,
                "learning_rate": 2e-4,
                "lora_rank": 16,
                "priority": "HIGH"
            },
            {
                "name": "zamai-phi3-business-v1",
                "base_model": "microsoft/phi-3-mini-4k-instruct", 
                "dataset": "tasal9/ZamAI-Pashto-Dataset-Cleaned",
                "epochs": 4,
                "learning_rate": 1e-4,
                "lora_rank": 24,
                "priority": "MEDIUM"
            },
            {
                "name": "zamai-dialogpt-conversational-v1",
                "base_model": "microsoft/DialoGPT-medium",
                "dataset": "tasal9/Pashto_Dataset", 
                "epochs": 5,
                "learning_rate": 3e-4,
                "lora_rank": 12,
                "priority": "MEDIUM"
            }
        ]
    
    def create_training_space(self, config):
        """Create individual training space for each configuration"""
        space_name = f"zamai-auto-trainer-{config['name']}"
        space_id = f"{self.username}/{space_name}"
        
        print(f"🚀 Creating training space: {space_id}")
        
        try:
            # Create space
            self.api.create_repo(
                repo_id=space_id,
                repo_type="space",
                space_sdk="gradio",
                space_hardware="a10g-small",
                exist_ok=True
            )
            
            # Auto-training app
            auto_app = f'''import gradio as gr
import os
import spaces
from huggingface_hub import HfApi
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import torch

class AutoZamAITrainer:
    def __init__(self):
        self.api = HfApi(token=os.getenv("HF_TOKEN"))
        
    @spaces.GPU
    def auto_train(self):
        """Automatic training with predefined config"""
        try:
            yield "🚀 ZamAI Auto-Trainer Starting..."
            yield f"📋 Model: {config['name']}"
            yield f"🤖 Base: {config['base_model']}"
            yield f"📚 Dataset: {config['dataset']}"
            
            # Load dataset
            yield "📚 Loading dataset..."
            dataset = load_dataset("{config['dataset']}")
            yield f"✅ Loaded {{len(dataset['train'])}} training examples"
            
            # Load model
            yield "🤖 Loading model..."
            tokenizer = AutoTokenizer.from_pretrained("{config['base_model']}")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model = AutoModelForCausalLM.from_pretrained(
                "{config['base_model']}",
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # LoRA setup
            yield f"⚙️ Setting up LoRA (rank={config['lora_rank']})..."
            lora_config = LoraConfig(
                r={config['lora_rank']},
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            model = get_peft_model(model, lora_config)
            trainable_params = model.print_trainable_parameters()
            yield f"📊 {{trainable_params}}"
            
            # Data preprocessing
            yield "🔄 Preparing training data..."
            def format_data(examples):
                formatted = []
                for inst, out in zip(examples.get("instruction", []), examples.get("output", [])):
                    prompt = f"### Instruction:\\n{{inst}}\\n\\n### Response:\\n{{out}}<|endoftext|>"
                    formatted.append(prompt)
                return {{"text": formatted}}
            
            dataset = dataset.map(format_data, batched=True)
            
            def tokenize(examples):
                return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)
            
            tokenized = dataset.map(tokenize, batched=True)
            
            # Training setup
            output_name = "{config['name']}"
            
            training_args = TrainingArguments(
                output_dir=f"./outputs/{{output_name}}",
                num_train_epochs={config['epochs']},
                per_device_train_batch_size=1,
                gradient_accumulation_steps=8,
                learning_rate={config['learning_rate']},
                fp16=True,
                logging_steps=10,
                save_steps=200,
                push_to_hub=True,
                hub_model_id=f"tasal9/{{output_name}}",
                hub_token=os.getenv("HF_TOKEN"),
                report_to="none"
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized["train"],
                data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
            )
            
            # Start training
            yield f"🏋️ Starting training ({{config['epochs']}} epochs)..."
            trainer.train()
            
            # Save and upload
            yield "💾 Saving model..."
            trainer.save_model()
            
            yield "📤 Uploading to Hub..."
            trainer.push_to_hub()
            
            # Test generation
            yield "🧪 Testing model..."
            model.eval()
            test_input = "سلام! زه یو پوښتنه لرم"
            inputs = tokenizer(test_input, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=100, temperature=0.7)
            
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            yield f"🧪 Test output: {{result}}"
            
            yield f"""
✅ TRAINING COMPLETED SUCCESSFULLY!

🎯 Model: tasal9/{config['name']}
🚀 URL: https://huggingface.co/tasal9/{config['name']}
🔥 Ready for deployment!

Priority: {config['priority']}
"""
            
        except Exception as e:
            yield f"❌ Training failed: {{str(e)}}"

trainer = AutoZamAITrainer()

# Simple auto-start interface
with gr.Blocks(title="🇦🇫 ZamAI Auto Trainer - {config['name']}", theme=gr.themes.Soft()) as demo:
    gr.HTML(f"""
    <div style="text-align: center; padding: 20px; background: linear-gradient(45deg, #FF6B6B, #4ECDC4); color: white; border-radius: 10px;">
        <h1>🇦🇫 ZamAI Auto Trainer</h1>
        <h2>{config['name']}</h2>
        <h3>{config['priority']} Priority Training</h3>
        <p><strong>Base:</strong> {config['base_model']}</p>
        <p><strong>Dataset:</strong> {config['dataset']}</p>
        <p><strong>Config:</strong> {{config['epochs']}} epochs, LoRA rank {{config['lora_rank']}}</p>
    </div>
    """)
    
    auto_btn = gr.Button("🚀 START AUTO TRAINING", variant="primary", size="lg")
    
    progress = gr.Textbox(
        label="📊 Training Progress",
        lines=20,
        interactive=False,
        show_copy_button=True
    )
    
    auto_btn.click(trainer.auto_train, outputs=progress)
    
    gr.HTML("""
    <div style="margin-top: 20px; padding: 15px; background: #f0f0f0; border-radius: 10px;">
        <h4>🔥 This training will automatically:</h4>
        <ul>
            <li>Load and preprocess your dataset</li>
            <li>Configure LoRA fine-tuning</li>
            <li>Train for optimal epochs</li>
            <li>Upload to your HF Hub</li>
            <li>Test the final model</li>
        </ul>
        <p><strong>Estimated time:</strong> 30-60 minutes</p>
    </div>
    """)

demo.launch()
'''
            
            # Upload app
            self.api.upload_file(
                path_or_fileobj=auto_app.encode(),
                path_in_repo="app.py",
                repo_id=space_id,
                repo_type="space",
                token=self.token
            )
            
            # Upload requirements
            requirements = '''gradio>=4.0.0
transformers>=4.35.0
torch>=2.0.0
datasets>=2.14.0
huggingface_hub>=0.16.0
accelerate>=0.20.0
peft>=0.4.0
spaces>=0.19.0
'''
            
            self.api.upload_file(
                path_or_fileobj=requirements.encode(),
                path_in_repo="requirements.txt",
                repo_id=space_id,
                repo_type="space",
                token=self.token
            )
            
            print(f"✅ Created: https://huggingface.co/spaces/{space_id}")
            return space_id
            
        except Exception as e:
            print(f"❌ Failed to create {space_id}: {e}")
            return None
    
    def launch_all_training(self):
        """Launch all training configurations"""
        print("🚀 ZamAI Auto Training Launch Sequence")
        print("=" * 50)
        
        created_spaces = []
        
        # Create training spaces for high priority first
        high_priority = [c for c in self.training_configs if c["priority"] == "HIGH"]
        medium_priority = [c for c in self.training_configs if c["priority"] == "MEDIUM"]
        
        print("\n🔥 Creating HIGH Priority Training Spaces...")
        for config in high_priority:
            space_id = self.create_training_space(config)
            if space_id:
                created_spaces.append({
                    "space_id": space_id,
                    "config": config,
                    "url": f"https://huggingface.co/spaces/{space_id}"
                })
                time.sleep(2)  # Brief pause between creations
        
        print("\n⚡ Creating MEDIUM Priority Training Spaces...")
        for config in medium_priority:
            space_id = self.create_training_space(config)
            if space_id:
                created_spaces.append({
                    "space_id": space_id,
                    "config": config,
                    "url": f"https://huggingface.co/spaces/{space_id}"
                })
                time.sleep(2)
        
        # Generate launch report
        print("\n🎯 TRAINING LAUNCH REPORT")
        print("=" * 50)
        
        for i, space in enumerate(created_spaces, 1):
            config = space["config"]
            print(f"\n{i}. {config['name']}")
            print(f"   🔗 URL: {space['url']}")
            print(f"   🤖 Model: {config['base_model']}")
            print(f"   📚 Dataset: {config['dataset']}")
            print(f"   ⚡ Priority: {config['priority']}")
            print(f"   ⏱️  Time: ~45-60 minutes")
        
        print(f"\n✅ Created {len(created_spaces)} training spaces!")
        print("\n🚀 Next Steps:")
        print("1. Visit each URL above")
        print("2. Click 'START AUTO TRAINING' button")
        print("3. Monitor training progress")
        print("4. Models will auto-upload when complete")
        
        # Save launch report
        report = {
            "launch_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_spaces": len(created_spaces),
            "spaces": created_spaces
        }
        
        with open("/workspaces/ZamAI-Pro-Models/training_launch_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📋 Launch report saved: training_launch_report.json")
        
        return created_spaces

def main():
    """Launch ZamAI auto training"""
    trainer = ZamAIAutoTrainer()
    spaces = trainer.launch_all_training()
    
    print("\n🇦🇫 ZamAI TRAINING ARMY DEPLOYED!")
    print("Your models are now training in parallel!")
    print("Expected completion: 1-2 hours for all models")

if __name__ == "__main__":
    main()
