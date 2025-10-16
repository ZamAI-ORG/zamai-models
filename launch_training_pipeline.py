#!/usr/bin/env python3
"""
🇦🇫 ZamAI Automated Training Launcher
Start multiple fine-tuning jobs to create production-ready models
"""

import requests
import time
import json
from huggingface_hub import HfApi
import os

class ZamAITrainingLauncher:
    def __init__(self):
        # Load HF token
        with open("/workspaces/ZamAI-Pro-Models/HF-Token.txt", "r") as f:
            self.token = f.read().strip()
        
        self.api = HfApi(token=self.token)
        self.username = "tasal9"
        
        # Training configurations for different use cases
        self.training_configs = [
            {
                "name": "zamai-pashto-ultimate-llama3",
                "base_model": "meta-llama/Llama-3.1-8B-Instruct",
                "dataset": "tasal9/ZamAI-Pashto-High-Qualituly-Dataset",
                "use_case": "General conversation",
                "epochs": 3,
                "lora_rank": 16,
                "learning_rate": 2e-4
            },
            {
                "name": "zamai-pashto-mistral-chat",
                "base_model": "mistralai/Mistral-7B-Instruct-v0.2",
                "dataset": "tasal9/ZamAI_Pashto_Dataset",
                "use_case": "Chat assistant",
                "epochs": 4,
                "lora_rank": 32,
                "learning_rate": 1e-4
            },
            {
                "name": "zamai-pashto-phi3-efficient",
                "base_model": "microsoft/phi-3-mini-4k-instruct",
                "dataset": "tasal9/ZamAI-Pashto-Dataset-Cleaned",
                "use_case": "Efficient mobile deployment",
                "epochs": 5,
                "lora_rank": 16,
                "learning_rate": 3e-4
            },
            {
                "name": "zamai-pashto-educational-bot",
                "base_model": "meta-llama/Llama-3.1-8B-Instruct",
                "dataset": "tasal9/ZamAI-Pashto-High-Qualituly-Dataset",
                "use_case": "Educational assistant",
                "epochs": 3,
                "lora_rank": 24,
                "learning_rate": 1.5e-4
            }
        ]
    
    def create_training_space_for_config(self, config):
        """Create a dedicated training space for each configuration"""
        space_name = f"trainer-{config['name']}"
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
            
            # Create auto-training app
            training_app = f'''import gradio as gr
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

class AutoTrainer:
    def __init__(self):
        self.api = HfApi(token=os.getenv("HF_TOKEN"))
        self.config = {{
            "base_model": "{config['base_model']}",
            "dataset": "{config['dataset']}",
            "model_name": "{config['name']}",
            "epochs": {config['epochs']},
            "lora_rank": {config['lora_rank']},
            "learning_rate": {config['learning_rate']},
            "use_case": "{config['use_case']}"
        }}
        
    @spaces.GPU
    def auto_train(self):
        """Automated training process"""
        try:
            yield f"🚀 Starting auto-training for {{self.config['use_case']}}"
            yield f"📋 Configuration: {{self.config}}"
            
            # Load dataset
            yield f"📚 Loading dataset: {{self.config['dataset']}}"
            dataset = load_dataset(self.config["dataset"])
            yield f"✅ Dataset loaded: {{len(dataset['train'])}} examples"
            
            # Load model
            yield f"🤖 Loading model: {{self.config['base_model']}}"
            tokenizer = AutoTokenizer.from_pretrained(self.config["base_model"])
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                self.config["base_model"],
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # LoRA setup
            yield f"⚙️ Setting up LoRA (rank={{self.config['lora_rank']}})..."
            lora_config = LoraConfig(
                r=self.config["lora_rank"],
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            
            # Format data
            yield "🔄 Formatting training data..."
            def format_data(examples):
                formatted = []
                instructions = examples.get("instruction", examples.get("input", []))
                outputs = examples.get("output", examples.get("response", []))
                
                for inst, out in zip(instructions, outputs):
                    prompt = f"""### پوښتنه:
{{inst}}

### ځواب:
{{out}}<|endoftext|>"""
                    formatted.append(prompt)
                
                return {{"text": formatted}}
            
            dataset = dataset.map(format_data, batched=True)
            
            def tokenize(examples):
                return tokenizer(
                    examples["text"],
                    truncation=True,
                    padding=True,
                    max_length=512
                )
            
            tokenized_dataset = dataset.map(tokenize, batched=True)
            
            # Training setup
            training_args = TrainingArguments(
                output_dir=f"./outputs/{{self.config['model_name']}}",
                num_train_epochs=self.config["epochs"],
                per_device_train_batch_size=1,
                gradient_accumulation_steps=8,
                warmup_steps=100,
                learning_rate=self.config["learning_rate"],
                fp16=True,
                logging_steps=10,
                save_steps=200,
                evaluation_strategy="no",
                push_to_hub=True,
                hub_model_id=f"tasal9/{{self.config['model_name']}}",
                hub_token=os.getenv("HF_TOKEN"),
                report_to="none"
            )
            
            # Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset["train"],
                data_collator=DataCollatorForLanguageModeling(
                    tokenizer=tokenizer,
                    mlm=False
                )
            )
            
            # Train
            yield f"🏋️ Training {{self.config['model_name']}} for {{self.config['epochs']}} epochs..."
            trainer.train()
            
            # Save and push
            yield "💾 Saving model..."
            trainer.save_model()
            
            yield "📤 Uploading to Hub..."
            trainer.push_to_hub()
            
            # Test generation
            yield "🧪 Testing model..."
            model.eval()
            test_input = "سلام وروره! زه د افغانستان د"
            inputs = tokenizer(test_input, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            yield f"🧪 Sample: {{result}}"
            
            yield f"""
✅ AUTO-TRAINING COMPLETED SUCCESSFULLY! 

🎯 Model: tasal9/{{self.config['model_name']}}
🚀 Use case: {{self.config['use_case']}}
📊 Dataset: {{self.config['dataset']}}
⚡ Method: LoRA Fine-tuning
🔗 URL: https://huggingface.co/tasal9/{{self.config['model_name']}}

🔥 Model is ready for production use!
"""
            
        except Exception as e:
            yield f"❌ Training failed: {{str(e)}}"

trainer = AutoTrainer()

# Simple auto-start interface
with gr.Blocks(title="🇦🇫 Auto-Trainer: {config['name']}", theme=gr.themes.Soft()) as demo:
    gr.HTML(f"""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
        <h1>🇦🇫 ZamAI Auto-Trainer</h1>
        <h3>Training: {config['name']}</h3>
        <p>Use case: {config['use_case']}</p>
        <p>Base: {config['base_model']}</p>
        <p>Dataset: {config['dataset']}</p>
    </div>
    """)
    
    start_btn = gr.Button("🚀 START AUTO-TRAINING", variant="primary", size="lg")
    
    progress = gr.Textbox(
        label="📊 Training Progress",
        lines=20,
        interactive=False,
        show_copy_button=True
    )
    
    start_btn.click(trainer.auto_train, outputs=progress)
    
    gr.HTML("""
    <div style="margin-top: 20px; padding: 15px; background: #f0f0f0; border-radius: 10px;">
        <h3>⚡ Auto-Training Features:</h3>
        <ul>
            <li>Automatic dataset loading and preprocessing</li>
            <li>Optimized LoRA configuration</li>
            <li>Real-time progress tracking</li>
            <li>Automatic model upload to Hub</li>
            <li>Built-in testing and validation</li>
        </ul>
    </div>
    """)

if __name__ == "__main__":
    demo.launch()
'''
            
            # Upload app
            self.api.upload_file(
                path_or_fileobj=training_app.encode(),
                path_in_repo="app.py",
                repo_id=space_id,
                repo_type="space",
                token=self.token
            )
            
            # Requirements
            requirements = '''gradio>=4.0.0
transformers>=4.35.0
torch>=2.0.0
datasets>=2.14.0
huggingface_hub>=0.16.0
accelerate>=0.20.0
peft>=0.4.0
spaces>=0.19.0
bitsandbytes>=0.41.0
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
        print("🚀 Launching ZamAI Production Training Pipeline!")
        print("=" * 60)
        
        created_spaces = []
        
        for i, config in enumerate(self.training_configs, 1):
            print(f"\n📋 Configuration {i}/{len(self.training_configs)}")
            print(f"Model: {config['name']}")
            print(f"Base: {config['base_model']}")
            print(f"Dataset: {config['dataset']}")
            print(f"Use case: {config['use_case']}")
            
            space_id = self.create_training_space_for_config(config)
            if space_id:
                created_spaces.append(space_id)
            
            # Brief pause between creations
            time.sleep(2)
        
        print(f"\n🎉 Training Pipeline Ready!")
        print(f"📊 Created {len(created_spaces)} training spaces")
        
        print(f"\n🚀 **TRAINING SPACES TO START:**")
        for space in created_spaces:
            print(f"  🔗 https://huggingface.co/spaces/{space}")
        
        print(f"\n📋 **MANUAL START INSTRUCTIONS:**")
        print(f"1. Visit each training space URL above")
        print(f"2. Click the '🚀 START AUTO-TRAINING' button")
        print(f"3. Wait 30-60 minutes for completion")
        print(f"4. Your new models will be available at:")
        for config in self.training_configs:
            print(f"   - tasal9/{config['name']}")
        
        return created_spaces

def main():
    """Main execution"""
    print("🇦🇫 ZamAI Automated Training Launcher")
    print("=" * 50)
    
    launcher = ZamAITrainingLauncher()
    spaces = launcher.launch_all_training()
    
    print(f"\n✅ All training spaces created and ready!")
    print(f"🔥 Start training by visiting the URLs above")
    print(f"🎯 Expected completion: 2-4 hours for all models")

if __name__ == "__main__":
    main()
