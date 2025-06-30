#!/usr/bin/env python3
"""
ZamAI V4 Training Script
Train ZamAI V4 model using your Pashto dataset
"""

import os
import sys
import json
from datetime import datetime

def main():
    print("🇦🇫 ZamAI V4 Training Script")
    print("=" * 50)
    
    # Check if config exists
    config_path = "fine-tuning/configs/pashto_chat_config.json"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        print("Creating default config...")
        
        # Create config directory if it doesn't exist
        os.makedirs("fine-tuning/configs", exist_ok=True)
        
        # Create default config
        default_config = {
            "base_model": "mistralai/Mistral-7B-Instruct-v0.1",
            "model_version": "v4.0",
            "output_dir": "./outputs/zamai-v4-mistral-7b",
            "hub_model_id": "tasal9/ZamAI-V4-Mistral-7B-Pashto",
            "dataset_name": "tasal9/ZamAI_Pashto_Dataset",
            "dataset_format": "instruction",
            "train_file": "pashto_train_instruction.jsonl",
            "validation_file": "pashto_val_instruction.jsonl",
            "max_length": 2048,
            "push_to_hub": True,
            "private_repo": False,
            "use_wandb": True,
            "lora": {
                "rank": 64,
                "alpha": 128,
                "dropout": 0.05,
                "target_modules": [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ]
            },
            "training": {
                "epochs": 3,
                "batch_size": 2,
                "gradient_accumulation": 8,
                "learning_rate": 2e-5,
                "warmup_steps": 500,
                "logging_steps": 10,
                "save_steps": 500,
                "eval_steps": 500,
                "max_grad_norm": 1.0
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Created config file: {config_path}")
    
    # Check if training script exists
    train_script = "fine-tuning/train_pashto_chat.py"
    if not os.path.exists(train_script):
        print(f"❌ Training script not found: {train_script}")
        return
    
    print(f"🚀 Starting ZamAI V4 training...")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Using dataset: tasal9/ZamAI_Pashto_Dataset")
    print(f"🤖 Base model: Mistral-7B-Instruct")
    print(f"📝 Config: {config_path}")
    
    # Change to fine-tuning directory and run training
    os.chdir("fine-tuning")
    
    # Import and run the trainer
    try:
        sys.path.append(".")
        from train_pashto_chat import PashtoModelTrainer
        
        # Initialize trainer
        trainer = PashtoModelTrainer("configs/pashto_chat_config.json")
        
        # Start training
        trainer.train()
        
        # Push to hub
        trainer.push_to_hub()
        
        print("\n🎉 ZamAI V4 training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
