"""
Prepare ZamAI Models for Training
Based on HF Hub analysis, this script will:
1. Identify models that need training
2. Generate training configurations
3. Set up training environments
4. Create missing model templates
"""

import json
import os
from typing import Dict, List
from pathlib import Path
import shutil
from datetime import datetime

class ZamAITrainingPreparator:
    """Prepare models for training based on HF Hub analysis"""
    
    def __init__(self):
        self.workspace_root = Path('/workspaces/ZamAI-Pro-Models')
        self.configs_dir = self.workspace_root / 'configs'
        self.models_dir = self.workspace_root / 'models'
        self.training_dir = self.workspace_root / 'fine-tuning'
        self.data_dir = self.workspace_root / 'data'
    
    def load_hub_report(self) -> Dict:
        """Load the HF Hub analysis report"""
        report_path = self.data_dir / 'processed' / 'hf_hub_model_report.json'
        
        if not report_path.exists():
            print("❌ No HF Hub report found. Run check_hf_hub_models.py first")
            return {}
        
        with open(report_path, 'r') as f:
            return json.load(f)
    
    def identify_missing_models(self, report: Dict) -> List[Dict]:
        """Identify models that need to be created/trained"""
        # Check expected models from configs
        expected_models = []
        
        # Load pashto chat config
        pashto_config_path = self.configs_dir / 'pashto_chat_config.json'
        if pashto_config_path.exists():
            with open(pashto_config_path, 'r') as f:
                config = json.load(f)
                expected_models.append({
                    'hub_model_id': config.get('hub_model_id', ''),
                    'base_model': config.get('base_model', ''),
                    'config_file': str(pashto_config_path),
                    'type': 'pashto_chat'
                })
        
        # Check other model configs
        for model_category in self.models_dir.iterdir():
            if model_category.is_dir():
                for model_file in model_category.glob('*.json'):
                    with open(model_file, 'r') as f:
                        try:
                            model_config = json.load(f)
                            if 'model_id' in model_config or 'hub_model_id' in model_config:
                                expected_models.append({
                                    'hub_model_id': model_config.get('hub_model_id', model_config.get('model_id', '')),
                                    'base_model': model_config.get('base_model', ''),
                                    'config_file': str(model_file),
                                    'type': f"{model_category.name}_{model_file.stem}"
                                })
                        except json.JSONDecodeError:
                            continue
        
        # Find which models are missing
        existing_models = [model['id'] for model in report.get('models', [])]
        missing_models = []
        
        for expected in expected_models:
            if expected['hub_model_id'] and expected['hub_model_id'] not in existing_models:
                missing_models.append(expected)
        
        return missing_models
    
    def create_training_config(self, model_info: Dict) -> Dict:
        """Create training configuration for a missing model"""
        base_model = model_info.get('base_model', '')
        model_type = model_info.get('type', 'unknown')
        hub_model_id = model_info.get('hub_model_id', '')
        
        # Base configuration template
        config = {
            'base_model': base_model,
            'model_version': 'v1.0',
            'output_dir': f"./outputs/{hub_model_id.split('/')[-1] if '/' in hub_model_id else hub_model_id}",
            'hub_model_id': hub_model_id,
            'push_to_hub': True,
            'private_repo': False,
            'use_wandb': True,
            'max_length': 2048
        }
        
        # Model-specific configurations
        if 'llama' in base_model.lower():
            config.update({
                'lora': {
                    'rank': 64,
                    'alpha': 128,
                    'dropout': 0.05,
                    'target_modules': [
                        "q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"
                    ]
                },
                'training': {
                    'epochs': 3,
                    'batch_size': 2,
                    'gradient_accumulation': 8,
                    'learning_rate': 2e-5,
                    'warmup_steps': 500,
                    'logging_steps': 10,
                    'save_steps': 500,
                    'eval_steps': 500,
                    'max_grad_norm': 1.0
                }
            })
        elif 'mistral' in base_model.lower():
            config.update({
                'lora': {
                    'rank': 32,
                    'alpha': 64,
                    'dropout': 0.1,
                    'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj"]
                },
                'training': {
                    'epochs': 2,
                    'batch_size': 4,
                    'gradient_accumulation': 4,
                    'learning_rate': 1e-5,
                    'warmup_steps': 200,
                    'logging_steps': 10,
                    'save_steps': 250,
                    'eval_steps': 250
                }
            })
        else:
            # Default configuration
            config.update({
                'lora': {
                    'rank': 16,
                    'alpha': 32,
                    'dropout': 0.1,
                    'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj"]
                },
                'training': {
                    'epochs': 2,
                    'batch_size': 4,
                    'gradient_accumulation': 4,
                    'learning_rate': 5e-6,
                    'warmup_steps': 100,
                    'logging_steps': 10,
                    'save_steps': 200,
                    'eval_steps': 200
                }
            })
        
        # Add dataset configuration if it's a Pashto model
        if 'pashto' in model_type.lower() or 'pashto' in hub_model_id.lower():
            config.update({
                'dataset_name': 'tasal9/ZamAI_Pashto_Dataset',
                'dataset_format': 'instruction',
                'train_file': 'pashto_train_instruction.jsonl',
                'validation_file': 'pashto_val_instruction.jsonl',
                'pashto_specialization': {
                    'cultural_context': True,
                    'islamic_references': True,
                    'afghan_geography': True,
                    'pashto_literature': True,
                    'code_switching': True,
                    'formal_informal_register': True
                },
                'system_prompts': {
                    'general': 'تاسو د پښتو ژبې یو ګټور مرستیال یاست. د افغان کلتور په درناوي سره ځواب ورکړئ.',
                    'educational': 'تاسو د پښتو ژبې ښوونکی یاست. د زده کونکو سره صبر وکړئ او ښه تشریح ورکړئ.',
                    'cultural': 'د افغانستان د کلتور، تاریخ او دودونو په اړه مالومات ورکړئ. د اسلامي ارزښتونو درناوی وکړئ.'
                }
            })
        
        return config
    
    def create_training_script(self, model_info: Dict, config: Dict) -> str:
        """Create training script for a model"""
        model_type = model_info.get('type', 'unknown')
        script_name = f"train_{model_type.lower().replace('_', '_')}.py"
        script_path = self.training_dir / script_name
        
        script_content = f'''"""
Training script for {model_info.get('hub_model_id', 'ZamAI Model')}
Generated automatically from model configuration
"""

import json
import os
import sys
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset, Dataset
import wandb
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def load_config():
    """Load training configuration"""
    config_path = "{model_info.get('config_file', '')}"
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # Fallback to embedded config
        return {json.dumps(config, indent=8)}

def setup_model_and_tokenizer(config):
    """Setup model and tokenizer"""
    base_model = config['base_model']
    
    print(f"Loading model: {{base_model}}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Setup LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config['lora']['rank'],
        lora_alpha=config['lora']['alpha'],
        lora_dropout=config['lora']['dropout'],
        target_modules=config['lora'].get('target_modules', ["q_proj", "v_proj"])
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def load_dataset_for_training(config):
    """Load and prepare dataset"""
    dataset_name = config.get('dataset_name')
    
    if dataset_name:
        print(f"Loading dataset: {{dataset_name}}")
        try:
            dataset = load_dataset(dataset_name)
            return dataset['train'], dataset.get('validation')
        except Exception as e:
            print(f"Error loading dataset: {{e}}")
            return None, None
    else:
        print("No dataset specified in config")
        return None, None

def format_data(examples, tokenizer, max_length):
    """Format data for training"""
    # This is a placeholder - implement based on your data format
    inputs = []
    
    for example in examples:
        if 'instruction' in example and 'output' in example:
            # Instruction format
            text = f"### Instruction\\n{{example['instruction']}}\\n\\n### Response\\n{{example['output']}}"
        elif 'text' in example:
            text = example['text']
        else:
            continue
        
        inputs.append(text)
    
    # Tokenize
    tokenized = tokenizer(
        inputs,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    return tokenized

def main():
    """Main training function"""
    print("🚀 Starting {model_info.get('hub_model_id', 'model')} training")
    
    # Load configuration
    config = load_config()
    
    # Setup wandb if enabled
    if config.get('use_wandb', False):
        wandb.init(
            project="zamai-models",
            name=f"{{config.get('hub_model_id', 'model').split('/')[-1]}}-{{datetime.now().strftime('%Y%m%d-%H%M%S')}}",
            config=config
        )
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Load dataset
    train_dataset, eval_dataset = load_dataset_for_training(config)
    
    if train_dataset is None:
        print("❌ No training dataset available")
        return
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        num_train_epochs=config['training']['epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation'],
        learning_rate=config['training']['learning_rate'],
        warmup_steps=config['training']['warmup_steps'],
        logging_steps=config['training']['logging_steps'],
        save_steps=config['training']['save_steps'],
        eval_steps=config['training']['eval_steps'] if eval_dataset else None,
        evaluation_strategy="steps" if eval_dataset else "no",
        save_strategy="steps",
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False,
        max_grad_norm=config['training'].get('max_grad_norm', 1.0),
        dataloader_drop_last=True,
        push_to_hub=config.get('push_to_hub', False),
        hub_model_id=config.get('hub_model_id') if config.get('push_to_hub') else None,
        hub_private_repo=config.get('private_repo', False),
        report_to="wandb" if config.get('use_wandb') else None,
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
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Start training
    print("🏃 Starting training...")
    trainer.train()
    
    # Save final model
    print("💾 Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    # Push to hub if configured
    if config.get('push_to_hub', False):
        print("📤 Pushing to Hub...")
        trainer.push_to_hub()
    
    print("✅ Training completed!")

if __name__ == "__main__":
    main()
'''
        
        # Create training directory if it doesn't exist
        os.makedirs(self.training_dir, exist_ok=True)
        
        # Write script
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        return str(script_path)
    
    def setup_model_directories(self, missing_models: List[Dict]):
        """Setup directory structure for missing models"""
        for model_info in missing_models:
            model_type = model_info.get('type', 'unknown')
            hub_model_id = model_info.get('hub_model_id', '')
            
            if not hub_model_id:
                continue
            
            # Create output directory
            output_dir = self.workspace_root / 'outputs' / hub_model_id.split('/')[-1]
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"📁 Created output directory: {output_dir}")
    
    def generate_training_plan(self, report: Dict) -> Dict:
        """Generate a comprehensive training plan"""
        missing_models = self.identify_missing_models(report)
        
        plan = {
            'generated_at': datetime.now().isoformat(),
            'missing_models_count': len(missing_models),
            'existing_models_count': len(report.get('models', [])),
            'training_plan': []
        }
        
        for i, model_info in enumerate(missing_models, 1):
            print(f"📋 Planning training for {model_info.get('hub_model_id', 'unknown')} ({i}/{len(missing_models)})")
            
            # Create training config
            config = self.create_training_config(model_info)
            
            # Create training script
            script_path = self.create_training_script(model_info, config)
            
            # Setup directories
            self.setup_model_directories([model_info])
            
            plan_item = {
                'model_info': model_info,
                'training_config': config,
                'training_script': script_path,
                'priority': 'high' if 'pashto' in model_info.get('type', '').lower() else 'medium',
                'estimated_training_time': '2-4 hours',
                'requirements': [
                    'GPU with 16GB+ VRAM',
                    'Hugging Face token with write access',
                    'Training dataset access'
                ]
            }
            
            plan['training_plan'].append(plan_item)
        
        # Save training plan
        plan_path = self.data_dir / 'processed' / 'training_plan.json'
        with open(plan_path, 'w') as f:
            json.dump(plan, f, indent=2)
        
        print(f"📋 Training plan saved to: {plan_path}")
        return plan

def main():
    """Main function to prepare models for training"""
    print("🔧 ZamAI Training Preparator")
    print("=" * 50)
    
    preparator = ZamAITrainingPreparator()
    
    # Load HF Hub report
    report = preparator.load_hub_report()
    if not report:
        return
    
    # Generate training plan
    plan = preparator.generate_training_plan(report)
    
    print("\n📊 TRAINING PLAN SUMMARY")
    print("-" * 30)
    print(f"Missing models: {plan['missing_models_count']}")
    print(f"Existing models: {plan['existing_models_count']}")
    
    if plan['training_plan']:
        print("\n🎯 MODELS TO TRAIN")
        print("-" * 30)
        for item in plan['training_plan']:
            model_id = item['model_info'].get('hub_model_id', 'unknown')
            priority = item['priority']
            print(f"{priority.upper()}: {model_id}")
            print(f"  Script: {item['training_script']}")
            print(f"  Time: {item['estimated_training_time']}")
            print()
        
        print("🚀 Next steps:")
        print("1. Review the generated training scripts")
        print("2. Ensure datasets are available") 
        print("3. Run training scripts in order of priority")
        print("4. Monitor training progress with wandb")
    else:
        print("✅ All expected models are already available!")

if __name__ == "__main__":
    main()
