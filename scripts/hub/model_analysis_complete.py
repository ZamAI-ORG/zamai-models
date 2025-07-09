"""
ZamAI Model Analysis and Training Preparation
Comprehensive analysis of your HF Hub models and training readiness
"""

import json
import os
from pathlib import Path
from datetime import datetime

def analyze_zamai_models():
    """Analyze all ZamAI model configurations"""
    
    models_found = []
    
    # 1. Check pashto chat config
    pashto_config_path = '/workspaces/ZamAI-Pro-Models/configs/pashto_chat_config.json'
    if os.path.exists(pashto_config_path):
        with open(pashto_config_path, 'r') as f:
            config = json.load(f)
        
        models_found.append({
            'model_id': config.get('hub_model_id', ''),
            'base_model': config.get('base_model', ''),
            'type': 'pashto_chat',
            'priority': 'HIGH',
            'source': 'configs/pashto_chat_config.json',
            'description': 'Main Pashto conversational AI',
            'training_config': config
        })
    
    # 2. Check models in models/ directory
    models_dir = Path('/workspaces/ZamAI-Pro-Models/models')
    
    for category_dir in models_dir.iterdir():
        if category_dir.is_dir():
            for model_file in category_dir.glob('*.json'):
                try:
                    with open(model_file, 'r') as f:
                        model_config = json.load(f)
                    
                    # Determine priority
                    priority = 'HIGH' if model_config.get('priority') == 'primary' else 'MEDIUM'
                    
                    models_found.append({
                        'model_id': model_config.get('model_id', ''),
                        'base_model': model_config.get('training_config', {}).get('base_model', ''),
                        'type': f"{category_dir.name}_{model_file.stem}",
                        'priority': priority,
                        'source': str(model_file),
                        'description': model_config.get('description', ''),
                        'training_config': model_config.get('training_config', {})
                    })
                except Exception as e:
                    print(f"⚠️  Error reading {model_file}: {e}")
    
    return models_found

def categorize_models(models):
    """Categorize models by training status"""
    needs_training = []
    base_models = []
    
    for model in models:
        model_id = model['model_id']
        
        # Check if this is a custom ZamAI model that needs training
        if any(keyword in model_id.lower() for keyword in ['zamai', 'tasal9']):
            needs_training.append(model)
        else:
            base_models.append(model)
    
    return needs_training, base_models

def create_training_priority_list(models_to_train):
    """Create prioritized training list"""
    # Sort by priority
    priority_order = {'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
    sorted_models = sorted(models_to_train, key=lambda x: priority_order.get(x['priority'], 3))
    
    return sorted_models

def generate_training_commands(models_to_train):
    """Generate training commands for each model"""
    commands = []
    
    for model in models_to_train:
        model_id = model['model_id']
        model_type = model['type']
        
        # Generate training script name
        script_name = f"train_{model_type.replace('-', '_').lower()}.py"
        
        # Create command
        command = {
            'model_id': model_id,
            'script': f"fine-tuning/{script_name}",
            'command': f"python fine-tuning/{script_name}",
            'estimated_time': '2-4 hours',
            'gpu_required': '16GB+ VRAM',
            'prerequisites': [
                'HF token with write access',
                'Access to training datasets',
                'CUDA-compatible GPU'
            ]
        }
        
        commands.append(command)
    
    return commands

def check_training_readiness():
    """Check if system is ready for training"""
    checks = []
    
    # Check HF token
    hf_token_path = '/workspaces/ZamAI-Pro-Models/HF-Token.txt'
    if os.path.exists(hf_token_path):
        with open(hf_token_path, 'r') as f:
            token = f.read().strip()
        if len(token) > 20:
            checks.append(('HF Token', '✅', 'Valid token found'))
        else:
            checks.append(('HF Token', '⚠️', 'Token seems invalid'))
    else:
        checks.append(('HF Token', '❌', 'Token file not found'))
    
    # Check training directory
    training_dir = '/workspaces/ZamAI-Pro-Models/fine-tuning'
    if os.path.exists(training_dir):
        checks.append(('Training Directory', '✅', 'Directory exists'))
    else:
        checks.append(('Training Directory', '⚠️', 'Directory missing'))
    
    # Check requirements
    requirements_file = '/workspaces/ZamAI-Pro-Models/requirements.txt'
    if os.path.exists(requirements_file):
        checks.append(('Requirements', '✅', 'requirements.txt found'))
    else:
        checks.append(('Requirements', '❌', 'requirements.txt missing'))
    
    return checks

def create_comprehensive_report():
    """Create comprehensive model analysis report"""
    
    print("🔍 ZamAI Model Analysis & Training Preparation")
    print("=" * 60)
    
    # Analyze models
    all_models = analyze_zamai_models()
    models_to_train, base_models = categorize_models(all_models)
    priority_list = create_training_priority_list(models_to_train)
    training_commands = generate_training_commands(models_to_train)
    readiness_checks = check_training_readiness()
    
    # Display summary
    print(f"\n📊 SUMMARY")
    print("-" * 30)
    print(f"Total models found: {len(all_models)}")
    print(f"Models needing training: {len(models_to_train)}")
    print(f"Base models: {len(base_models)}")
    
    # Display models needing training
    if models_to_train:
        print(f"\n🎯 MODELS REQUIRING TRAINING ({len(models_to_train)})")
        print("-" * 50)
        for i, model in enumerate(priority_list, 1):
            print(f"{i}. {model['model_id']}")
            print(f"   Priority: {model['priority']}")
            print(f"   Base: {model['base_model']}")
            print(f"   Type: {model['type']}")
            print(f"   Description: {model['description']}")
            print()
    
    # Display base models
    if base_models:
        print(f"\n✅ BASE MODELS ({len(base_models)})")
        print("-" * 30)
        for model in base_models:
            print(f"- {model['model_id']} ({model['type']})")
    
    # Training readiness
    print(f"\n🔧 TRAINING READINESS")
    print("-" * 30)
    for check_name, status, message in readiness_checks:
        print(f"{status} {check_name}: {message}")
    
    # Training commands
    if training_commands:
        print(f"\n🚀 TRAINING COMMANDS")
        print("-" * 30)
        for i, cmd in enumerate(training_commands, 1):
            print(f"{i}. {cmd['model_id']}")
            print(f"   Command: {cmd['command']}")
            print(f"   Time: {cmd['estimated_time']}")
            print(f"   GPU: {cmd['gpu_required']}")
            print()
    
    # Next steps
    print(f"\n📋 NEXT STEPS")
    print("-" * 30)
    print("1. Ensure HF token is set up (if not done)")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Check dataset access")
    print("4. Start training high-priority models first")
    print("5. Monitor training with wandb")
    print("6. Test models after training")
    print("7. Deploy to HF Inference Endpoints")
    
    # Create detailed report file
    report = {
        'generated_at': datetime.now().isoformat(),
        'summary': {
            'total_models': len(all_models),
            'models_to_train': len(models_to_train),
            'base_models': len(base_models)
        },
        'models_requiring_training': priority_list,
        'base_models': base_models,
        'training_commands': training_commands,
        'readiness_checks': readiness_checks
    }
    
    # Save report
    report_path = '/workspaces/ZamAI-Pro-Models/MODEL_TRAINING_REPORT.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Detailed report saved to: {report_path}")
    
    return report

if __name__ == "__main__":
    create_comprehensive_report()
