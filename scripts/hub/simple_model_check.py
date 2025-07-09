"""
Simple HF Hub Model Checker
Check your HF Hub models and prepare them for training
"""

import json
import os
from typing import Dict, List, Optional

def check_hf_token():
    """Check if HF token is available"""
    token_path = '/workspaces/ZamAI-Pro-Models/HF-Token.txt'
    
    if os.path.exists(token_path):
        with open(token_path, 'r') as f:
            token = f.read().strip()
        
        if len(token) > 10:  # Basic validation
            print(f"✅ HF Token found (length: {len(token)})")
            return token
        else:
            print("⚠️  HF Token seems too short")
            return None
    else:
        print("❌ HF Token not found")
        print("Please create HF-Token.txt with your Hugging Face token")
        print("Get your token from: https://huggingface.co/settings/tokens")
        return None

def check_expected_models():
    """Check what models are expected based on your configs"""
    expected_models = []
    
    # Check pashto chat config
    pashto_config = '/workspaces/ZamAI-Pro-Models/configs/pashto_chat_config.json'
    if os.path.exists(pashto_config):
        with open(pashto_config, 'r') as f:
            config = json.load(f)
        
        expected_models.append({
            'id': config.get('hub_model_id', ''),
            'base_model': config.get('base_model', ''),
            'type': 'pashto_chat',
            'config_file': pashto_config
        })
    
    # Check model configs in models/ directory
    models_dir = '/workspaces/ZamAI-Pro-Models/models'
    if os.path.exists(models_dir):
        for root, dirs, files in os.walk(models_dir):
            for file in files:
                if file.endswith('.json'):
                    try:
                        with open(os.path.join(root, file), 'r') as f:
                            config = json.load(f)
                        
                        model_id = config.get('model_id', config.get('hub_model_id', ''))
                        if model_id:
                            expected_models.append({
                                'id': model_id,
                                'base_model': config.get('base_model', ''),
                                'type': f"{os.path.basename(root)}_{file.replace('.json', '')}",
                                'config_file': os.path.join(root, file)
                            })
                    except:
                        continue
    
    return expected_models

def analyze_model_training_readiness(expected_models: List[Dict]):
    """Analyze which models need training"""
    print("\\n📋 EXPECTED MODELS ANALYSIS")
    print("=" * 50)
    
    for model in expected_models:
        model_id = model['id']
        model_type = model['type']
        base_model = model['base_model']
        
        print(f"\\n🎯 {model_id}")
        print(f"   Type: {model_type}")
        print(f"   Base: {base_model}")
        print(f"   Config: {model['config_file']}")
        
        # Check if this looks like a fine-tuned model
        if 'zamai' in model_id.lower() or 'pashto' in model_id.lower():
            print("   Status: 🔄 Needs fine-tuning (ZamAI custom model)")
            print("   Action: Train from base model")
        else:
            print("   Status: ✅ Base model (should be available)")
            print("   Action: No training needed")

def create_simple_training_guide():
    """Create a simple guide for training missing models"""
    guide = '''
# ZamAI Model Training Guide

## Models That Need Training

Based on your configurations, these models need to be fine-tuned:

### 1. Pashto Chat Model
- **Model ID**: tasal9/zamai-pashto-chat-8b
- **Base Model**: meta-llama/Llama-3.1-8B-Instruct
- **Purpose**: Pashto language conversational AI
- **Training Script**: fine-tuning/train_pashto_chat.py

### Training Steps:

1. **Prepare Environment**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set HF Token**:
   ```bash
   # Add your token to HF-Token.txt
   echo "your_token_here" > HF-Token.txt
   ```

3. **Check Dataset Access**:
   ```bash
   python -c "from datasets import load_dataset; print(load_dataset('tasal9/ZamAI_Pashto_Dataset'))"
   ```

4. **Start Training**:
   ```bash
   python fine-tuning/train_pashto_chat.py
   ```

5. **Monitor Progress**:
   - Check wandb dashboard
   - Monitor GPU usage
   - Watch loss curves

### Training Configuration:
- **LoRA Rank**: 64
- **Learning Rate**: 2e-5
- **Batch Size**: 2 (with gradient accumulation)
- **Epochs**: 3
- **Expected Time**: 2-4 hours on A100

### After Training:
1. Model will be automatically pushed to HF Hub
2. Test the model using inference scripts
3. Deploy to production endpoints

## Troubleshooting:

### Common Issues:
1. **CUDA out of memory**: Reduce batch_size in config
2. **Dataset not found**: Check HF token permissions
3. **Push to hub fails**: Verify token has write access

### Support:
- Check logs in outputs/ directory
- Review wandb runs for debugging
- Test locally before deploying
'''
    
    guide_path = '/workspaces/ZamAI-Pro-Models/TRAINING_GUIDE.md'
    with open(guide_path, 'w') as f:
        f.write(guide)
    
    print(f"\\n📖 Training guide created: {guide_path}")

def main():
    """Main function"""
    print("🔍 ZamAI HF Hub Model Checker")
    print("=" * 50)
    
    # Check HF token
    token = check_hf_token()
    
    # Check expected models
    expected_models = check_expected_models()
    
    if not expected_models:
        print("\\n⚠️  No model configurations found")
        print("Make sure you have configs in configs/ and models/ directories")
        return
    
    print(f"\\n✅ Found {len(expected_models)} expected models")
    
    # Analyze training readiness
    analyze_model_training_readiness(expected_models)
    
    # Create training guide
    create_simple_training_guide()
    
    print("\\n🎯 NEXT STEPS:")
    print("1. Add your HF token to HF-Token.txt if not done")
    print("2. Review the TRAINING_GUIDE.md")
    print("3. Run the training scripts for models that need fine-tuning")
    print("4. Test models after training")
    
    if token:
        print("\\n🔄 To check your actual HF Hub models, run:")
        print("python scripts/hub/check_hf_hub_models.py")

if __name__ == "__main__":
    main()
