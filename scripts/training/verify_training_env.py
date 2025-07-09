#!/usr/bin/env python3
"""
ZamAI Training Environment Verification
Test if everything is ready for training
"""

import sys
import os
import json
import traceback

def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name()}")
            print(f"   CUDA version: {torch.version.cuda}")
    except Exception as e:
        print(f"❌ PyTorch: {e}")
        return False
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
    except Exception as e:
        print(f"❌ Transformers: {e}")
        return False
    
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        import peft
        print(f"✅ PEFT {peft.__version__}")
    except Exception as e:
        print(f"❌ PEFT: {e}")
        return False
    
    try:
        from datasets import load_dataset
        import datasets
        print(f"✅ Datasets {datasets.__version__}")
    except Exception as e:
        print(f"❌ Datasets: {e}")
        return False
    
    try:
        import wandb
        print(f"✅ WandB {wandb.__version__}")
    except Exception as e:
        print(f"❌ WandB: {e}")
        return False
    
    return True

def test_config():
    """Test configuration loading"""
    print("\n🔍 Testing configuration...")
    
    config_path = "configs/pashto_chat_config.json"
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print("✅ Configuration loaded successfully")
        print(f"   Base model: {config.get('base_model', 'Not specified')}")
        print(f"   Hub model ID: {config.get('hub_model_id', 'Not specified')}")
        print(f"   Dataset: {config.get('dataset_name', 'Not specified')}")
        print(f"   Output dir: {config.get('output_dir', 'Not specified')}")
        
        # Check required fields
        required_fields = ['base_model', 'hub_model_id', 'output_dir']
        for field in required_fields:
            if not config.get(field):
                print(f"⚠️ Missing required field: {field}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False

def test_hf_token():
    """Test HF token"""
    print("\n🔍 Testing HF token...")
    
    token_path = "HF-Token.txt"
    if not os.path.exists(token_path):
        print(f"❌ HF token file not found: {token_path}")
        print("   Create this file with your HuggingFace token")
        return False
    
    try:
        with open(token_path, 'r') as f:
            token = f.read().strip()
        
        if len(token) < 20:
            print("⚠️ Token seems too short")
            return False
        
        print("✅ HF token found")
        print(f"   Token length: {len(token)} characters")
        
        # Test token validity
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
            user_info = api.whoami()
            print(f"   Connected as: {user_info['name']}")
            return True
        except Exception as e:
            print(f"⚠️ Token validation failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Error reading token: {e}")
        return False

def test_dataset_access():
    """Test dataset access"""
    print("\n🔍 Testing dataset access...")
    
    try:
        from datasets import load_dataset
        
        # Try to load the dataset
        dataset_name = "tasal9/ZamAI_Pashto_Dataset"
        print(f"   Attempting to load: {dataset_name}")
        
        dataset = load_dataset(dataset_name)
        print("✅ Dataset loaded successfully")
        print(f"   Train examples: {len(dataset['train'])}")
        if 'validation' in dataset:
            print(f"   Validation examples: {len(dataset['validation'])}")
        
        # Show first example
        example = dataset['train'][0]
        print(f"   First example keys: {list(example.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset access failed: {e}")
        print("   This might be normal if the dataset doesn't exist yet")
        return False

def test_training_script():
    """Test if training script can be imported"""
    print("\n🔍 Testing training script...")
    
    script_path = "fine-tuning/train_pashto_chat.py"
    if not os.path.exists(script_path):
        print(f"❌ Training script not found: {script_path}")
        return False
    
    try:
        # Try to import the training module
        sys.path.insert(0, 'fine-tuning')
        import train_pashto_chat
        print("✅ Training script can be imported")
        
        # Try to create trainer instance
        trainer = train_pashto_chat.PashtoModelTrainer()
        print("✅ Training class can be instantiated")
        
        return True
        
    except Exception as e:
        print(f"❌ Training script error: {e}")
        traceback.print_exc()
        return False

def main():
    """Main verification function"""
    print("🚀 ZamAI Training Environment Verification")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("HF Token", test_hf_token),
        ("Dataset Access", test_dataset_access),
        ("Training Script", test_training_script)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n📊 SUMMARY")
    print("=" * 30)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("\n🎉 ALL TESTS PASSED!")
        print("Your environment is ready for training!")
        print("\nTo start training:")
        print("python fine-tuning/train_pashto_chat.py")
    else:
        print(f"\n⚠️ {len(results) - passed} tests failed")
        print("Please fix the issues before training")
    
    return passed == len(results)

if __name__ == "__main__":
    main()
