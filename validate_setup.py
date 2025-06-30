#!/usr/bin/env python3
"""
ZamAI Setup Validation
Quick validation that everything is ready for training
"""

import os
import json

def validate_setup():
    print("🇦🇫 ZamAI Setup Validation")
    print("=" * 40)
    
    validation_results = {
        "environment": {},
        "files": {},
        "config": {},
        "ready_to_train": False
    }
    
    # Check 1: Files exist
    required_files = [
        "HF-Token.txt",
        "fine-tuning/train_pashto_chat.py",
        "fine-tuning/configs/pashto_chat_config.json",
        "train_zamai_v4.py",
        "analyze_zamai_dataset.py",
        "test_models.py"
    ]
    
    print("📁 Checking required files...")
    files_ok = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
            validation_results["files"][file_path] = True
        else:
            print(f"   ❌ {file_path}")
            validation_results["files"][file_path] = False
            files_ok = False
    
    # Check 2: Config validation
    config_path = "fine-tuning/configs/pashto_chat_config.json"
    print(f"\n⚙️  Checking configuration...")
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            required_config_keys = [
                "base_model", "dataset_name", "dataset_format",
                "train_file", "validation_file", "hub_model_id"
            ]
            
            config_ok = True
            for key in required_config_keys:
                if key in config:
                    print(f"   ✅ {key}: {config[key]}")
                    validation_results["config"][key] = config[key]
                else:
                    print(f"   ❌ Missing: {key}")
                    validation_results["config"][key] = None
                    config_ok = False
            
            validation_results["config"]["valid"] = config_ok
        except Exception as e:
            print(f"   ❌ Config error: {e}")
            validation_results["config"]["valid"] = False
            config_ok = False
    else:
        print(f"   ❌ Config file not found")
        config_ok = False
    
    # Check 3: HF Token
    print(f"\n🔑 Checking Hugging Face token...")
    if os.path.exists("HF-Token.txt"):
        with open("HF-Token.txt", 'r') as f:
            token = f.read().strip()
        if token and len(token) > 10:
            print(f"   ✅ Token found ({token[:10]}...)")
            validation_results["environment"]["hf_token"] = True
        else:
            print(f"   ❌ Invalid token")
            validation_results["environment"]["hf_token"] = False
    else:
        print(f"   ❌ Token file not found")
        validation_results["environment"]["hf_token"] = False
    
    # Overall status
    print(f"\n📊 Validation Summary:")
    all_files_ok = all(validation_results["files"].values())
    config_ok = validation_results["config"].get("valid", False)
    token_ok = validation_results["environment"]["hf_token"]
    
    print(f"   Files: {'✅' if all_files_ok else '❌'}")
    print(f"   Config: {'✅' if config_ok else '❌'}")
    print(f"   HF Token: {'✅' if token_ok else '❌'}")
    
    validation_results["ready_to_train"] = all_files_ok and config_ok and token_ok
    
    if validation_results["ready_to_train"]:
        print(f"\n🎉 Ready to train ZamAI models!")
        print(f"\n🚀 Available commands:")
        print(f"   python train_zamai_v4.py       # Train ZamAI V4 (Mistral)")
        print(f"   python analyze_zamai_dataset.py # Analyze dataset")
        print(f"   python test_models.py          # Test existing models")
    else:
        print(f"\n⚠️  Setup incomplete. Please fix the issues above.")
    
    # Save validation results
    with open("setup_validation.json", 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    print(f"\n💾 Validation results saved to: setup_validation.json")
    
    return validation_results

if __name__ == "__main__":
    validate_setup()
