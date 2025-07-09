"""
ZamAI Model Testing with HF Inference
Integrates with your existing model configurations
"""

import json
import os
from pathlib import Path
from hf_inference_client import ZamAIInferenceClient

def load_model_configs():
    """Load your model configurations"""
    configs = {}
    
    # Load Pashto chat config
    pashto_config_path = "/workspaces/ZamAI-Pro-Models/configs/pashto_chat_config.json"
    if os.path.exists(pashto_config_path):
        with open(pashto_config_path, 'r', encoding='utf-8') as f:
            configs['pashto_chat'] = json.load(f)
    
    # Load other model configs
    models_dir = Path("/workspaces/ZamAI-Pro-Models/models")
    if models_dir.exists():
        for category_dir in models_dir.iterdir():
            if category_dir.is_dir():
                for model_file in category_dir.glob("*.json"):
                    with open(model_file, 'r') as f:
                        model_config = json.load(f)
                        configs[f"{category_dir.name}_{model_file.stem}"] = model_config
    
    return configs

def test_model_inference(model_type: str, test_prompts: list):
    """Test model inference with various prompts"""
    client = ZamAIInferenceClient()
    configs = load_model_configs()
    
    print(f"\n=== Testing {model_type} ===")
    
    if model_type in configs:
        config = configs[model_type]
        model_id = config.get('hub_model_id', config.get('model_id', ''))
        
        if not model_id:
            print(f"No model ID found for {model_type}")
            return
        
        print(f"Model ID: {model_id}")
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\nTest {i}: {prompt}")
            try:
                if 'chat' in model_type.lower():
                    # Use chat format for chat models
                    messages = [
                        {"role": "system", "content": config.get('system_prompts', {}).get('general', '')},
                        {"role": "user", "content": prompt}
                    ]
                    response = client.chat_completion(model_id, messages)
                else:
                    # Use text generation for other models
                    response = client.text_generation(model_id, prompt)
                
                print(f"Response: {response}")
            except Exception as e:
                print(f"Error: {e}")
    else:
        print(f"Configuration for {model_type} not found")

def benchmark_models():
    """Benchmark your models against different tasks"""
    
    # Test prompts in different languages and contexts
    test_cases = {
        'pashto_chat': [
            "سلام ورور، زه د پښتو ژبې په اړه پوښتنه لرم",
            "د افغانستان د تاریخ په اړه راته ووایاست",
            "What is the capital of Afghanistan?",  # Code-switching test
            "د اسلام لومړني پنځه رکنونه کوم دي؟"
        ],
        'text_generation': [
            "The importance of preserving Pashto language",
            "Afghanistan's role in regional development",
            "Traditional Afghan culture and values"
        ]
    }
    
    for model_type, prompts in test_cases.items():
        test_model_inference(model_type, prompts)

def create_inference_endpoint_config():
    """Create configuration for HF Inference Endpoints deployment"""
    
    config = {
        "compute": {
            "accelerator": "gpu",
            "instance_size": "medium",
            "instance_type": "nvidia-tesla-t4",
            "scaling": {
                "min_replica": 0,
                "max_replica": 1
            }
        },
        "model": {
            "image": {
                "huggingface": {}
            },
            "task": "text-generation"
        },
        "name": "zamai-pashto-chat-endpoint"
    }
    
    # Save config for deployment
    config_path = "/workspaces/ZamAI-Pro-Models/configs/inference_endpoint_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Inference Endpoint config saved to: {config_path}")
    return config

def estimate_costs():
    """Estimate HF Inference costs for your usage"""
    
    print("\n=== HF Inference Cost Estimation ===")
    print("\n1. Inference API (Free Tier):")
    print("   - Rate limited")
    print("   - Good for testing and development")
    print("   - No cost")
    
    print("\n2. Inference Endpoints:")
    print("   - GPU Medium (Tesla T4): ~$0.60/hour")
    print("   - GPU Large (A10G): ~$1.30/hour")
    print("   - GPU XLarge (A100): ~$4.50/hour")
    print("   - Auto-scaling from 0 replicas")
    
    print("\n3. Serverless Inference:")
    print("   - Pay per request")
    print("   - ~$0.0002 per 1k characters")
    print("   - Cold start latency")
    
    print("\nRecommendation for ZamAI:")
    print("- Development: Use free Inference API")
    print("- Production: Use Inference Endpoints with auto-scaling")
    print("- High volume: Consider dedicated endpoints")

def main():
    """Main testing function"""
    print("ZamAI HF Inference Integration Test")
    print("=" * 50)
    
    # Test basic functionality
    client = ZamAIInferenceClient()
    
    # Load and display available models
    configs = load_model_configs()
    print(f"\nAvailable model configurations: {list(configs.keys())}")
    
    # Run benchmarks
    benchmark_models()
    
    # Create endpoint config
    create_inference_endpoint_config()
    
    # Show cost estimates
    estimate_costs()

if __name__ == "__main__":
    main()
