#!/usr/bin/env python3
"""
Simple Model Test - Quick debugging
"""

import requests
import json
from huggingface_hub import InferenceClient, model_info

# Load HF token
with open('HF-Token.txt', 'r') as f:
    token = f.read().strip()

# Test models one by one
models = [
    'tasal9/pashto-base-bloom',
    'tasal9/ZamAI-Mistral-7B-Pashto',
    'tasal9/Multilingual-ZamAI-Embeddings'
]

print("🧪 Quick Model Tests")
print("=" * 40)

for model_name in models:
    print(f"\n📦 {model_name}")
    
    # Check if model exists and get info
    try:
        info = model_info(model_name, token=token)
        print(f"✅ Exists: {info.pipeline_tag}, Private: {info.private}")
        
        if info.private:
            print("⚠️  Skipping private model")
            continue
            
    except Exception as e:
        print(f"❌ Model info error: {e}")
        continue
    
    # Test 1: Simple API call
    try:
        api_url = f'https://api-inference.huggingface.co/models/{model_name}'
        headers = {'Authorization': f'Bearer {token}'}
        
        # Simple test payload
        if 'embeddings' in model_name.lower():
            payload = {'inputs': 'سلام'}
        else:
            payload = {
                'inputs': 'سلام',
                'parameters': {'max_new_tokens': 20, 'return_full_text': False}
            }
        
        response = requests.post(api_url, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            print(f"✅ API Call: Success")
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict) and 'generated_text' in result[0]:
                    print(f"   Generated: {result[0]['generated_text'][:50]}...")
                elif isinstance(result[0], list):  # Embeddings
                    print(f"   Embedding length: {len(result[0])}")
                else:
                    print(f"   Result: {str(result)[:50]}...")
            else:
                print(f"   Raw result: {str(result)[:100]}...")
        else:
            print(f"❌ API Call: HTTP {response.status_code}")
            print(f"   Error: {response.text[:100]}...")
            
    except Exception as e:
        print(f"❌ API Call: {e}")
    
    print("-" * 30)

print("\n🎉 Quick test complete!")
