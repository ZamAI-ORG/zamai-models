#!/usr/bin/env python3
"""
Simple ZamAI Model Tester
Test your Hugging Face models quickly
"""

import os
import json
import requests
import time

def test_model_simple(model_id, token, test_input="سلام وروره، څنګه یاست؟"):
    """Simple model testing function"""
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": test_input,
        "parameters": {
            "max_length": 200,
            "temperature": 0.7,
            "do_sample": True
        }
    }
    
    try:
        print(f"🧪 Testing: {model_id}")
        print(f"📝 Input: {test_input}")
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                output = result[0].get('generated_text', 'No response')
            else:
                output = str(result)
            
            print(f"✅ Output: {output[:150]}...")
            return {"status": "success", "output": output}
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return {"status": "error", "error": response.text}
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return {"status": "error", "error": str(e)}

def main():
    """Main function"""
    print("🇦🇫 ZamAI Model Quick Tester")
    print("=" * 40)
    
    # Load token
    with open('HF-Token.txt', 'r') as f:
        token = f.read().strip()
    
    # Your models to test
    models = [
        "tasal9/ZamAI-Mistral-7B-Pashto",
        "tasal9/ZamAI-LIama3-Pashto", 
        "tasal9/pashto-base-bloom",
        "tasal9/Multilingual-ZamAI-Embeddings"
    ]
    
    test_inputs = [
        "سلام وروره، څنګه یاست؟",
        "د افغانستان تاریخ ووایه",
        "زه د پښتو ژبې زده کړه غواړم"
    ]
    
    results = {}
    
    for model in models:
        print(f"\n{'='*50}")
        results[model] = []
        
        for test_input in test_inputs:
            result = test_model_simple(model, token, test_input)
            results[model].append(result)
            print("-" * 30)
            time.sleep(2)  # Avoid rate limiting
    
    # Save results
    with open('quick_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: quick_test_results.json")
    
    # Summary
    print(f"\n📊 Test Summary:")
    for model, tests in results.items():
        success_count = sum(1 for test in tests if test.get('status') == 'success')
        print(f"  {model}: {success_count}/{len(tests)} successful")

if __name__ == "__main__":
    main()
