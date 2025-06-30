#!/usr/bin/env python3
"""
ZamAI Model Debugging and Testing Script
Debug model inference issues and test API connectivity
"""

import os
import json
import requests
import time
from typing import Dict, List, Optional
from huggingface_hub import HfApi, model_info, InferenceClient
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelDebugger:
    def __init__(self, token_file: str = "HF-Token.txt"):
        """Initialize the model debugger"""
        with open(token_file, 'r') as f:
            self.token = f.read().strip()
        
        self.api = HfApi(token=self.token)
        self.inference_client = InferenceClient(token=self.token)
        self.username = "tasal9"
        
        # Test prompts in different languages
        self.test_prompts = {
            'pashto': 'سلام وروره، تاسو څنګه یاست؟',
            'english': 'Hello, how are you?',
            'simple': 'Test'
        }
    
    def check_model_status(self, model_name: str) -> Dict:
        """Check detailed model status and info"""
        try:
            info = model_info(model_name, token=self.token)
            return {
                'exists': True,
                'pipeline_tag': info.pipeline_tag,
                'library': info.library_name,
                'private': info.private,
                'gated': info.gated,
                'tags': info.tags,
                'downloads': getattr(info, 'downloads', 'N/A'),
                'likes': getattr(info, 'likes', 'N/A')
            }
        except Exception as e:
            return {
                'exists': False,
                'error': str(e)
            }
    
    def test_inference_api(self, model_name: str, prompt: str = None) -> Dict:
        """Test inference API with different methods"""
        if prompt is None:
            prompt = self.test_prompts['pashto']
        
        results = {}
        
        # Method 1: Direct API call
        try:
            api_url = f'https://api-inference.huggingface.co/models/{model_name}'
            headers = {
                'Authorization': f'Bearer {self.token}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'inputs': prompt,
                'parameters': {
                    'max_new_tokens': 50,
                    'temperature': 0.7,
                    'return_full_text': False
                }
            }
            
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            
            results['direct_api'] = {
                'status_code': response.status_code,
                'success': response.status_code == 200,
                'response': response.json() if response.status_code == 200 else response.text,
                'error': None if response.status_code == 200 else f"HTTP {response.status_code}"
            }
            
        except Exception as e:
            results['direct_api'] = {
                'success': False,
                'error': str(e)
            }
        
        # Method 2: InferenceClient
        try:
            if 'text-generation' in model_name or 'bloom' in model_name.lower() or 'mistral' in model_name.lower() or 'llama' in model_name.lower():
                result = self.inference_client.text_generation(
                    model=model_name,
                    prompt=prompt,
                    max_new_tokens=50,
                    temperature=0.7
                )
                results['inference_client'] = {
                    'success': True,
                    'result': result,
                    'error': None
                }
            elif 'embeddings' in model_name.lower():
                result = self.inference_client.feature_extraction(
                    model=model_name,
                    inputs=prompt
                )
                results['inference_client'] = {
                    'success': True,
                    'result': f"Embedding vector of length: {len(result) if isinstance(result, list) else 'Unknown'}",
                    'error': None
                }
            else:
                results['inference_client'] = {
                    'success': False,
                    'error': 'Unknown model type'
                }
                
        except Exception as e:
            results['inference_client'] = {
                'success': False,
                'error': str(e)
            }
        
        return results
    
    def test_all_models(self) -> Dict:
        """Test all available models"""
        print("🔍 ZamAI Model Debugging Report")
        print("=" * 60)
        
        # Load model inventory
        with open('model_inventory.json', 'r') as f:
            inventory = json.load(f)
        
        results = {}
        
        for model_data in inventory.get('models', []):
            model_name = model_data['id']
            print(f"\n📦 Testing Model: {model_name}")
            print("-" * 40)
            
            # Check model status
            status = self.check_model_status(model_name)
            print(f"Status: {'✅ Available' if status['exists'] else '❌ Not Found'}")
            
            if status['exists']:
                print(f"Pipeline: {status['pipeline_tag']}")
                print(f"Library: {status['library']}")
                print(f"Private: {status['private']}")
                print(f"Downloads: {status['downloads']}")
                
                # Test inference
                if not status['private']:  # Only test public models
                    print("\n🧪 Testing Inference...")
                    
                    # Test with different prompts
                    for prompt_type, prompt in self.test_prompts.items():
                        print(f"\n  📝 Testing with {prompt_type} prompt:")
                        inference_results = self.test_inference_api(model_name, prompt)
                        
                        for method, result in inference_results.items():
                            if result['success']:
                                print(f"    ✅ {method}: Success")
                                if 'result' in result:
                                    print(f"       Result: {result['result'][:100]}...")
                            else:
                                print(f"    ❌ {method}: {result['error']}")
                        
                        # Add small delay between tests
                        time.sleep(1)
                else:
                    print("⚠️  Skipping inference test (private model)")
            
            results[model_name] = {
                'status': status,
                'inference_tests': self.test_inference_api(model_name) if status['exists'] and not status.get('private') else None
            }
            
            print()
        
        return results
    
    def generate_integration_fixes(self, debug_results: Dict) -> str:
        """Generate fixes for integration issues"""
        fixes = []
        
        fixes.append("# ZamAI Model Integration Fixes\n")
        fixes.append("Based on debugging results, here are recommended fixes:\n")
        
        for model_name, results in debug_results.items():
            fixes.append(f"## {model_name}")
            
            if not results['status']['exists']:
                fixes.append("- ❌ Model not found - check model name and permissions")
                continue
            
            if results['status']['private']:
                fixes.append("- ⚠️  Model is private - make public or update access tokens")
                continue
            
            if results['inference_tests']:
                working_methods = []
                failing_methods = []
                
                for method, result in results['inference_tests'].items():
                    if result['success']:
                        working_methods.append(method)
                    else:
                        failing_methods.append((method, result['error']))
                
                if working_methods:
                    fixes.append(f"- ✅ Working methods: {', '.join(working_methods)}")
                
                if failing_methods:
                    fixes.append("- ❌ Failing methods:")
                    for method, error in failing_methods:
                        fixes.append(f"  - {method}: {error}")
            
            fixes.append("")
        
        return "\n".join(fixes)

def main():
    """Main debugging function"""
    debugger = ModelDebugger()
    
    print("Starting ZamAI Model Debugging...")
    print()
    
    # Run comprehensive tests
    results = debugger.test_all_models()
    
    # Save results
    with open('debug_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Generate fixes
    fixes = debugger.generate_integration_fixes(results)
    with open('MODEL_FIXES.md', 'w') as f:
        f.write(fixes)
    
    print("🎉 Debugging complete!")
    print("📄 Results saved to: debug_results.json")
    print("🔧 Fixes saved to: MODEL_FIXES.md")

if __name__ == "__main__":
    main()
