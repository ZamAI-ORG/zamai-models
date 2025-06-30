#!/usr/bin/env python3
"""
Test ZamAI Models
Test and validate your existing Hugging Face models
"""

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login, HfApi
import time

class ZamAIModelTester:
    def __init__(self):
        """Initialize the model tester"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🖥️  Using device: {self.device}")
        
        # Login to HF
        token_path = "HF-Token.txt"
        if os.path.exists(token_path):
            with open(token_path, 'r') as f:
                token = f.read().strip()
            login(token=token)
            print("🔑 Logged in to Hugging Face")
        
        self.api = HfApi()
        
    def list_user_models(self):
        """List all models under tasal9"""
        print("\n📋 Your Hugging Face Models:")
        print("-" * 40)
        
        try:
            models = self.api.list_models(author="tasal9")
            model_list = []
            
            for model in models:
                model_info = {
                    "name": model.id,
                    "downloads": model.downloads,
                    "private": model.private,
                    "tags": model.tags
                }
                model_list.append(model_info)
                print(f"📦 {model.id}")
                print(f"   📊 Downloads: {model.downloads}")
                print(f"   🏷️  Tags: {model.tags}")
                print(f"   🔒 Private: {model.private}")
                print()
            
            return model_list
            
        except Exception as e:
            print(f"❌ Error listing models: {e}")
            return []
    
    def test_model(self, model_name, test_prompts=None):
        """Test a specific model"""
        print(f"\n🧪 Testing Model: {model_name}")
        print("-" * 50)
        
        if test_prompts is None:
            test_prompts = [
                "سلام وروره! ته څنګه یې؟",  # Hello brother! How are you?
                "د افغانستان پلازمینه کوم ښار دی؟",  # What is the capital of Afghanistan?
                "Write a short poem in Pashto about mountains.",
                "Translate to Pashto: Good morning, how are you today?"
            ]
        
        try:
            print("📥 Loading model...")
            start_time = time.time()
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            load_time = time.time() - start_time
            print(f"✅ Model loaded in {load_time:.2f}s")
            
            # Create pipeline
            generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            
            results = []
            
            # Test each prompt
            for i, prompt in enumerate(test_prompts, 1):
                print(f"\n🔤 Test {i}: {prompt}")
                print("💭 Generating response...")
                
                try:
                    start_time = time.time()
                    
                    # Generate response
                    response = generator(
                        prompt,
                        max_length=200,
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    
                    generation_time = time.time() - start_time
                    generated_text = response[0]['generated_text']
                    
                    # Extract only the new text (remove prompt)
                    if generated_text.startswith(prompt):
                        response_text = generated_text[len(prompt):].strip()
                    else:
                        response_text = generated_text
                    
                    print(f"📝 Response ({generation_time:.2f}s): {response_text}")
                    
                    results.append({
                        "prompt": prompt,
                        "response": response_text,
                        "generation_time": generation_time,
                        "status": "success"
                    })
                    
                except Exception as e:
                    print(f"❌ Error generating response: {e}")
                    results.append({
                        "prompt": prompt,
                        "response": None,
                        "generation_time": 0,
                        "status": "error",
                        "error": str(e)
                    })
            
            # Clean up
            del model
            del tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return {
                "model_name": model_name,
                "load_time": load_time,
                "test_results": results,
                "status": "completed"
            }
            
        except Exception as e:
            print(f"❌ Error testing model: {e}")
            return {
                "model_name": model_name,
                "status": "failed",
                "error": str(e)
            }
    
    def test_all_models(self):
        """Test all available models"""
        print("🇦🇫 ZamAI Model Testing Suite")
        print("=" * 50)
        
        # Get model list
        models = self.list_user_models()
        
        if not models:
            print("❌ No models found to test")
            return
        
        all_results = []
        
        for model_info in models:
            model_name = model_info["name"]
            
            # Skip if model seems to be a dataset
            if any(keyword in model_name.lower() for keyword in ["dataset", "data"]):
                print(f"⏭️  Skipping dataset: {model_name}")
                continue
            
            result = self.test_model(model_name)
            all_results.append(result)
            
            print(f"\n{'='*50}")
        
        # Save results
        with open("model_test_results.json", 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Test results saved to: model_test_results.json")
        
        # Summary
        successful_tests = [r for r in all_results if r.get("status") == "completed"]
        failed_tests = [r for r in all_results if r.get("status") == "failed"]
        
        print(f"\n📊 Test Summary:")
        print(f"   ✅ Successful: {len(successful_tests)}")
        print(f"   ❌ Failed: {len(failed_tests)}")
        
        if failed_tests:
            print(f"\n❌ Failed Models:")
            for test in failed_tests:
                print(f"   - {test['model_name']}: {test.get('error', 'Unknown error')}")

def main():
    """Main function"""
    tester = ZamAIModelTester()
    
    # Test all models
    tester.test_all_models()
    
    print("\n🎉 Model testing completed!")

if __name__ == "__main__":
    main()
