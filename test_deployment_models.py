#!/usr/bin/env python3
"""
ZamAI Pro Models - Comprehensive Testing Script
Test all models including the new ones: Whisper Large v3, Mistral 7B, Phi-3 Mini
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional
from huggingface_hub import HfApi, login
from transformers import AutoTokenizer, AutoModelForCausalLM, WhisperProcessor, WhisperForConditionalGeneration
import torch

class ZamAIModelTester:
    def __init__(self):
        self.setup_environment()
        self.load_test_config()
        
    def setup_environment(self):
        """Setup testing environment"""
        print("🔧 Setting up testing environment...")
        
        # Load HF token
        token_file = "HF-Token.txt"
        if os.path.exists(token_file):
            with open(token_file, 'r') as f:
                self.hf_token = f.read().strip()
        else:
            self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        
        if not self.hf_token:
            raise ValueError("❌ Hugging Face token not found. Set HUGGINGFACE_TOKEN or create HF-Token.txt")
        
        # Login to HF
        login(token=self.hf_token)
        self.api = HfApi(token=self.hf_token)
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📱 Using device: {self.device}")
        
    def load_test_config(self):
        """Load test configuration"""
        try:
            with open("deployment_config.json", "r") as f:
                config = json.load(f)
                self.models_to_test = config["deployment_models"]
        except Exception as e:
            print(f"⚠️ Could not load deployment config: {e}")
            # Fallback configuration
            self.models_to_test = {
                "speech_to_text": {"primary": "openai/whisper-large-v3"},
                "text_generation": {
                    "primary": "mistralai/Mistral-7B-Instruct-v0.3",
                    "secondary": "microsoft/Phi-3-mini-4k-instruct"
                }
            }
    
    def test_model_availability(self, model_id: str) -> Dict:
        """Test if a model is available on HF Hub"""
        print(f"🔍 Testing availability: {model_id}")
        
        try:
            # Check if model exists
            model_info = self.api.model_info(model_id)
            
            # Test inference endpoint
            inference_url = f"https://api-inference.huggingface.co/models/{model_id}"
            headers = {"Authorization": f"Bearer {self.hf_token}"}
            
            # Simple test request
            response = requests.post(
                inference_url,
                headers=headers,
                json={"inputs": "test"},
                timeout=30
            )
            
            status = "available"
            if response.status_code == 503:
                status = "loading"
            elif response.status_code != 200:
                status = "error"
            
            return {
                "model_id": model_id,
                "available": True,
                "inference_status": status,
                "status_code": response.status_code,
                "downloads": model_info.downloads if hasattr(model_info, 'downloads') else 0,
                "size": getattr(model_info, 'safetensors', {}).get('total', 'unknown')
            }
            
        except Exception as e:
            return {
                "model_id": model_id,
                "available": False,
                "error": str(e)
            }
    
    def test_whisper_model(self, model_id: str = "openai/whisper-large-v3") -> Dict:
        """Test Whisper model for speech recognition"""
        print(f"🎤 Testing Whisper model: {model_id}")
        
        try:
            # Load processor and model
            processor = WhisperProcessor.from_pretrained(model_id)
            model = WhisperForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
            
            # Test with dummy audio (silence)
            import numpy as np
            dummy_audio = np.zeros(16000)  # 1 second of silence at 16kHz
            
            # Process audio
            inputs = processor(dummy_audio, sampling_rate=16000, return_tensors="pt")
            
            # Generate transcription
            with torch.no_grad():
                generated_ids = model.generate(inputs["input_features"])
            
            transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return {
                "model_id": model_id,
                "status": "success",
                "test_result": transcription,
                "capabilities": ["speech-to-text", "multilingual", "timestamps"],
                "memory_usage": f"{torch.cuda.max_memory_allocated() / 1024**3:.2f}GB" if torch.cuda.is_available() else "N/A"
            }
            
        except Exception as e:
            return {
                "model_id": model_id,
                "status": "error",
                "error": str(e)
            }
    
    def test_text_generation_model(self, model_id: str) -> Dict:
        """Test text generation model"""
        print(f"💬 Testing text generation model: {model_id}")
        
        try:
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
            
            # Test prompts
            test_prompts = [
                "Hello, how are you?",
                "What is the capital of Afghanistan?",
                "Tell me about Pashto language.",
                "سلام وروره! څنګه یاست؟"  # Pashto greeting
            ]
            
            results = []
            for prompt in test_prompts:
                try:
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=50,
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    response = response.replace(prompt, "").strip()
                    
                    results.append({
                        "prompt": prompt,
                        "response": response[:100] + "..." if len(response) > 100 else response
                    })
                    
                except Exception as e:
                    results.append({
                        "prompt": prompt,
                        "error": str(e)
                    })
            
            return {
                "model_id": model_id,
                "status": "success",
                "test_results": results,
                "model_size": f"{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters",
                "memory_usage": f"{torch.cuda.max_memory_allocated() / 1024**3:.2f}GB" if torch.cuda.is_available() else "N/A"
            }
            
        except Exception as e:
            return {
                "model_id": model_id,
                "status": "error",
                "error": str(e)
            }
    
    def test_inference_api(self, model_id: str) -> Dict:
        """Test model via HF Inference API"""
        print(f"🌐 Testing inference API: {model_id}")
        
        try:
            inference_url = f"https://api-inference.huggingface.co/models/{model_id}"
            headers = {"Authorization": f"Bearer {self.hf_token}"}
            
            # Test text generation
            if "whisper" not in model_id.lower():
                payload = {
                    "inputs": "Hello, how can I help you today?",
                    "parameters": {
                        "max_new_tokens": 50,
                        "temperature": 0.7,
                        "do_sample": True
                    }
                }
                
                response = requests.post(
                    inference_url,
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "model_id": model_id,
                        "status": "success",
                        "api_response": result,
                        "latency": response.elapsed.total_seconds()
                    }
                else:
                    return {
                        "model_id": model_id,
                        "status": "error",
                        "status_code": response.status_code,
                        "error": response.text
                    }
            else:
                # For Whisper models, we can't easily test without audio
                return {
                    "model_id": model_id,
                    "status": "skipped",
                    "reason": "Audio input required for Whisper models"
                }
                
        except Exception as e:
            return {
                "model_id": model_id,
                "status": "error",
                "error": str(e)
            }
    
    def run_comprehensive_tests(self) -> Dict:
        """Run all tests and return comprehensive results"""
        print("🚀 Starting comprehensive model testing...")
        
        results = {
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "environment": {
                "device": str(self.device),
                "cuda_available": torch.cuda.is_available(),
                "torch_version": torch.__version__
            },
            "models": {}
        }
        
        # Test models from deployment config
        all_models = set()
        
        # Collect all models to test
        for category, models in self.models_to_test.items():
            if isinstance(models, dict):
                if "primary" in models:
                    all_models.add(models["primary"])
                if "secondary" in models:
                    all_models.add(models["secondary"])
                if "models" in models:
                    for model_id in models["models"].keys():
                        all_models.add(model_id)
            elif isinstance(models, str):
                all_models.add(models)
        
        # Test each model
        for model_id in all_models:
            print(f"\n{'='*60}")
            print(f"Testing: {model_id}")
            print(f"{'='*60}")
            
            model_results = {
                "availability": self.test_model_availability(model_id),
                "inference_api": self.test_inference_api(model_id)
            }
            
            # Specific tests based on model type
            if "whisper" in model_id.lower():
                model_results["whisper_test"] = self.test_whisper_model(model_id)
            else:
                model_results["text_generation_test"] = self.test_text_generation_model(model_id)
            
            results["models"][model_id] = model_results
        
        return results
    
    def generate_report(self, results: Dict) -> str:
        """Generate a comprehensive test report"""
        report = f"""
# 🇦🇫 ZamAI Pro Models - Test Report

**Generated**: {results['test_timestamp']}
**Environment**: {results['environment']['device']} (CUDA: {results['environment']['cuda_available']})
**PyTorch Version**: {results['environment']['torch_version']}

## 📊 Test Summary

"""
        
        total_models = len(results['models'])
        successful_models = sum(1 for model_data in results['models'].values() 
                              if any(test.get('status') == 'success' for test in model_data.values()))
        
        report += f"- **Total Models Tested**: {total_models}\n"
        report += f"- **Successful Tests**: {successful_models}\n"
        report += f"- **Success Rate**: {successful_models/total_models*100:.1f}%\n\n"
        
        # Detailed results
        for model_id, model_data in results['models'].items():
            report += f"## 🤖 {model_id}\n\n"
            
            # Availability
            avail = model_data.get('availability', {})
            if avail.get('available'):
                report += f"✅ **Available** - Downloads: {avail.get('downloads', 'N/A')}\n"
            else:
                report += f"❌ **Not Available** - {avail.get('error', 'Unknown error')}\n"
            
            # Test results
            for test_type, test_data in model_data.items():
                if test_type == 'availability':
                    continue
                    
                status = test_data.get('status', 'unknown')
                if status == 'success':
                    report += f"✅ **{test_type.replace('_', ' ').title()}**: Passed\n"
                elif status == 'error':
                    report += f"❌ **{test_type.replace('_', ' ').title()}**: {test_data.get('error', 'Unknown error')}\n"
                elif status == 'skipped':
                    report += f"⏭️ **{test_type.replace('_', ' ').title()}**: {test_data.get('reason', 'Skipped')}\n"
            
            report += "\n"
        
        return report
    
    def save_results(self, results: Dict, filename: str = "model_test_results.json"):
        """Save test results to JSON file"""
        os.makedirs("data/processed", exist_ok=True)
        filepath = f"data/processed/{filename}"
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Results saved to: {filepath}")

def main():
    """Main testing function"""
    print("🇦🇫 ZamAI Pro Models - Comprehensive Testing")
    print("=" * 50)
    
    try:
        tester = ZamAIModelTester()
        results = tester.run_comprehensive_tests()
        
        # Save results
        tester.save_results(results)
        
        # Generate and save report
        report = tester.generate_report(results)
        
        with open("data/processed/model_test_report.md", "w") as f:
            f.write(report)
        
        print("\n" + "=" * 50)
        print("📊 TESTING COMPLETED")
        print("=" * 50)
        print(report)
        
        return results
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        return None

if __name__ == "__main__":
    main()
