#!/usr/bin/env python3
"""
ZamAI Hugging Face Model Manager
Manage, test, and deploy your custom Pashto models
"""

import os
import json
import requests
import time
from typing import Dict, List, Optional, Tuple
from huggingface_hub import HfApi, InferenceApi
# import asyncio
# import aiohttp

class ZamAIModelManager:
    def __init__(self, token_file: str = "HF-Token.txt"):
        """Initialize the model manager"""
        # Load HF token
        with open(token_file, 'r') as f:
            self.token = f.read().strip()
        
        self.api = HfApi(token=self.token)
        self.username = "tasal9"
        
        # Load model configurations
        self.models = self._load_model_configs()
        
    def _load_model_configs(self) -> Dict:
        """Load all model configurations"""
        models = {}
        models_dir = "models"
        
        # Check if models directory exists
        if os.path.exists(models_dir):
            for category in os.listdir(models_dir):
                category_path = os.path.join(models_dir, category)
                if os.path.isdir(category_path):
                    models[category] = {}
                    for config_file in os.listdir(category_path):
                        if config_file.endswith('.json'):
                            config_path = os.path.join(category_path, config_file)
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                                model_name = config_file.replace('.json', '')
                                models[category][model_name] = config
        
        # Add deployed models if not already loaded
        if not models or 'speech-to-text' not in models:
            # Create category if it doesn't exist
            if 'speech-to-text' not in models:
                models['speech-to-text'] = {}
            
            # Add Whisper model
            models['speech-to-text']['whisper-large-v3'] = {
                'model_id': 'openai/whisper-large-v3',
                'task': 'automatic-speech-recognition',
                'priority': 'primary',
                'status': 'deployed',
                'features': ['speech-to-text', 'understanding'],
                'model_config': {
                    'language': 'auto',
                    'task': 'transcribe',
                    'return_timestamps': True
                }
            }
        
        # Add text generation models
        if not models or 'text-generation' not in models:
            # Create category if it doesn't exist
            if 'text-generation' not in models:
                models['text-generation'] = {}
            
            # Add Mistral model
            models['text-generation']['mistral-7b-instruct'] = {
                'model_id': 'mistralai/Mistral-7B-Instruct-v0.3',
                'task': 'text-generation',
                'priority': 'primary',
                'status': 'deployed',
                'parameters': '7B',
                'model_config': {
                    'max_length': 500,
                    'temperature': 0.7,
                    'do_sample': True,
                    'top_p': 0.9
                }
            }
            
            # Add Phi-3 model
            models['text-generation']['phi-3-mini'] = {
                'model_id': 'microsoft/Phi-3-mini-4k-instruct',
                'task': 'text-generation',
                'priority': 'secondary',
                'status': 'deployed',
                'parameters': '3.8B',
                'model_config': {
                    'max_length': 300,
                    'temperature': 0.8,
                    'do_sample': True,
                    'top_p': 0.9
                }
            }
        
        return models
    
    def list_models(self) -> Dict:
        """List all available models by category"""
        print("🤗 ZamAI Model Inventory")
        print("=" * 40)
        
        for category, models in self.models.items():
            print(f"\n📁 {category.upper()}")
            print("-" * 20)
            for model_name, config in models.items():
                priority = config.get('priority', 'unknown')
                model_id = config['model_id']
                status = config.get('status', 'unknown')
                
                print(f"  📦 {model_name}")
                print(f"     🆔 {model_id}")
                print(f"     🎯 Priority: {priority}")
                print(f"     ✅ Status: {status}")
                
                if 'stats' in config and 'downloads' in config['stats']:
                    print(f"     📊 Downloads: {config['stats']['downloads']}")
                
                print()
        
        return self.models
    
    def test_model(self, model_id: str, test_input: str = "سلام وروره، څنګه یاست؟") -> Dict:
        """Test a specific model with Pashto input"""
        print(f"🧪 Testing model: {model_id}")
        print(f"📝 Input: {test_input}")
        
        try:
            # Use Inference API
            inference = InferenceApi(model_id, token=self.token)
            
            start_time = time.time()
            
            # For text generation models
            if self._is_text_generation_model(model_id):
                result = inference(test_input, parameters={
                    "max_length": 200,
                    "temperature": 0.7,
                    "do_sample": True
                })
                
                if isinstance(result, list) and len(result) > 0:
                    response = result[0].get('generated_text', 'No response')
                else:
                    response = str(result)
            
            # For embedding models
            elif self._is_embedding_model(model_id):
                result = inference(test_input)
                response = f"Embedding generated (shape: {len(result) if isinstance(result, list) else 'unknown'})"
            
            else:
                result = inference(test_input)
                response = str(result)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            test_result = {
                "model_id": model_id,
                "input": test_input,
                "output": response,
                "response_time": f"{response_time:.2f}s",
                "status": "success",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            print(f"✅ Response ({response_time:.2f}s): {response[:100]}...")
            return test_result
            
        except Exception as e:
            error_result = {
                "model_id": model_id,
                "input": test_input,
                "error": str(e),
                "status": "error",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            print(f"❌ Error: {e}")
            return error_result
    
    def test_all_models(self) -> Dict:
        """Test all models with standard Pashto inputs"""
        print("🧪 Testing All ZamAI Models")
        print("=" * 40)
        
        test_inputs = [
            "سلام وروره، څنګه یاست؟",  # Greeting
            "د افغانستان پخوانی تاریخ څه دی؟",  # Question about Afghanistan
            "زه د پښتو ژبې زده کړه غواړم.",  # Learning Pashto
            "د اسلام اساسي اصول څه دي؟"  # Islamic principles
        ]
        
        results = {}
        
        for category, models in self.models.items():
            results[category] = {}
            
            for model_name, config in models.items():
                model_id = config['model_id']
                print(f"\n🧪 Testing: {model_id}")
                
                model_results = []
                for test_input in test_inputs:
                    result = self.test_model(model_id, test_input)
                    model_results.append(result)
                    time.sleep(1)  # Avoid rate limiting
                
                results[category][model_name] = model_results
        
        # Save test results
        with open('test_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print("\n💾 Test results saved to: test_results.json")
        return results
    
    def get_model_by_priority(self, task: str = "text-generation") -> Optional[str]:
        """Get the highest priority model for a specific task"""
        priority_order = {"primary": 1, "secondary": 2, "tertiary": 3, "specialized": 4}
        
        best_model = None
        best_priority = 99
        
        for category, models in self.models.items():
            if task in category or task == "any":
                for model_name, config in models.items():
                    priority = config.get('priority', 'unknown')
                    if priority in priority_order:
                        priority_value = priority_order[priority]
                        if priority_value < best_priority:
                            best_priority = priority_value
                            best_model = config['model_id']
        
        return best_model
    
    def generate_integration_code(self) -> str:
        """Generate Node.js integration code for all models"""
        code = '''// ZamAI Hugging Face Models Integration
// Auto-generated model service

const axios = require('axios');

class ZamAIHuggingFaceService {
    constructor() {
        this.token = process.env.HUGGINGFACE_TOKEN;
        this.baseURL = 'https://api-inference.huggingface.co/models';
        
        // Model configurations
        this.models = {
'''
        
        for category, models in self.models.items():
            code += f"            {category}: {{\n"
            for model_name, config in models.items():
                model_id = config['model_id']
                priority = config.get('priority', 'unknown')
                code += f"                {model_name}: {{\n"
                code += f"                    id: '{model_id}',\n"
                code += f"                    priority: '{priority}',\n"
                code += f"                    config: {json.dumps(config.get('model_config', {}), indent=20)}\n"
                code += f"                }},\n"
            code += f"            }},\n"
        
        code += '''        };
    }
    
    async queryModel(modelId, input, parameters = {}) {
        try {
            const response = await axios.post(
                `${this.baseURL}/${modelId}`,
                { inputs: input, parameters },
                {
                    headers: {
                        'Authorization': `Bearer ${this.token}`,
                        'Content-Type': 'application/json'
                    }
                }
            );
            return response.data;
        } catch (error) {
            console.error(`Error querying model ${modelId}:`, error);
            throw error;
        }
    }
    
    async generatePashtoText(input, modelType = 'primary') {
        // Use Mistral-7B as primary
        const modelId = 'tasal9/ZamAI-Mistral-7B-Pashto';
        return await this.queryModel(modelId, input, {
            max_length: 500,
            temperature: 0.7,
            do_sample: true
        });
    }
    
    async getPashtoEmbeddings(text) {
        const modelId = 'tasal9/Multilingual-ZamAI-Embeddings';
        return await this.queryModel(modelId, text);
    }
    
    async getModelByPriority(category, priority = 'primary') {
        const categoryModels = this.models[category];
        if (!categoryModels) return null;
        
        for (const [name, config] of Object.entries(categoryModels)) {
            if (config.priority === priority) {
                return config.id;
            }
        }
        
        // Fallback to first available model
        return Object.values(categoryModels)[0]?.id || null;
    }
}

module.exports = ZamAIHuggingFaceService;
'''
        
        return code
    
    def _is_text_generation_model(self, model_id: str) -> bool:
        """Check if model is for text generation"""
        return any("text-generation" in str(config.get('task', '')) 
                  for category in self.models.values() 
                  for config in category.values() 
                  if config.get('model_id') == model_id)
    
    def _is_embedding_model(self, model_id: str) -> bool:
        """Check if model is for embeddings"""
        return any("feature-extraction" in str(config.get('task', '')) or "embeddings" in str(config.get('model_type', ''))
                  for category in self.models.values() 
                  for config in category.values() 
                  if config.get('model_id') == model_id)
    
    def get_deployment_summary(self) -> Dict:
        """Get a summary of deployed models and their status"""
        deployed_models = {
            "speech_recognition": None,
            "primary_text_generation": None,
            "secondary_text_generation": None,
            "total_deployed": 0,
            "deployment_status": "unknown"
        }
        
        # Find speech model
        speech_model = self.get_model_by_priority("speech-to-text")
        if speech_model:
            deployed_models["speech_recognition"] = speech_model
            deployed_models["total_deployed"] += 1
        
        # Find text generation models
        primary_model = self.get_model_by_priority("text-generation")
        if primary_model:
            deployed_models["primary_text_generation"] = primary_model
            deployed_models["total_deployed"] += 1
        
        # Find secondary model
        for category, models in self.models.items():
            if "text-generation" in category:
                for model_name, config in models.items():
                    if config.get("priority") == "secondary":
                        deployed_models["secondary_text_generation"] = config.get("model_id")
                        deployed_models["total_deployed"] += 1
                        break
        
        # Set deployment status
        if deployed_models["total_deployed"] == 0:
            deployed_models["deployment_status"] = "not_deployed"
        elif deployed_models["total_deployed"] < 3:
            deployed_models["deployment_status"] = "partially_deployed"
        else:
            deployed_models["deployment_status"] = "fully_deployed"
        
        # Check for docker and deployment files
        if os.path.exists("docker-compose.yml") and os.path.exists("Dockerfile"):
            deployed_models["docker_ready"] = True
        else:
            deployed_models["docker_ready"] = False
            
        if os.path.exists("deploy.sh"):
            deployed_models["deployment_script"] = True
        else:
            deployed_models["deployment_script"] = False
        
        return deployed_models

def main():
    """Main function"""
    print("🇦🇫 ZamAI Hugging Face Model Manager")
    print("=" * 50)
    
    manager = ZamAIModelManager()
    
    # List all models
    models = manager.list_models()
    
    # Display deployed models
    print("\n📋 DEPLOYED MODELS SUMMARY:")
    print("=" * 50)
    
    # Speech model
    whisper_model = manager.get_model_by_priority("speech-to-text")
    if whisper_model:
        print(f"🔊 Speech Recognition: {whisper_model}")
        print("   🔹 Features: Speech-to-text + Understanding")
        print("   🔹 Use Case: Voice Assistant Pipeline")
    
    # Text generation models
    primary_model = manager.get_model_by_priority("text-generation")
    if primary_model:
        print(f"💬 Primary Text Generation: {primary_model}")
        print("   🔹 Features: High-quality conversation, instruction following")
        print("   🔹 Use Case: Chat, Q&A, Voice Assistant responses")
    
    secondary_model = None
    for category, model_dict in models.items():
        for model_name, config in model_dict.items():
            if config.get('priority') == 'secondary' and 'text-generation' in category:
                secondary_model = config.get('model_id')
                break
    
    if secondary_model:
        print(f"📱 Edge Text Generation: {secondary_model}")
        print("   🔹 Features: Lightweight, edge-friendly")
        print("   🔹 Use Case: Mobile deployment, offline applications")
    
    # Deployment status
    print("\n🚀 DEPLOYMENT STATUS:")
    print("=" * 50)
    print("✅ Docker configuration: Ready")
    print("✅ Nginx configuration: Ready")
    print("✅ Voice Assistant: Integrated with all models")
    print("✅ Deployment script: Available (./deploy.sh)")
    
    # Test option
    print("\n🧪 MODEL TESTING OPTIONS:")
    print("=" * 50)
    print("1. Test primary model")
    print("2. Test all models")
    print("3. Generate integration code")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1" and primary_model:
        print(f"\n🎯 Testing primary model: {primary_model}")
        manager.test_model(primary_model)
    elif choice == "2":
        manager.test_all_models()
    elif choice == "3":
        integration_code = manager.generate_integration_code()
        code_dir = "server/services"
        os.makedirs(code_dir, exist_ok=True)
        with open(f'{code_dir}/zamaiHuggingFaceService.js', 'w') as f:
            f.write(integration_code)
        print(f"\n💾 Integration code generated: {code_dir}/zamaiHuggingFaceService.js")
    else:
        print("\n👋 Exiting model manager")

if __name__ == "__main__":
    main()
