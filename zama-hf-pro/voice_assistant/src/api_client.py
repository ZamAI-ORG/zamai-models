import requests
import os
import json
import base64
from typing import Optional, Union
import tempfile

class HFInferenceClient:
    def __init__(self, token: str):
        self.token = token
        self.headers = {"Authorization": f"Bearer {token}"}
        self.api_url = "https://api-inference.huggingface.co/models"
        
        # Default models from deployment config
        self.default_models = {
            "stt": "openai/whisper-large-v3",
            "llm": "mistralai/Mistral-7B-Instruct-v0.3",
            "llm_lightweight": "microsoft/Phi-3-mini-4k-instruct",
            "tts": "microsoft/speecht5_tts"
        }
    
    def generate_text(self, prompt: str, model: str = None, max_length: int = 150, temperature: float = 0.7) -> str:
        """Generate text using HF model with improved parameters"""
        if model is None:
            model = self.default_models["llm"]
            
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_length,
                "temperature": temperature,
                "do_sample": True,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1,
                "return_full_text": False
            }
        }
        
        response = requests.post(
            f"{self.api_url}/{model}",
            headers=self.headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
                # Clean up the response
                if prompt in generated_text:
                    generated_text = generated_text.replace(prompt, "").strip()
                return generated_text
            elif isinstance(result, dict):
                return result.get("generated_text", str(result))
            return str(result)
        else:
            raise Exception(f"Text Generation Error: {response.status_code} - {response.text}")
    
    def stt_process(self, audio_file: str, model: str = None) -> str:
        """Process speech to text using Whisper Large v3"""
        if model is None:
            model = self.default_models["stt"]
            
        if not os.path.exists(audio_file):
            raise Exception(f"Audio file not found: {audio_file}")
            
        try:
            with open(audio_file, "rb") as f:
                audio_data = f.read()
                
            response = requests.post(
                f"{self.api_url}/{model}",
                headers=self.headers,
                data=audio_data,
                timeout=120  # Whisper can take longer
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, dict):
                    return result.get("text", "")
                elif isinstance(result, list) and len(result) > 0:
                    return result[0].get("text", "")
                return str(result)
            else:
                raise Exception(f"STT Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            raise Exception(f"STT Processing Error: {str(e)}")
    
    def tts_process(self, text: str, model: str = None) -> Optional[bytes]:
        """Process text to speech"""
        if model is None:
            model = self.default_models["tts"]
            
        try:
            response = requests.post(
                f"{self.api_url}/{model}",
                headers=self.headers,
                json={"inputs": text},
                timeout=60
            )
            
            if response.status_code == 200:
                # Check if response is audio data
                if response.headers.get('content-type', '').startswith('audio/'):
                    return response.content
                else:
                    # Some TTS models return base64 encoded audio
                    result = response.json()
                    if isinstance(result, dict) and 'audio' in result:
                        return base64.b64decode(result['audio'])
                    return None
            else:
                print(f"TTS Warning: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"TTS Error: {str(e)}")
            return None
    
    def chat_completion(self, messages: list, model: str = None, max_tokens: int = 300) -> str:
        """Chat completion with conversation context"""
        if model is None:
            model = self.default_models["llm"]
        
        # Convert messages to a single prompt
        prompt = ""
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        
        prompt += "Assistant:"
        
        return self.generate_text(prompt, model, max_tokens)
    
    def get_model_info(self, model: str) -> dict:
        """Get information about a model"""
        try:
            response = requests.get(
                f"https://huggingface.co/api/models/{model}",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Could not get model info: {response.status_code}"}
                
        except Exception as e:
            return {"error": str(e)}
    
    def check_model_status(self, model: str) -> dict:
        """Check if a model is loaded and ready"""
        try:
            # Test with a simple request
            response = requests.post(
                f"{self.api_url}/{model}",
                headers=self.headers,
                json={"inputs": "test"},
                timeout=10
            )
            
            return {
                "model": model,
                "status": "ready" if response.status_code in [200, 503] else "error",
                "status_code": response.status_code
            }
            
        except Exception as e:
            return {
                "model": model,
                "status": "error",
                "error": str(e)
            }
        
        if response.status_code == 200:
            return response.content
        else:
            raise Exception(f"TTS Error: {response.status_code}")
