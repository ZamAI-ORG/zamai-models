import requests
import os
from typing import Optional

class HFInferenceClient:
    def __init__(self, token: str):
        self.token = token
        self.headers = {"Authorization": f"Bearer {token}"}
        self.api_url = "https://api-inference.huggingface.co/models"
    
    def generate_text(self, prompt: str, model: str, max_tokens: int = 150) -> str:
        """Generate text using HF model"""
        response = requests.post(
            f"{self.api_url}/{model}",
            headers=self.headers,
            json={"inputs": prompt, "parameters": {"max_new_tokens": max_tokens}}
        )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "").replace(prompt, "").strip()
            return str(result)
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
    
    def stt_process(self, audio_file: str, model: str) -> str:
        """Process speech to text"""
        with open(audio_file, "rb") as f:
            audio_data = f.read()
            
        response = requests.post(
            f"{self.api_url}/{model}",
            headers=self.headers,
            data=audio_data
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("text", "")
        else:
            raise Exception(f"STT Error: {response.status_code}")
    
    def tts_process(self, text: str, model: str) -> bytes:
        """Process text to speech"""
        response = requests.post(
            f"{self.api_url}/{model}",
            headers=self.headers,
            json={"inputs": text}
        )
        
        if response.status_code == 200:
            return response.content
        else:
            raise Exception(f"TTS Error: {response.status_code}")
