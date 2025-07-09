"""
Hugging Face Inference Client for ZamAI Models
Supports both Inference API and Inference Endpoints
"""

import os
import json
from typing import Dict, List, Optional, Union
from huggingface_hub import InferenceClient
import asyncio

class ZamAIInferenceClient:
    """Client for running ZamAI models using HF Inference"""
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize the inference client
        
        Args:
            token: HF token (reads from HF-Token.txt if not provided)
        """
        if token is None:
            try:
                with open('/workspaces/ZamAI-Pro-Models/HF-Token.txt', 'r') as f:
                    token = f.read().strip()
            except FileNotFoundError:
                print("Warning: No HF token found. Some features may be limited.")
                token = None
        
        self.client = InferenceClient(token=token)
        self.token = token
    
    def chat_completion(
        self, 
        model_id: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Run chat completion using HF Inference API
        
        Args:
            model_id: Model ID on HuggingFace Hub
            messages: Chat messages in OpenAI format
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response
        """
        try:
            response = self.client.chat_completion(
                messages=messages,
                model=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in chat completion: {e}")
            return f"Error: {e}"
    
    def text_generation(
        self,
        model_id: str,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate text using HF Inference API
        
        Args:
            model_id: Model ID on HuggingFace Hub
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        try:
            response = self.client.text_generation(
                prompt=prompt,
                model=model_id,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs
            )
            return response
        except Exception as e:
            print(f"Error in text generation: {e}")
            return f"Error: {e}"
    
    def embeddings(
        self,
        model_id: str,
        texts: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings using HF Inference API
        
        Args:
            model_id: Embedding model ID
            texts: Text(s) to embed
            
        Returns:
            Embeddings
        """
        try:
            response = self.client.feature_extraction(
                text=texts,
                model=model_id
            )
            return response
        except Exception as e:
            print(f"Error in embeddings: {e}")
            return []
    
    def automatic_speech_recognition(
        self,
        model_id: str,
        audio_path: str
    ) -> str:
        """
        Convert speech to text using HF Inference API
        
        Args:
            model_id: ASR model ID
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        try:
            with open(audio_path, "rb") as f:
                response = self.client.automatic_speech_recognition(
                    audio=f.read(),
                    model=model_id
                )
            return response.get("text", "")
        except Exception as e:
            print(f"Error in ASR: {e}")
            return f"Error: {e}"
    
    def test_pashto_chat(self, prompt: str) -> str:
        """
        Test your fine-tuned Pashto chat model
        """
        # Use your model ID from the config
        model_id = "tasal9/zamai-pashto-chat-8b"
        
        messages = [
            {
                "role": "system", 
                "content": "تاسو د پښتو ژبې یو ګټور مرستیال یاست. د افغان کلتور په درناوي سره ځواب ورکړئ."
            },
            {"role": "user", "content": prompt}
        ]
        
        return self.chat_completion(model_id, messages)

# Example usage functions
def test_inference_api():
    """Test the Inference API with various models"""
    client = ZamAIInferenceClient()
    
    print("=== Testing HF Inference API ===\n")
    
    # Test 1: General language model
    print("1. Testing general chat model:")
    response = client.chat_completion(
        model_id="microsoft/DialoGPT-medium",
        messages=[{"role": "user", "content": "Hello, how are you?"}]
    )
    print(f"Response: {response}\n")
    
    # Test 2: Text generation
    print("2. Testing text generation:")
    response = client.text_generation(
        model_id="gpt2",
        prompt="The future of AI in Afghanistan",
        max_new_tokens=100
    )
    print(f"Generated: {response}\n")
    
    # Test 3: Embeddings
    print("3. Testing embeddings:")
    embeddings = client.embeddings(
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        texts=["Hello world", "پښتو ژبه"]
    )
    print(f"Embedding dimensions: {len(embeddings[0]) if embeddings else 0}\n")

def test_zamai_models():
    """Test your specific ZamAI models"""
    client = ZamAIInferenceClient()
    
    print("=== Testing ZamAI Models ===\n")
    
    # Test your Pashto chat model (when deployed)
    print("Testing Pashto Chat Model:")
    try:
        response = client.test_pashto_chat("سلام ورور، ستاسو څنګه یاست؟")
        print(f"Pashto Response: {response}")
    except Exception as e:
        print(f"Model not yet deployed: {e}")

if __name__ == "__main__":
    test_inference_api()
    test_zamai_models()
