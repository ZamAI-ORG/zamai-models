"""
Deploy ZamAI Models to HF Inference Endpoints
Automates the deployment process for production use
"""

import json
import os
import time
from typing import Dict, Optional
from huggingface_hub import HfApi, InferenceClient
import requests

class ZamAIEndpointManager:
    """Manage HF Inference Endpoints for ZamAI models"""
    
    def __init__(self, token: Optional[str] = None):
        """Initialize with HF token"""
        if token is None:
            try:
                with open('/workspaces/ZamAI-Pro-Models/HF-Token.txt', 'r') as f:
                    token = f.read().strip()
            except FileNotFoundError:
                raise ValueError("HF token required. Place it in HF-Token.txt")
        
        self.api = HfApi(token=token)
        self.token = token
        self.headers = {"Authorization": f"Bearer {token}"}
    
    def create_endpoint(
        self,
        endpoint_name: str,
        model_id: str,
        instance_size: str = "medium",
        instance_type: str = "nvidia-tesla-t4",
        min_replica: int = 0,
        max_replica: int = 1,
        **kwargs
    ) -> Dict:
        """
        Create a new Inference Endpoint
        
        Args:
            endpoint_name: Name for the endpoint
            model_id: HuggingFace model ID
            instance_size: small, medium, large, xlarge
            instance_type: GPU type
            min_replica: Minimum replicas (0 for auto-scaling)
            max_replica: Maximum replicas
            
        Returns:
            Endpoint creation response
        """
        
        endpoint_config = {
            "compute": {
                "accelerator": "gpu",
                "instanceSize": instance_size,
                "instanceType": instance_type,
                "scaling": {
                    "minReplica": min_replica,
                    "maxReplica": max_replica
                }
            },
            "model": {
                "framework": "pytorch",
                "repository": model_id,
                "task": "text-generation",
                "image": {
                    "huggingface": {}
                }
            },
            "name": endpoint_name,
            "provider": {
                "region": "us-east-1",
                "vendor": "aws"
            }
        }
        
        # Add any additional config
        endpoint_config.update(kwargs)
        
        try:
            response = requests.post(
                "https://api.endpoints.huggingface.co/v2/endpoint",
                headers=self.headers,
                json=endpoint_config
            )
            response.raise_for_status()
            result = response.json()
            
            print(f"✅ Endpoint '{endpoint_name}' creation initiated")
            print(f"Status: {result.get('status', 'Unknown')}")
            print(f"URL: {result.get('url', 'Pending')}")
            
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error creating endpoint: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Response: {e.response.text}")
            return {}
    
    def list_endpoints(self) -> Dict:
        """List all your endpoints"""
        try:
            response = requests.get(
                "https://api.endpoints.huggingface.co/v2/endpoint",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error listing endpoints: {e}")
            return {}
    
    def get_endpoint_status(self, endpoint_name: str) -> Dict:
        """Get status of specific endpoint"""
        try:
            response = requests.get(
                f"https://api.endpoints.huggingface.co/v2/endpoint/{endpoint_name}",
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting endpoint status: {e}")
            return {}
    
    def wait_for_endpoint(self, endpoint_name: str, timeout: int = 900) -> bool:
        """
        Wait for endpoint to be ready
        
        Args:
            endpoint_name: Name of endpoint
            timeout: Timeout in seconds (default 15 minutes)
            
        Returns:
            True if endpoint is ready, False if timeout
        """
        print(f"⏳ Waiting for endpoint '{endpoint_name}' to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_endpoint_status(endpoint_name)
            current_status = status.get('status', 'unknown')
            
            print(f"Status: {current_status}")
            
            if current_status == 'running':
                print(f"✅ Endpoint '{endpoint_name}' is ready!")
                return True
            elif current_status == 'failed':
                print(f"❌ Endpoint '{endpoint_name}' failed to deploy")
                return False
            
            time.sleep(30)  # Check every 30 seconds
        
        print(f"⏰ Timeout waiting for endpoint '{endpoint_name}'")
        return False
    
    def test_endpoint(self, endpoint_url: str, test_prompt: str) -> str:
        """Test the deployed endpoint"""
        client = InferenceClient(model=endpoint_url, token=self.token)
        
        try:
            response = client.text_generation(
                prompt=test_prompt,
                max_new_tokens=100
            )
            return response
        except Exception as e:
            return f"Error testing endpoint: {e}"
    
    def delete_endpoint(self, endpoint_name: str) -> bool:
        """Delete an endpoint"""
        try:
            response = requests.delete(
                f"https://api.endpoints.huggingface.co/v2/endpoint/{endpoint_name}",
                headers=self.headers
            )
            response.raise_for_status()
            print(f"✅ Endpoint '{endpoint_name}' deleted")
            return True
        except Exception as e:
            print(f"❌ Error deleting endpoint: {e}")
            return False

def deploy_zamai_models():
    """Deploy ZamAI models to Inference Endpoints"""
    
    manager = ZamAIEndpointManager()
    
    # Load Pashto chat config
    config_path = "/workspaces/ZamAI-Pro-Models/configs/pashto_chat_config.json"
    if not os.path.exists(config_path):
        print("❌ Pashto chat config not found")
        return
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model_id = config['hub_model_id']
    endpoint_name = f"zamai-pashto-chat-{int(time.time())}"
    
    print(f"🚀 Deploying {model_id} to endpoint '{endpoint_name}'")
    
    # Create endpoint
    result = manager.create_endpoint(
        endpoint_name=endpoint_name,
        model_id=model_id,
        instance_size="medium",  # Adjust based on your needs
        min_replica=0,  # Auto-scaling
        max_replica=2
    )
    
    if not result:
        print("❌ Failed to create endpoint")
        return
    
    # Wait for deployment
    if manager.wait_for_endpoint(endpoint_name):
        # Test the endpoint
        endpoint_url = result.get('url', '')
        if endpoint_url:
            test_prompt = "سلام ورور، ستاسو څنګه یاست؟"
            response = manager.test_endpoint(endpoint_url, test_prompt)
            print(f"🧪 Test Response: {response}")
            
            # Save endpoint info
            endpoint_info = {
                "name": endpoint_name,
                "url": endpoint_url,
                "model_id": model_id,
                "created_at": time.time(),
                "status": "running"
            }
            
            info_path = "/workspaces/ZamAI-Pro-Models/data/processed/endpoint_info.json"
            with open(info_path, 'w') as f:
                json.dump(endpoint_info, f, indent=2)
            
            print(f"💾 Endpoint info saved to: {info_path}")

def manage_endpoints():
    """Interactive endpoint management"""
    manager = ZamAIEndpointManager()
    
    while True:
        print("\n=== ZamAI Endpoint Manager ===")
        print("1. List endpoints")
        print("2. Create endpoint")
        print("3. Check endpoint status")
        print("4. Test endpoint")
        print("5. Delete endpoint")
        print("6. Exit")
        
        choice = input("\nChoose an option (1-6): ").strip()
        
        if choice == '1':
            endpoints = manager.list_endpoints()
            print(f"\nEndpoints: {json.dumps(endpoints, indent=2)}")
        
        elif choice == '2':
            model_id = input("Model ID: ").strip()
            endpoint_name = input("Endpoint name: ").strip()
            manager.create_endpoint(endpoint_name, model_id)
        
        elif choice == '3':
            endpoint_name = input("Endpoint name: ").strip()
            status = manager.get_endpoint_status(endpoint_name)
            print(f"\nStatus: {json.dumps(status, indent=2)}")
        
        elif choice == '4':
            endpoint_url = input("Endpoint URL: ").strip()
            test_prompt = input("Test prompt: ").strip()
            response = manager.test_endpoint(endpoint_url, test_prompt)
            print(f"\nResponse: {response}")
        
        elif choice == '5':
            endpoint_name = input("Endpoint name to delete: ").strip()
            confirm = input(f"Delete '{endpoint_name}'? (y/N): ").strip().lower()
            if confirm == 'y':
                manager.delete_endpoint(endpoint_name)
        
        elif choice == '6':
            break
        
        else:
            print("Invalid choice")

if __name__ == "__main__":
    print("ZamAI Inference Endpoint Deployment")
    print("=" * 40)
    
    choice = input("1. Deploy models\n2. Manage endpoints\nChoose (1-2): ").strip()
    
    if choice == '1':
        deploy_zamai_models()
    elif choice == '2':
        manage_endpoints()
    else:
        print("Invalid choice")
