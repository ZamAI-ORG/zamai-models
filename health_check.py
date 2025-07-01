#!/usr/bin/env python3
"""
ZamAI Pro Models - Health Check & Status Monitor
Check system health, model availability, and service status
"""

import os
import sys
import json
import time
import requests
import subprocess
from pathlib import Path
from typing import Dict, List
import psutil

class ZamAIHealthChecker:
    def __init__(self):
        self.services = {
            "voice_assistant": {"port": 7860, "endpoint": "/"},
            "tutor_bot": {"port": 7861, "endpoint": "/"},
            "business_automation": {"port": 7862, "endpoint": "/"},
            "api_backend": {"port": 8000, "endpoint": "/health"},
            "redis": {"port": 6379, "endpoint": None}
        }
        
        self.models_to_check = [
            "openai/whisper-large-v3",
            "mistralai/Mistral-7B-Instruct-v0.3", 
            "microsoft/Phi-3-mini-4k-instruct"
        ]
    
    def check_system_resources(self) -> Dict:
        """Check system resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "status": "healthy",
                "cpu_usage": f"{cpu_percent:.1f}%",
                "memory_usage": f"{memory.percent:.1f}%",
                "memory_available": f"{memory.available / 1024**3:.2f}GB",
                "disk_usage": f"{disk.percent:.1f}%",
                "disk_free": f"{disk.free / 1024**3:.2f}GB"
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def check_python_environment(self) -> Dict:
        """Check Python environment and dependencies"""
        try:
            import torch
            import transformers
            import gradio
            import fastapi
            
            return {
                "status": "healthy",
                "python_version": sys.version.split()[0],
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "gradio_version": gradio.__version__,
                "fastapi_version": fastapi.__version__,
                "cuda_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        except ImportError as e:
            return {"status": "error", "error": f"Missing dependency: {e}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def check_hf_token(self) -> Dict:
        """Check Hugging Face token validity"""
        try:
            token = os.getenv("HUGGINGFACE_TOKEN")
            if not token and os.path.exists("HF-Token.txt"):
                with open("HF-Token.txt", "r") as f:
                    token = f.read().strip()
            
            if not token:
                return {"status": "error", "error": "No Hugging Face token found"}
            
            # Test token validity
            headers = {"Authorization": f"Bearer {token}"}
            response = requests.get(
                "https://huggingface.co/api/whoami",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                user_info = response.json()
                return {
                    "status": "healthy",
                    "token_valid": True,
                    "username": user_info.get("name", "unknown"),
                    "token_length": len(token)
                }
            else:
                return {"status": "error", "error": "Invalid Hugging Face token"}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def check_service_health(self, service: str, config: Dict) -> Dict:
        """Check individual service health"""
        port = config["port"]
        endpoint = config.get("endpoint", "/")
        
        try:
            if service == "redis":
                # Special check for Redis
                try:
                    import redis
                    r = redis.Redis(host='localhost', port=port, db=0)
                    r.ping()
                    return {"status": "healthy", "port": port, "response_time": "< 1ms"}
                except:
                    return {"status": "down", "port": port, "error": "Redis not accessible"}
            else:
                # HTTP service check
                url = f"http://localhost:{port}{endpoint}"
                start_time = time.time()
                response = requests.get(url, timeout=10)
                response_time = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    return {
                        "status": "healthy",
                        "port": port,
                        "response_time": f"{response_time:.0f}ms",
                        "status_code": response.status_code
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "port": port,
                        "status_code": response.status_code,
                        "error": response.text[:100]
                    }
                    
        except requests.exceptions.ConnectionError:
            return {"status": "down", "port": port, "error": "Service not running"}
        except Exception as e:
            return {"status": "error", "port": port, "error": str(e)}
    
    def check_model_availability(self, model_id: str) -> Dict:
        """Check if a model is available via HF API"""
        try:
            token = os.getenv("HUGGINGFACE_TOKEN")
            if not token and os.path.exists("HF-Token.txt"):
                with open("HF-Token.txt", "r") as f:
                    token = f.read().strip()
            
            if not token:
                return {"status": "error", "error": "No HF token"}
            
            headers = {"Authorization": f"Bearer {token}"}
            
            # Check model info
            info_url = f"https://huggingface.co/api/models/{model_id}"
            info_response = requests.get(info_url, headers=headers, timeout=10)
            
            # Check inference API
            api_url = f"https://api-inference.huggingface.co/models/{model_id}"
            api_response = requests.post(
                api_url,
                headers=headers,
                json={"inputs": "test"},
                timeout=30
            )
            
            model_info = info_response.json() if info_response.status_code == 200 else {}
            
            return {
                "status": "available" if api_response.status_code in [200, 503] else "error",
                "model_id": model_id,
                "api_status_code": api_response.status_code,
                "downloads": model_info.get("downloads", 0),
                "last_modified": model_info.get("lastModified", "unknown"),
                "inference_ready": api_response.status_code == 200
            }
            
        except Exception as e:
            return {"status": "error", "model_id": model_id, "error": str(e)}
    
    def check_docker_status(self) -> Dict:
        """Check Docker and container status"""
        try:
            # Check if Docker is running
            result = subprocess.run(
                ["docker", "version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return {"status": "unavailable", "error": "Docker not running"}
            
            # Check container status
            result = subprocess.run(
                ["docker", "ps", "--format", "table {{.Names}}\\t{{.Status}}"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            containers = []
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")[1:]  # Skip header
                for line in lines:
                    if line:
                        parts = line.split("\t")
                        if len(parts) >= 2:
                            containers.append({
                                "name": parts[0],
                                "status": parts[1]
                            })
            
            return {
                "status": "available",
                "containers": containers,
                "total_containers": len(containers)
            }
            
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": "Docker command timeout"}
        except FileNotFoundError:
            return {"status": "unavailable", "error": "Docker not installed"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_comprehensive_check(self) -> Dict:
        """Run all health checks"""
        print("🏥 Running ZamAI Health Check...")
        
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "overall_status": "healthy",
            "checks": {}
        }
        
        # System resources
        print("📊 Checking system resources...")
        results["checks"]["system"] = self.check_system_resources()
        
        # Python environment
        print("🐍 Checking Python environment...")
        results["checks"]["python"] = self.check_python_environment()
        
        # HF token
        print("🤗 Checking Hugging Face token...")
        results["checks"]["hf_token"] = self.check_hf_token()
        
        # Services
        print("🔧 Checking services...")
        results["checks"]["services"] = {}
        for service, config in self.services.items():
            print(f"  - Checking {service}...")
            results["checks"]["services"][service] = self.check_service_health(service, config)
        
        # Models
        print("🤖 Checking models...")
        results["checks"]["models"] = {}
        for model_id in self.models_to_check:
            print(f"  - Checking {model_id}...")
            results["checks"]["models"][model_id] = self.check_model_availability(model_id)
        
        # Docker
        print("🐳 Checking Docker...")
        results["checks"]["docker"] = self.check_docker_status()
        
        # Determine overall status
        error_count = 0
        for category, checks in results["checks"].items():
            if isinstance(checks, dict):
                if checks.get("status") == "error":
                    error_count += 1
                elif isinstance(checks, dict) and "status" not in checks:
                    # Nested checks (like services, models)
                    for sub_check in checks.values():
                        if isinstance(sub_check, dict) and sub_check.get("status") in ["error", "down", "unhealthy"]:
                            error_count += 1
        
        if error_count == 0:
            results["overall_status"] = "healthy"
        elif error_count < 3:
            results["overall_status"] = "degraded"
        else:
            results["overall_status"] = "unhealthy"
        
        return results
    
    def generate_report(self, results: Dict) -> str:
        """Generate human-readable health report"""
        status_emoji = {
            "healthy": "✅",
            "degraded": "⚠️",
            "unhealthy": "❌",
            "available": "✅",
            "unavailable": "❌",
            "error": "❌",
            "down": "🔴"
        }
        
        report = f"""
# 🏥 ZamAI Pro Models - Health Report

**Generated**: {results['timestamp']}
**Overall Status**: {status_emoji.get(results['overall_status'], '❓')} {results['overall_status'].upper()}

## 📊 System Resources
{status_emoji.get(results['checks']['system']['status'], '❓')} **CPU**: {results['checks']['system'].get('cpu_usage', 'N/A')}
{status_emoji.get(results['checks']['system']['status'], '❓')} **Memory**: {results['checks']['system'].get('memory_usage', 'N/A')} ({results['checks']['system'].get('memory_available', 'N/A')} available)
{status_emoji.get(results['checks']['system']['status'], '❓')} **Disk**: {results['checks']['system'].get('disk_usage', 'N/A')} ({results['checks']['system'].get('disk_free', 'N/A')} free)

## 🐍 Python Environment
{status_emoji.get(results['checks']['python']['status'], '❓')} **Python**: {results['checks']['python'].get('python_version', 'N/A')}
{status_emoji.get(results['checks']['python']['status'], '❓')} **PyTorch**: {results['checks']['python'].get('torch_version', 'N/A')}
{status_emoji.get(results['checks']['python']['status'], '❓')} **CUDA**: {results['checks']['python'].get('cuda_available', False)}
{status_emoji.get(results['checks']['python']['status'], '❓')} **GPUs**: {results['checks']['python'].get('gpu_count', 0)}

## 🤗 Hugging Face
{status_emoji.get(results['checks']['hf_token']['status'], '❓')} **Token**: {"Valid" if results['checks']['hf_token'].get('token_valid') else "Invalid"}
{status_emoji.get(results['checks']['hf_token']['status'], '❓')} **User**: {results['checks']['hf_token'].get('username', 'N/A')}

## 🔧 Services
"""
        
        for service, check in results['checks']['services'].items():
            status = check.get('status', 'unknown')
            port = check.get('port', 'N/A')
            response_time = check.get('response_time', 'N/A')
            
            report += f"{status_emoji.get(status, '❓')} **{service}**: Port {port}"
            if response_time != 'N/A':
                report += f" ({response_time})"
            if status == 'error':
                report += f" - {check.get('error', 'Unknown error')}"
            report += "\n"
        
        report += "\n## 🤖 Models\n"
        
        for model_id, check in results['checks']['models'].items():
            status = check.get('status', 'unknown')
            model_name = model_id.split('/')[-1]
            
            report += f"{status_emoji.get(status, '❓')} **{model_name}**: "
            if status == 'available':
                report += f"Ready (Downloads: {check.get('downloads', 0)})"
            else:
                report += f"Error - {check.get('error', 'Unknown error')}"
            report += "\n"
        
        report += f"\n## 🐳 Docker\n"
        docker_check = results['checks']['docker']
        docker_status = docker_check.get('status', 'unknown')
        report += f"{status_emoji.get(docker_status, '❓')} **Docker**: {docker_status}"
        
        if docker_status == 'available':
            container_count = docker_check.get('total_containers', 0)
            report += f" ({container_count} containers running)"
        elif docker_status == 'error':
            report += f" - {docker_check.get('error', 'Unknown error')}"
        
        return report.strip()

def main():
    """Main function"""
    print("🇦🇫 ZamAI Pro Models - Health Check")
    print("=" * 40)
    
    checker = ZamAIHealthChecker()
    results = checker.run_comprehensive_check()
    
    # Save results
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/health_check.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate report
    report = checker.generate_report(results)
    
    with open("data/processed/health_report.md", "w") as f:
        f.write(report)
    
    print("\n" + "=" * 40)
    print("🏥 HEALTH CHECK COMPLETED")
    print("=" * 40)
    print(report)
    
    # Exit code based on health
    if results["overall_status"] == "healthy":
        sys.exit(0)
    elif results["overall_status"] == "degraded":
        sys.exit(1)
    else:
        sys.exit(2)

if __name__ == "__main__":
    main()
