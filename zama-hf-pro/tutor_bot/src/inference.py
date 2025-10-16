import requests
import os

class ZamAITutorInference:
    def __init__(self):
        self.token = os.getenv("HUGGINGFACE_TOKEN", "")
        self.api_url = "https://api-inference.huggingface.co/models"
        self.headers = {"Authorization": f"Bearer {self.token}"}
        self.demo_mode = not self.token  # Enable demo mode if no token
    
    def generate_response(self, prompt: str, model: str = "tasal9/ZamAI-Tutor-Bot"):
        """Generate educational response using HF Inference API or demo"""
        
        if self.demo_mode:
            # Demo responses for testing
            demo_responses = {
                "ریاضی": "دا د ریاضیاتو یوه ښه پوښتنه ده. په ریاضیاتو کې...",
                "ساینس": "ساینس زموږ د ژوند مهمه برخه ده. دا علم...",
                "تاریخ": "د افغانستان تاریخ ډېر بډایه او اوږد دی...",
                "default": "ښه پوښتنه! زه دلته د مرستې لپاره یم. نور پوښتنې وکړئ."
            }
            
            # Simple keyword matching for demo
            for keyword, response in demo_responses.items():
                if keyword in prompt:
                    return response
            return demo_responses["default"]
        
        try:
            response = requests.post(
                f"{self.api_url}/{model}",
                headers=self.headers,
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 200,
                        "temperature": 0.7,
                        "do_sample": True
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "").replace(prompt, "").strip()
                return "د دې پوښتنې ځواب نشم ورکولی"
            else:
                return f"د سرویس خطا: {response.status_code}"
                
        except Exception as e:
            return f"خطا: {str(e)}"
