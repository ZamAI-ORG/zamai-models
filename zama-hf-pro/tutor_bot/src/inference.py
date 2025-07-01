from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

class PashtoTutor:
    def __init__(self, model_name="tasal9/pashto-tutor-bot"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    def generate_response(self, question):
        """Generate educational response"""
        prompt = f"Q: {question}\nA:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("A:")[-1].strip()
