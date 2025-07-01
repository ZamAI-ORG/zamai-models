#!/usr/bin/env python3
"""
Test the Phi-3 Mini Pashto model for basic Pashto language generation.
"""

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
import json
import os

def test_phi3_mini_pashto(model_id, prompt, max_length=200):
    print(f"🔄 Loading model: {model_id}")
    start_time = time.time()
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    load_time = time.time() - start_time
    print(f"✅ Model loaded in {load_time:.2f} seconds on {device}")
    
    # Prepare input
    print(f"📝 Using prompt: '{prompt}'")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate text
    print("🚀 Generating text...")
    generation_start = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    generation_time = time.time() - generation_start
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"\n=== Generated Text (in {generation_time:.2f}s) ===")
    print(generated_text)
    print("=====================================\n")
    
    # Save results
    os.makedirs("data/processed", exist_ok=True)
    results = {
        "model": model_id,
        "device": device,
        "load_time_seconds": round(load_time, 2),
        "generation_time_seconds": round(generation_time, 2),
        "prompt": prompt,
        "generated_text": generated_text
    }
    
    with open("data/processed/phi3_mini_pashto_test.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Results saved to data/processed/phi3_mini_pashto_test.json")
    return results

def main():
    parser = argparse.ArgumentParser(description="Test the Phi-3 Mini Pashto model")
    parser.add_argument("--model", type=str, default="tasal9/ZamAI-Phi-3-Mini-Pashto", 
                        help="Model ID to test (default: tasal9/ZamAI-Phi-3-Mini-Pashto)")
    parser.add_argument("--prompt", type=str, 
                        default="په پښتو کې ماته وایاست چې افغانستان څه ډول هیواد دی؟", 
                        help="Prompt to use for generation")
    parser.add_argument("--max-length", type=int, default=200,
                        help="Maximum length of generated text")
    
    args = parser.parse_args()
    test_phi3_mini_pashto(args.model, args.prompt, args.max_length)

if __name__ == "__main__":
    main()
