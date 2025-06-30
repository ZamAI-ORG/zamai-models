#!/usr/bin/env python3
"""
ZamAI Dataset Analysis Script
Analyze your Pashto dataset structure and content
"""

import os
import json
import pandas as pd
from datasets import load_dataset
from huggingface_hub import login
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_zamai_dataset():
    print("🇦🇫 ZamAI Dataset Analysis")
    print("=" * 50)
    
    # Login with HF token
    token_path = "HF-Token.txt"
    if os.path.exists(token_path):
        with open(token_path, 'r') as f:
            token = f.read().strip()
        login(token=token)
        print("🔑 Logged in to Hugging Face")
    
    dataset_name = "tasal9/ZamAI_Pashto_Dataset"
    
    try:
        print(f"📊 Loading dataset: {dataset_name}")
        
        # Load the instruction datasets
        dataset = load_dataset(
            dataset_name,
            data_files={
                "train": "pashto_train_instruction.jsonl",
                "validation": "pashto_val_instruction.jsonl"
            }
        )
        
        train_data = dataset["train"]
        val_data = dataset["validation"]
        
        print(f"✅ Dataset loaded successfully!")
        print(f"📈 Training examples: {len(train_data)}")
        print(f"📈 Validation examples: {len(val_data)}")
        
        # Analyze data structure
        print(f"\n📋 Data Structure:")
        print(f"   Columns: {train_data.column_names}")
        
        # Sample data analysis
        sample = train_data[0]
        print(f"\n📖 Sample Example:")
        for key, value in sample.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"   {key}: {value[:100]}...")
            else:
                print(f"   {key}: {value}")
        
        # Analyze instruction types
        instructions = [example["instruction"] for example in train_data]
        instruction_types = Counter()
        
        for inst in instructions:
            if "translate" in inst.lower():
                instruction_types["Translation"] += 1
            elif "write" in inst.lower():
                instruction_types["Writing"] += 1
            elif "summarize" in inst.lower():
                instruction_types["Summarization"] += 1
            elif "answer" in inst.lower():
                instruction_types["Question Answering"] += 1
            else:
                instruction_types["Other"] += 1
        
        print(f"\n📊 Instruction Types Distribution:")
        for inst_type, count in instruction_types.most_common():
            percentage = (count / len(instructions)) * 100
            print(f"   {inst_type}: {count} ({percentage:.1f}%)")
        
        # Analyze text lengths
        input_lengths = [len(example["input"]) for example in train_data if example["input"]]
        output_lengths = [len(example["output"]) for example in train_data]
        
        print(f"\n📏 Text Length Statistics:")
        print(f"   Input lengths:")
        print(f"     - Average: {sum(input_lengths)/len(input_lengths):.1f} chars")
        print(f"     - Min: {min(input_lengths)} chars")
        print(f"     - Max: {max(input_lengths)} chars")
        
        print(f"   Output lengths:")
        print(f"     - Average: {sum(output_lengths)/len(output_lengths):.1f} chars")
        print(f"     - Min: {min(output_lengths)} chars")
        print(f"     - Max: {max(output_lengths)} chars")
        
        # Language analysis
        pashto_chars = set("آابپتټثجځچحخدډذرړزژسشښصضطظعغفقکګلمنڼوؤهیۍ")
        
        def count_pashto_chars(text):
            return sum(1 for char in text if char in pashto_chars)
        
        pashto_ratio_inputs = []
        pashto_ratio_outputs = []
        
        for example in train_data[:1000]:  # Sample first 1000 for speed
            if example["input"]:
                input_total = len(example["input"])
                input_pashto = count_pashto_chars(example["input"])
                if input_total > 0:
                    pashto_ratio_inputs.append(input_pashto / input_total)
            
            output_total = len(example["output"])
            output_pashto = count_pashto_chars(example["output"])
            if output_total > 0:
                pashto_ratio_outputs.append(output_pashto / output_total)
        
        avg_pashto_input = sum(pashto_ratio_inputs) / len(pashto_ratio_inputs) if pashto_ratio_inputs else 0
        avg_pashto_output = sum(pashto_ratio_outputs) / len(pashto_ratio_outputs) if pashto_ratio_outputs else 0
        
        print(f"\n🇦🇫 Pashto Content Analysis (sample of 1000):")
        print(f"   Average Pashto ratio in inputs: {avg_pashto_input:.2%}")
        print(f"   Average Pashto ratio in outputs: {avg_pashto_output:.2%}")
        
        # Save analysis results
        analysis_results = {
            "dataset_name": dataset_name,
            "total_examples": {
                "train": len(train_data),
                "validation": len(val_data)
            },
            "instruction_types": dict(instruction_types),
            "text_statistics": {
                "avg_input_length": sum(input_lengths)/len(input_lengths) if input_lengths else 0,
                "avg_output_length": sum(output_lengths)/len(output_lengths),
                "max_input_length": max(input_lengths) if input_lengths else 0,
                "max_output_length": max(output_lengths)
            },
            "pashto_content": {
                "avg_pashto_ratio_input": avg_pashto_input,
                "avg_pashto_ratio_output": avg_pashto_output
            },
            "sample_example": sample
        }
        
        with open("dataset_analysis_results.json", 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Analysis saved to: dataset_analysis_results.json")
        print(f"\n✅ Dataset analysis completed!")
        
        return analysis_results
        
    except Exception as e:
        print(f"❌ Error analyzing dataset: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    analyze_zamai_dataset()
